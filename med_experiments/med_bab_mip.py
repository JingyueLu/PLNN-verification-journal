#!/usr/bin/env python
import argparse
#from plnn.relu_bnb_com import com_bab
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn 
import torch
from exp_utils.model_utils import load_cifar_1to1_exp, load_1to1_exp
from plnn.conv_kwinter_kw import KWConvNetwork
from plnn.mip_solver import MIPNetwork
from plnn.relu_conv_any_kw import relu_bab
import time
import pandas as pd
import multiprocessing
import os

'''
This script supports MIP and Branch-and-Bound with ReLU splitting verifications.

19/11/19
added support for MNIST dataset
'''

def gurobi(verif_layers, domain,  return_dict):
    mip_network = MIPNetwork(verif_layers)
    #mip_binary_network.setup_model(inp_domain, x=x.view(1, -1), ball_eps = eps_temp, bounds=bounds)
    mip_network.setup_model(domain, use_obj_function=True, bounds="interval-kw")
    #mip_network.setup_model(domain, use_obj_function=False, bounds="interval-kw")
    sat, solution, nb_states = mip_network.solve(domain)
    return_dict["out"] = sat  
    return_dict["nb_states"] = nb_states  


def bab(verif_layers, domain, x, eps_temp, branching,linear, model_name, bounded, return_dict):
    epsilon=1e-4 
    decision_bound=0
    pgd_threshold = 1.
    network = KWConvNetwork(verif_layers)
    min_lb, min_ub, ub_point, nb_states = relu_bab(network, domain, x, eps_temp, epsilon, pgd_threshold=pgd_threshold, split_decision =branching, decision_bound=decision_bound, linear=linear, model_path = model_name, bounded = bounded)
    return_dict["min_lb"] = min_lb  
    return_dict["min_ub"] = min_ub  
    return_dict["ub_point"] = ub_point  
    return_dict["nb_states"] = nb_states  


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--cpus_total', type = int, help='total number of cpus used')
    parser.add_argument('--cpu_id', type=int, help='the index of the cpu from 0 to cpus_total')
    parser.add_argument('--pdprops', type =str, help='pandas table with all props we are interested in')
    #parser.add_argument('--branching',  type=str, choices = ['kw', 'graph'] )
    parser.add_argument('--data', type=str)
    parser.add_argument('--bab_kw', action='store_true')
    parser.add_argument('--gurobi', action='store_true')
    parser.add_argument('--testing', action='store_true')
    args = parser.parse_args()

    method_name = ''
    if args.bab_kw is True:
        method_name +='_KW_'
    if args.gurobi is True:
        method_name +='_GRB'

    # initialize a file to record all results, record should be a pandas dataframe
    path = ''
    if args.data == 'cifar':
        path = path + 'cifar_exp/'
    elif args.data == 'mnist':
        path = path + 'mnist_exp/'
    else:
        raise NotImplementedError

    # load all properties
    gt_results = pd.read_pickle(path + args.pdprops)
    bnb_ids = gt_results.index
    assert args.cpu_id < args.cpus_total, 'cpu index exceeds total cpus available'
    batch_size = len(bnb_ids)//args.cpus_total +1
    start_id = args.cpu_id*batch_size
    end_id = min(len(bnb_ids), start_id+batch_size)
    batch_ids = bnb_ids[start_id: end_id]
    #import pdb; pdb.set_trace()


    if args.record:
        record_name = path + f'results/{args.pdprops[:-4]}_{method_name}_{args.cpu_id}.pkl'
        if os.path.isfile(record_name):
            graph_df = pd.read_pickle(record_name)
        else:
            indices= list(range(len(batch_ids)))
            

            columns = ["Idx", "Eps", "prop",  
                     "BSAT_KW", "BBran_KW" ,"BTime_KW",
                        "GSAT", "GTime"]


            graph_df = pd.DataFrame(index = indices, columns=columns)
            graph_df.to_pickle(record_name) 

    for new_idx, idx in enumerate(batch_ids):
        

        # record_info 
        if args.record:
            graph_df = pd.read_pickle(record_name)
            if pd.isna(graph_df.loc[new_idx]["Eps"])==False:
                print(f'the {new_idx}th element is done')
                continue

        imag_idx = gt_results.loc[idx]["Idx"]
        prop_idx = gt_results.loc[idx]['prop']
        eps_temp = gt_results.loc[idx]["Eps"]


        x, verif_layers, test_idx = load_cifar_1to1_exp("cifar_kw_m2", imag_idx, prop_idx)
        assert test_idx == prop_idx

        #if prop_idx is None:
        #    print('At ',imag_idx, ' model prediction is incorrect\n\n')
        #    continue

        bounded = False
        domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)



        linear=False

        if args.testing:
            epsilon=1e-4 
            decision_bound=0
            pgd_threshold = 1
            network = KWConvNetwork(verif_layers)
            trace_name = None
            #model_name = models[args.model_name]
            model_name = None
            test_start = time.time()
            min_lb, min_ub, ub_point, nb_states = relu_bab(network, domain, x, args.epslion, bounded, epsilon, pgd_threshold=pgd_threshold, split_decision =args.branching, decision_bound=decision_bound, linear=linear, model_path = model_name, dump_trace = trace_name)
            test_end = time.time()
            test_total = test_end -  test_start
            print('total testing time: ', test_total)
            import pdb; pdb.set_trace()



        ### BaB
        if args.bab_kw:
            gt_prop = f'idx_{imag_idx}_prop_{prop_idx}_eps_{eps_temp}'
            print(gt_prop)
            kw_start = time.time()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(target = bab, args=(verif_layers, domain, x, eps_temp, 'kw', linear, None, bounded, return_dict))
            p.start()
            p.join(args.timeout)
            if p.is_alive():
                print("BNB Timeouti\n\n")
                p.terminate()
                p.join()
                kw_min_lb = None; kw_min_ub = None; kw_ub_point = None; kw_nb_states= None
                kw_out="timeout"
            else:
                kw_min_lb = return_dict["min_lb"] 
                kw_min_ub = return_dict["min_ub"] 
                kw_ub_point = return_dict["ub_point"] 
                kw_nb_states = return_dict["nb_states"] 
                if kw_min_lb >= 0:
                    print("UNSAT")
                    kw_out = "False"
                elif kw_min_ub < 0:
                    # Verify that it is a valid solution
                    print("SAT")
                    kw_out = "True"
                else:
                    print("Unknown")
                    import pdb; pdb.set_trace()
                print(f"Nb states visited: {kw_nb_states}")
                #print('bnb takes: ', bnb_time)
                print('\n')
            #except KeyboardInterrupt:
            #    return
            #except:
            #    min_lb = None; min_ub = None; ub_point = None; nb_states= None
            #    out='Error'

            kw_end = time.time()
            kw_time = kw_end - kw_start
            print('total time required: ', kw_time)

            print('\n')


        if args.gurobi:
            gt_prop = f'idx_{imag_idx}_prop_{prop_idx}_eps_{eps_temp}'
            print(gt_prop)
            guro_start = time.time()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            #try:
            p = multiprocessing.Process(target = gurobi, args=(verif_layers, domain, return_dict))
            p.start()
            p.join(args.timeout)
            if p.is_alive():
                print("gurobi Timeout")
                p.terminate()
                p.join()
                guro_out="timeout"
            else:
                guro_out  = return_dict["out"] 
                guro_nb_states = return_dict["nb_states"] 
            #except KeyboardInterrupt:
            #    return
            #except:
            #    min_lb = None; min_ub = None; ub_point = None; nb_states= None
            #    out='Error'

            guro_end = time.time()
            guro_time = guro_end - guro_start
            print('total time required: ', guro_time)
            print('results: ', guro_out)

            print('\n')



        if args.record:  
            graph_df.loc[new_idx]["Idx"] = imag_idx 
            graph_df.loc[new_idx]["Eps"] = eps_temp 
            graph_df.loc[new_idx]["prop"] = prop_idx
            
            if args.bab_kw is True:
                graph_df.loc[new_idx]["BSAT_KW"] = kw_out
                graph_df.loc[new_idx]["BBran_KW"] = kw_nb_states
                graph_df.loc[new_idx]["BTime_KW"] = kw_time


            if args.gurobi is True:
                graph_df.loc[new_idx]["GSAT"] = guro_out
                graph_df.loc[new_idx]["GTime"] = guro_time

            graph_df.to_pickle(record_name) 


if __name__ == '__main__':
    main()
