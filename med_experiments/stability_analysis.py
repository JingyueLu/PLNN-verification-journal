import argparse
#from plnn.relu_bnb_com import com_bab
from torch import nn 
import torch
from plnn.conv_kwinter_gen import KWConvGen
from plnn.relu_conv_any_kw import relu_bab
from plnn.relu_stability import relu_stab
from exp_utils.model_utils import load_med_exp, load_1to1_exp, load_cifar_1to1_exp
import random
import pickle
import os
import pandas as pd
import multiprocessing
import time
import glob

'''
This script is initially created for stability test but now is mainly used 
for generating training datasets
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type = int)
    parser.add_argument('--eps', type = float)
    parser.add_argument('--prop_idx', type=int)
    parser.add_argument('--prop', type=str)
    parser.add_argument('--criteria', type = str, choices=['kw', 'kw_perturbed', 'maxmin_perturbed', 
                                                           'maxmin', 'maxsum', 'maxsum_perturbed'])
    parser.add_argument('--record', action = 'store_true', help='whether to record results')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--max_branches', type=int)
    args = parser.parse_args()

    # build model and domain
    if args.prop is None:
        imag_idx = args.idx
        eps_temp = args.eps
        prop_idx = args.prop_idx
    else:
        temp = args.prop.split('_')
        imag_idx = int(temp[-5])
        prop_idx = int(temp[-3])
        eps_temp = float(temp[-1])
        ####
    
    x, verif_layers, test = load_cifar_1to1_exp('cifar_kw_m2',imag_idx, prop_idx)
    bounded = False
    assert test == prop_idx
    domain = torch.stack([x.squeeze(0) - eps_temp,x.squeeze(0) + eps_temp], dim=-1)

    if args.record:
        record_name = f'/gen_train_results/m2cifar/{args.criteria}_idx_{imag_idx}_prop_{prop_idx}_eps_{eps_temp}.txt'

        results = open(record_name, 'w')
    else:
        results =None


    print(f'verifying image {imag_idx} at inf_ball radius {eps_temp} using {args.criteria}\n')


    if args.testing:
        bnb_start = time.time()
        epsilon=1e-4 
        decision_bound=0
        pgd_threshold = 1
        network = KWConvGen(verif_layers)

        model_path = None

        min_lb, min_ub, ub_point, nb_states = relu_bab(network, domain, x, eps_temp,  epsilon, pgd_threshold=pgd_threshold, split_decision ='kw', decision_bound=decision_bound, model_path = model_path)
        print(f'finished with total number of states: {nb_states}')
        import pdb; pdb.set_trace()


    epsilon=1e-4 
    decision_bound=0
    network = KWConvGen(verif_layers)

    trace_name = f"idx_{imag_idx}_prop_{prop_idx}_ball_{eps_temp}"
    min_lb, min_ub,  nb_states, out= relu_stab(network, domain, bounded,  x, eps_temp, trace_name, epsilon, pgd_threshold=1,  decision_bound=decision_bound, criteria = args.criteria, record=results)
    print(f'nb states {nb_states}\n')
    print(f'out {out}\n')





if __name__ == '__main__':
    main()

