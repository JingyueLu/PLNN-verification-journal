import torch
import copy

from plnn.branch_and_bound import pick_out, add_domain, prune_domains
from torch import nn
import random
from plnn.kw_score_conv import choose_node_conv, choose_node_perturbed
from exp_utils.plnn_utils import dump_domain, dom_to_branch
import pickle
import glob
import os
import numpy as np
import time


dom_path = '/home/jodie/PLNN/PLNN-verification-journal/cifar_kw_m2_train_data/'

class ReLUDomain:
    '''
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.

    The domain is specified by `mask` which corresponds to a pattern of ReLUs.
    Neurons mapping to a  0 value are assumed to always have negative input (0 output slope)
          "               1                    "             positive input (1 output slope).
          "               -1 value are considered free and have no assumptions.

    For a MaxPooling unit, -1 indicates that we haven't picked a dominating input
    Otherwise, this indicates which one is the dominant one
    '''
    #def __init__(self, mask,  lower_bound=-float('inf'), upper_bound=float('inf'), lower_all=None, upper_all = None, ub_point=None, dual_vars = None, dual_vars_other=None, primals=None):
    #def __init__(self,mask,lower_bound=-float('inf'), upper_bound=float('inf'), lower_all=None, upper_all=None, dom_name =None):
    def __init__(self, lower_bound=-float('inf'), upper_bound=float('inf'), dom_name = None ):
        #self.mask = mask
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        #self.lower_all = lower_all
        #self.upper_all = upper_all
        self.dom_name = dom_name
        #self.ub_point = ub_point
        #self.dual_vars = dual_vars
        #self.dual_vars_other = dual_vars_other
        #self.primals = primals
     


    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound



def relu_stab(net, domain, bounded, x, ball_eps, dump_trace, eps=1e-4, pgd_threshold = 1,  sparsest_layer=0, decision_bound=None, criteria='kw', record=None):
    '''
    Uses branch and bound algorithm to evaluate the global minimum
    of a given neural network.
    `net`           : Neural Network class, defining the `get_upper_bound` and
                      `get_lower_bound` functions, supporting the `mask` argument
                      indicating the phase of the ReLU.
    `eps`           : Maximum difference between the UB and LB over the minimum
                      before we consider having converged
    `decision_bound`: If not None, stop the search if the UB and LB are both
                      superior or both inferior to this value.
    `pgd_threshold` : Once the number of relus being fixed during the algorithm
                      is above pdg_threshold percentage of the total ambiguous nodes
                      at the beginning, we initiate pgd attacks to find 
                      a better global upper bound

    Returns         : Lower bound and Upper bound on the global minimum,
                      as well as the point where the upper bound is achieved
    '''
    nb_visited_states = 0
    #global_ub_point, global_ub = net.get_upper_bound(domain)
    global_ub, global_lb, global_ub_point, dual_vars, dual_vars_other, primals, updated_mask, lower_bounds, upper_bounds, pre_relu_indices= net.build_the_model(domain, x, ball_eps, bounded)
    print('kw ball is bounded: ', bounded)
    print(global_lb)
    if global_lb > 0 :
        out = 'UNSAT'
        print('Early Stop, No need for BaB')
        return global_lb, global_ub,  nb_visited_states, out

    dom_idx = 0
    dom_name = dom_path+f'domain_{criteria}/'+dump_trace + f'_dom_{dom_idx}'
    candidate_domain = ReLUDomain(lower_bound=global_lb, upper_bound = global_ub, dom_name = dom_name)
    domains = [candidate_domain]
    dump_domain(dom_name,
                updated_mask,
                global_lb,
                global_ub,
                lower_bounds,
                upper_bounds,
                dual_vars,
                dual_vars_other,
                global_ub_point,
                primals
                ) 
    
    #import pdb; pdb.set_trace()
    prune_counter = 0
    icp_score = 0

    random_order = list(range(len(updated_mask)))
    try: 
        random_order.remove(sparsest_layer)
        random_order = [sparsest_layer]+random_order
    except:
        pass

    while global_ub - global_lb > eps:
        # Pick a domain to branch over and remove that from our current list of
        # domains. Also, potentially perform some pruning on the way.
        candidate_domain = pick_out(domains, global_ub - eps)
        # Generate new, smaller domains by splitting over a ReLU
        dom_name = candidate_domain.dom_name
        with open(dom_name,"rb") as f:
            dc = pickle.load(f)
        mask = dc['X']['mask']
        orig_lbs = dc['X']['lower_bounds']
        orig_ubs = dc['X']['upper_bounds'] 
        f.close()
        # debug choose_node_conv
        #mask[2] = torch.zeros(mask[2].shape)
        #icp_score =2
        
        #### return best kw, a perturbed kw, and list of choices
        kw_decision, perturbed_decision, pool_size, icp_score, score = choose_node_perturbed(orig_lbs, orig_ubs, mask, net.layers, pre_relu_indices, icp_score, random_order, sparsest_layer, gt =True)

        if criteria == 'kw':
            branching_decision = kw_decision
        elif criteria == 'kw_perturbed':
            branching_decision = perturbed_decision
        elif criteria.split('_')[0] =='maxmin' or criteria.split('_')[0] =='maxsum':
            selected_indices = testing_indices(mask, score)
            print('start computing ground truth')
            start = time.time()
            branching_decision, pool_size = gt_split(net, dump_trace, dom_name, nb_visited_states, mask, orig_lbs, orig_ubs, global_lb, selected_indices, criteria)
            end = time.time()
            print('one branch gt computation requires: ', end-start)
        
        print('branching decision: ', branching_decision)        
        print('total available perturbed options: ', pool_size)


        for choice in [0,1]:
            # Find the upper and lower bounds on the minimum in the domain
            # defined by n_mask_i
            nb_visited_states += 1
            if (nb_visited_states % 10) == 0:
                print(f"Running Nb states visited: {nb_visited_states}")
            
            mask_temp = [i.clone() for i in mask]
            dom_ub,dom_lb, dom_ub_point, dom_dual_vars, dom_dual_vars_other, dom_primals, updated_mask, dom_lb_all, dom_ub_all= net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, branching_decision, choice)
            
            if dom_ub < global_ub:
                global_ub = dom_ub
            

            print('dom_lb: ', dom_lb)
            print('dom_ub: ', dom_ub)

            if dom_lb < global_ub:
                dom_idx += 1
                dom_name = dom_path+f'domain_{criteria}/'+dump_trace + f'_dom_{dom_idx}'
                dom_to_add = ReLUDomain(lower_bound = dom_lb,
                                        upper_bound = dom_ub,
                                        dom_name = dom_name)
                                        
                                        
                dump_domain(dom_name,
                            updated_mask,
                            dom_lb,
                            dom_ub,
                            dom_lb_all,
                            dom_ub_all,
                            dom_dual_vars,
                            dom_dual_vars_other,
                            dom_ub_point,
                            dom_primals
                            ) 
                add_domain(dom_to_add, domains)
                prune_counter += 1

            if nb_visited_states > 1500:
                record.write('early termination\n')
                record.write(f'global_lb {global_lb}\n')
                record.write(f'global_ub {global_ub}\n')
                record.write(f'nb_states {nb_visited_states}\n')
                record.close()
                return global_lb, global_ub, nb_visited_states, True
        

        domains = prune_domains(domains, global_ub - eps)

        prune_counter = 0


        if len(domains) > 0:
            global_lb = domains[0].lower_bound
        else:
            # If there is no more domains, we have pruned them all
            global_lb = global_ub - eps

        print(f"Current: lb:{global_lb}\t ub: {global_ub}")
        if record is not None:
            record.write('glb {:.4f} branching decision {} choices {:d}\n'.format(global_lb, branching_decision, pool_size))
            record.flush()


        # Stopping criterion
        if (decision_bound is not None) and (global_lb >= decision_bound):
            break
        elif global_ub < decision_bound:
            break
        
    if global_ub < 0:
        out = 'SAT'
    elif global_lb >0:
        out = 'UNSAT'
    else:
        out = 'None'

    if record is not None:
        record.write(f'{out}\n')
        record.write(f'global_lb {global_lb}\n')
        record.write(f'global_ub {global_ub}\n')
        record.write(f'nb_states {nb_visited_states}\n')
        record.close()
   
    dom_name = dom_path+f'domain_{criteria}/'+dump_trace + '_dom*'
    files = glob.glob(dom_name)
    for i in files:
        os.remove(i)
    return global_lb, global_ub,  nb_visited_states, out



def testing_indices(mask, score):
    '''
    select a representative subset of indices of the set of all available unfixed relu choices
    1. ensure at least 10% coverage 34+15+2
    2. include the top 40 kw choices (with preference giving to layer 1 and layer 2)
    =====> only need to augment the choices on layer 0
    '''
    selected_indices = {}
    for i in range(len(mask)):
        selected_indices[i] = []
    new_score = {}
    new_score_l2 = {}
    new_score_l1 = {}
    for i in range(len(score)):
        for j in range(len(score[i])):
            if mask[i][j] == -1:
                new_score[f'relu_{i}_{j}'] = score[i][j].item()
                if (i==1):
                    new_score_l1[f'relu_{i}_{j}'] = score[i][j].item()
                if (i==2):
                    new_score_l2[f'relu_{i}_{j}'] = score[i][j].item()

    
    new_score = sorted(new_score.items(), key = lambda x : x[1])
    new_score_l1 = sorted(new_score_l1.items(), key = lambda x : x[1])
    new_score_l2 = sorted(new_score_l2.items(), key = lambda x : x[1])
    new_score.reverse()
    new_score_l1.reverse()
    new_score_l2.reverse()
    kw_choices = new_score[:60]+new_score_l1[:20]+new_score_l2[:20]
    for i in set(kw_choices):
        selected_indices[int(i[0].split('_')[1])].append(int(i[0].split('_')[2]))

    for relu_idx in range(len(mask)-1, -1, -1):
        all_available_choices = ((mask[relu_idx]==-1).nonzero().view(-1)).tolist()
        required_number = int(len(all_available_choices)*0.1)
        done_choices = selected_indices[relu_idx]
        required_number = required_number - len(done_choices)
        ## DEBUG
        # if len(done_choices) == 0:
        if required_number <= 0:
            # No need to add points on this layer
            continue
        else:
            remained_choices = np.setdiff1d(all_available_choices, done_choices)
            selected_choices = np.random.choice(remained_choices, required_number, replace=False)
            selected_indices[relu_idx].extend(selected_choices)

    print(selected_indices) 
    return selected_indices




def kw_split(net, candidate_domain):
    mask = candidate_domain.mask
    orig_lbs = candidate_domain.lower_all_pa
    orig_ubs = candidate_domain.upper_all_pa
    decision = choose_dim(orig_lbs, orig_ubs, mask, net.layers)
    mask_temp_1 = [i.copy() for i in mask]
    mask_temp_1[decision[0]][decision[1]]= 0
    mask_temp_2 = [i.copy() for i in mask]
    mask_temp_2[decision[0]][decision[1]]= 1
    print(f'idx: {decision}')
    all_new_masks = [mask_temp_1, mask_temp_2]
    return all_new_masks


def gt_split(net, dump_trace, dom_name, nb_visited_states,  mask, lower_bounds, upper_bounds, global_lb, selected_indices, criteria):

    #compute ground truths for domain_splits
    gt_score_relu = {}
    gt_lb_relu = {}


    # first get all interest choices
    for layer in range(len(selected_indices)):
        for index in selected_indices[layer]:
            try:
                mask_temp = [i.clone() for i in mask]
                _, dom_lb, _, _, _,_,_,_,_ = net.get_lower_bound( mask_temp, lower_bounds, upper_bounds, [layer, index], 0)

                mask_temp = [i.clone() for i in mask]
                _,dom_lb1,_, _,_,_, _, _, _ = net.get_lower_bound( mask_temp, lower_bounds, upper_bounds, [layer, index], 1)

                lbs = torch.Tensor([dom_lb, dom_lb1])
                print('decision: ', layer, index)
                if criteria.split('_')[0] == 'maxsum':
                    lowest_lb_relu = min(0, dom_lb) + min(0, dom_lb1) - 2*global_lb
                    gt_score_relu[f'relu_{layer}_{index}'] = lowest_lb_relu
                    print(f'sub lower bounds [{dom_lb}, {dom_lb1}]')
                    print(f'maxsum score: {lowest_lb_relu}')

                
                elif criteria.split('_')[0] == 'maxmin':
                    lowest_lb_relu = min(min(0, dom_lb), min(0, dom_lb1))-global_lb
                    gt_score_relu[f'relu_{layer}_{index}'] = lowest_lb_relu
                    print(f'current choice [{layer}, {index}]')
                    print(f'sub lower bounds [{dom_lb}, {dom_lb1}]')
                    print(f'maxmin score: {lowest_lb_relu}')

            except NotImplementedError:
                continue

            try:
                gt_lb_relu[layer][index] = lbs
            except KeyError:
                gt_lb_relu[layer]={}
                gt_lb_relu[layer][index] = lbs

    # choose a requied decision

    gt_score_relu = sorted(gt_score_relu.items(), key = lambda x : x[1])
    gt_score_relu.reverse()


    if criteria.split('_')[-1] == 'perturbed':
        candidates = []
        best_score = gt_score_relu[0][1]
        for i in gt_score_relu:
            if i[1]/best_score >= 0.8:
                candidates.append(i[0])
            else:
                break
        pool_size = len(candidates)
        temp = random.choice(candidates)
        branching_decision = [int(temp.split('_')[1]), int(temp.split('_')[-1])]

    else:
        pool_size = -1
        branching_decision = [int(gt_score_relu[0][0].split('_')[1]), int(gt_score_relu[0][0].split('_')[-1])]
        
    if dump_trace is not None:
        trace_fname = dom_path+ f'branch_{criteria}/'+dump_trace + '_branch_{}'.format(nb_visited_states)
        print("\n",trace_fname,"\n")

        dom_to_branch(trace_fname,
                         dom_name,
                         gt_lb_relu,
                         branching_decision,
                         ) 

    return branching_decision, pool_size 



def gt_split_kw(net, candidate_domain, kw_decision, kw_indices, nb_visited_states, dump_trace):
    mask = candidate_domain.mask
    orig_lbs = candidate_domain.lower_all
    orig_ubs = candidate_domain.upper_all
    largest_lowest_lb_dom = -float('inf')
    largest_lowest_lb_dim_dom = None
    largest_lowest_lb_relu = -float('inf')
    largest_lowest_lb_index_relu = None
    largest_lowest_lb_layer_relu = None
    records_relu = {}
    gt_lb_relu = {}

    ##compute ground truths for relu_splits

    # first get all interest choices
    for decision in kw_indices:
        mask_temp = [i.clone() for i in mask]
        dom_ub,dom_lb, dom_ub_point, updated_mask, dom_lb_all, dom_ub_all= net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, decision, 0)

        mask_temp = [i.clone() for i in mask]
        dom_ub1,dom_lb1, dom_ub_point1, updated_mask1, dom_lb_all1, dom_ub_all1= net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, decision, 1)

        lbs = torch.Tensor([dom_lb, dom_lb1])
        try:
            gt_lb_relu[decision[0]][decision[1]] = lbs
        except KeyError:
            gt_lb_relu[decision[0]]={}
            gt_lb_relu[decision[0]][decision[1]] = lbs

        print(f'idx: {decision}  solutions: {lbs}')

        if decision == kw_decision:
            records_relu["dom_ub"] =[dom_ub,dom_ub1]
            records_relu["dom_lb"] = [dom_lb, dom_lb1]
            records_relu["dom_ub_point"] = [dom_ub_point, dom_ub_point1]
            records_relu["mask"] = [updated_mask, updated_mask1]
            records_relu["dom_lb_all"] = [dom_lb_all, dom_lb_all1]
            records_relu["dom_ub_all"] = [dom_ub_all, dom_ub_all1]

    # dump traces
    if dump_trace is not None:

        trace_fname = dump_trace + '_branch_{}'.format(nb_visited_states)
        print("\n",trace_fname,"\n")

    dec = kw_decision
    print(f'final decision {dec}')

    if dump_trace is not None:
        dump_relu_problem(trace_fname,
                         candidate_domain.mask,
                         candidate_domain.lower_bound,
                         candidate_domain.upper_bound,
                         candidate_domain.lower_all,
                         candidate_domain.upper_all,
                         gt_lb_relu,
                         dec) 
    return records_relu 


def relu_split(layers, mask):
    '''
    Given a mask that defines a domain, split it according to a non-linerarity.

    The non-linearity is chosen to be as early as possible in the network, but
    this is just a heuristic.

    `layers`: list of layers in the network. Allows us to distinguish
              Maxpooling and ReLUs
    `mask`: A list of [list of {-1, 0, 1}] where each elements corresponds to a layer,
            giving constraints on the Neuron.
    Returns: A list of masks, in the same format

    '''
    done_split = False
    non_lin_layer_idx = 0
    all_new_masks = []
    for layer_idx, layer in enumerate(layers):
        if type(layer) in [nn.ReLU, nn.MaxPool1d]:
            non_lin_lay_mask = mask[non_lin_layer_idx]
            if done_split:
                # We have done our split, so no need for any additional split
                # -> Pass along all of the stuff
                for new_mask in all_new_masks:
                    new_mask.append(non_lin_lay_mask)
            elif all([neuron_dec != -1 for neuron_dec in non_lin_lay_mask]):
                # All the neuron in this layer have already an assumption.
                # This will just be passed along when we do our split.
                pass
            else:
                # This is the first layer we encounter that is not completely
                # assumed so we will take the first "undecided" neuron and
                # split on it.

                # Start by making two copies of everything that came before
                if type(layer) is nn.ReLU:
                    all_new_masks.append([])
                    all_new_masks.append([])
                elif type(layer) is nn.MaxPool1d:
                    for _ in range(layer.kernel_size):
                        all_new_masks.append([])
                else:
                    raise NotImplementedError

                for prev_lay_mask in mask[:non_lin_layer_idx]:
                    for new_mask in all_new_masks:
                        new_mask.append(prev_lay_mask)

                # Now, deal with the layer that we are actually splitting
                neuron_to_flip = non_lin_lay_mask.index(-1)
                for choice, new_mask in enumerate(all_new_masks):
                    # choice will be 0,1 for ReLU
                    # it will be 0, .. kernel_size-1 for MaxPool1d
                    mod_layer = non_lin_lay_mask[:]
                    mod_layer[neuron_to_flip] = choice
                    new_mask.append(mod_layer)

                done_split = True
            non_lin_layer_idx += 1
    for new_mask in all_new_masks:
        assert len(new_mask) == len(mask)
    if not done_split:
        all_new_masks = [mask]
    return all_new_masks
