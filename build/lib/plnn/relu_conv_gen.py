import torch
import copy

from plnn.branch_and_bound import pick_out, add_domain, prune_domains
from torch import nn
from plnn.kw_score_conv import choose_node_conv
from dataset_tools.utils import dump_relu_problem


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
    def __init__(self, mask,  lb=-float('inf'), ub=float('inf'), lb_all=None, up_all = None, dual_info = None):
        self.mask = mask
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
        self.dual_info = dual_info
     


    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound


def relu_bab(net, domain,  x, ball_eps, eps=1e-4, decision_bound=None, dump_trace=None):
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

    global_ub, global_lb, global_ub_point, updated_mask, lower_bounds, upper_bounds, pre_relu_indices, dual_info= net.build_the_model(domain, x, ball_eps)
    print(global_lb)
    candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub, lower_bounds, upper_bounds, dual_info)
    domains = [candidate_domain]
    tot_ambi_nodes = 0
    for layer_mask in updated_mask: 
        tot_ambi_nodes += torch.sum(layer_mask ==-1).item()
    

    #import pdb; pdb.set_trace()
    prune_counter = 0
    icp_score = 0

    while global_ub - global_lb > eps:
        # Pick a domain to branch over and remove that from our current list of
        # domains. Also, potentially perform some pruning on the way.
        candidate_domain = pick_out(domains, global_ub - eps)
        # Generate new, smaller domains by splitting over a ReLU
        mask = candidate_domain.mask
        orig_lbs = candidate_domain.lower_all
        orig_ubs = candidate_domain.upper_all
        orig_dual = candidate_domain.dual_info
        # debug choose_node_conv
        #mask[2] = torch.zeros(mask[2].shape)
        #icp_score =2
        kw_decision, icp_score, score = choose_node_conv(orig_lbs, orig_ubs, mask, net.layers, pre_relu_indices, icp_score, gt=True)
        
        # select potential good spliting choices
        new_score = {}
        new_score_l2 = {}
        new_score_l1 = {}
        for i in range(len(score)):
            for j in range(len(score[i])):
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
        kw_choices = new_score[:20]+new_score_l1[:30]+new_score_l2[:15]
        #kw_choices = new_score[:5]+new_score_l1[:5]+new_score_l2[:5]
        kw_choices = [i[0] for i in kw_choices] + [f'relu_{kw_decision[0]}_{kw_decision[1]}']
        kw_choices = list(set(kw_choices))
        kw_indices = [[int(i.split('_')[1]), int(i.split('_')[2])] for i in kw_choices]
        
        
        import pdb; pdb.set_trace()
        # call gt_split to generate gt for selected relu_nodes
        records_relu = gt_split(net, candidate_domain, kw_indices, nb_visited_states, dump_trace)

        #decision = choose_dim(orig_lbs, orig_ubs, mask, net.layers)
        for choice in [0,1]:
            # Find the upper and lower bounds on the minimum in the domain
            # defined by n_mask_i
            nb_visited_states += 1
            if (nb_visited_states % 10) == 0:
                print(f"Running Nb states visited: {nb_visited_states}")
            
            
            dom_ub = records_relu["dom_ub"][choice] 
            dom_lb = records_relu["dom_lb"][choice]
            dom_ub_point = records_relu["dom_ub_point"][choice] 
            updated_mask = records_relu["mask"][choice] 
            dom_lb_all = records_relu["dom_lb_all"][choice] 
            dom_ub_all = records_relu["dom_ub_all"][choice]
            dom_dual_info = records_relu["dom_dual_info"][choice] 
            
            if dom_ub < global_ub:
                global_ub = dom_ub
                global_ub_point = dom_ub_point
            

            print('dom_lb: ', dom_lb)
            print('dom_ub: ', dom_ub)

            if dom_lb < global_ub:
                dom_to_add = ReLUDomain(updated_mask, lb=dom_lb, ub= dom_ub, lb_all= dom_lb_all, up_all = dom_ub_all, dual_info=dom_dual_info)
                add_domain(dom_to_add, domains)
                prune_counter += 1
            if nb_visited_states > 1000:
                return global_lb, global_ub, global_ub_point, nb_visited_states, True

        if prune_counter >= 100 and len(domains) >= 100:
            domains = prune_domains(domains, global_ub - eps)
            prune_counter = 0

        if len(domains) > 0:
            global_lb = domains[0].lower_bound
        else:
            # If there is no more domains, we have pruned them all
            global_lb = global_ub - eps

        print(f"Current: lb:{global_lb}\t ub: {global_ub}")

        # Stopping criterion
        if (decision_bound is not None) and (global_lb >= decision_bound):
            break
        elif global_ub < decision_bound:
            break

    return global_lb, global_ub, global_ub_point, nb_visited_states, False



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


def gt_split(net, candidate_domain, kw_indices, nb_visited_states, dump_trace):
    #compute ground truths for domain_splits
    mask = candidate_domain.mask
    orig_lbs = candidate_domain.lower_all
    orig_ubs = candidate_domain.upper_all
    orig_dual = candidate_domain.dual_info
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
        dom_ub,dom_lb, dom_ub_point, updated_mask, dom_lb_all, dom_ub_all, dom_dual_info= net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, decision, 0, orig_dual)

        mask_temp = [i.clone() for i in mask]
        dom_ub1,dom_lb1, dom_ub_point1, updated_mask1, dom_lb_all1, dom_ub_all1, dom_dual_info1= net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, decision, 1, orig_dual)

        lbs = torch.Tensor([dom_lb, dom_lb1])
        try:
            gt_lb_relu[decision[0]][decision[1]] = lbs
        except KeyError:
            gt_lb_relu[decision[0]]={}
            gt_lb_relu[decision[0]][decision[1]] = lbs

        print(f'idx: {decision}  solutions: {lbs}')

        lowest_lb_relu = lbs.min()
        if lowest_lb_relu > largest_lowest_lb_relu:
            largest_lowest_lb_relu = lowest_lb_relu
            largest_lowest_lb_index_relu = decision[1]
            largest_lowest_lb_layer_relu = decision[0]
            records_relu["dom_ub"] =[dom_ub,dom_ub1]
            records_relu["dom_lb"] = [dom_lb, dom_lb1]
            records_relu["dom_ub_point"] = [dom_ub_point, dom_ub_point1]
            records_relu["mask"] = [updated_mask, updated_mask1]
            records_relu["dom_lb_all"] = [dom_lb_all, dom_lb_all1]
            records_relu["dom_ub_all"] = [dom_ub_all, dom_ub_all1]
            records_relu["dom_dual_info"] = [dom_dual_info, dom_dual_info1]

    # dump traces
    if dump_trace is not None:

        trace_fname = dump_trace + '_branch_{}'.format(nb_visited_states)
        print("\n",trace_fname,"\n")

    dec = [largest_lowest_lb_layer_relu, largest_lowest_lb_index_relu]
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
