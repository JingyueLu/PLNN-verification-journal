import torch
import _pickle as pickle


def dump_problem(fname,
                    net,
                    domain,
                    ground_truth_lower_bounds,
                    global_lb,
                    global_ub,
                    domain_lb,
                    domain_width,
                    lp_time=-1):
    """
    Dump the problem specification. we will save ;
    - the network, as a list of pytorch layers
    - the current domain and their upper/lower bounds
    - the ground truth lower bounds
    """
    model_layers = net.layers

    lb_in, ub_in, dom = domain.lower_bound, domain.upper_bound, domain.domain # get these as tensors to save
    
    lb_all, ub_all = net.get_var_bounds(domain_lb + domain.domain * domain_width)

    dump = {}
    # the 'X' domains are (potential) inputs to the model
    dump['X'] = {}
    dump['Y'] = {}

    dump['X']['model'] = model_layers
    dump['X']['input_lower_bound'] = lb_in
    dump['X']['input_upper_bound'] = ub_in
    dump['X']['lower_bounds'] = lb_all
    dump['X']['upper_bounds'] = ub_all
    dump['X']['domain']      = dom
    dump['X']['global_lower_bound'] = global_lb

    dump['X']['domain_lb'] =     domain_lb,
    dump['X']['domain_width'] =  domain_width,

    dump['X']['global_upper_bound'] = global_ub
    # the 'Y' is stuff we want to predict
    dump['Y']['true_lp_bounds'] = ground_truth_lower_bounds
    
    dump['domainobj'] = domain

#    if decision is not None:
        # optionally, also record the decision taken by the smart branching method
#        lowest_bound, _ = torch.min(ground_truth_lower_bounds, 1)
#        _, optimal_greedy_decision = torch.max(lowest_bound,0 )
#        _, actual_decision = torch.max(decision, 0)

#    dump['final_valid_decision'] = final_kw_decision
#    dump['kw_valid?'] = valid
#    dump['correct?'] = correct
#    dump['approx_bounds'] = approx_bounds
    if lp_time > 0:
        dump['lp_time'] = lp_time

    with open(fname, 'wb') as f:
        pickle.dump(dump, f)
    
    return



def dump_relu_problem(fname,
                    mask,
                    global_lb,
                    global_ub,
                    lower_bounds,
                    upper_bounds,
                    gt_lb_relu,
                    decision,
                    ):

    dump = {}
    # the 'X' domains are (potential) inputs to the model
    dump['X'] = {}
    dump['relu'] = {}
    dump['Y'] = {}

    dump['X']['mask']  = mask
    dump['X']['global_lower_bound'] = global_lb
    dump['X']['global_upper_bound'] = global_ub
    dump['X']['lower_bounds'] = lower_bounds
    dump['X']['upper_bounds'] = upper_bounds

    #  split results
    dump['relu']['true_lp_bounds'] = gt_lb_relu

    # results after split
    dump['Y']['decision'] = decision

    with open(fname, 'wb') as f:
        pickle.dump(dump, f)

