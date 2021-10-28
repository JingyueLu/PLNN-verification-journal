import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from plnn.modules import Flatten

'''
the original version with more details and comments are in 
/convex_adversarial/examples/gt_stats/kw_analysis.py

08/01/19
switching to domain split after consequtive relu_0 splits has very little effects
difficult case: digit_2_idx_172_property_0 
potential solution ?: 
keep a running record of important split decisions (especially
the decisions on the second or first relu layer). Then when we encounter the sitution
where no signal from relu can be detected, we pick a viable decision from recorded 
decisions??

'''

def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp/(upper_temp-lower_temp)
    intercept = -1*lower_temp*slope_ratio

    #if (lower >= 0):
    #    decision = [1,0]
    #elif (upper <=0):
    #    decision = [0, 0]
    #else: 
    #    temp = upper/(upper - lower)
    #    decision = [temp, -temp*lower]
    #return decision
    return slope_ratio, intercept



def choose_node_conv(lower_bounds,upper_bounds, orig_mask, layers, pre_relu_indices, icp_score_counter, gt=False):
    '''
    choose the dimension to split on
    based on each node's contribution to the cost function
    in the KW formulation.

    '''
         
    mask = [(i==-1).float().view(-1) for i in orig_mask]
    score=[]
    intercept_tb = []

    ratio  = torch.ones(1)
    # starting from 1, back-propogating: if the weight is negative
    # introduce bias; otherwise, intercept is 0
    # we are only interested in two terms for now: the slope x bias of the node
    # and bias x the amount of argumentation introduced by later layers.
    # From the last relu-containing layer to the first relu-containing layer

    # Record score in a dic
    #new_score = {}
    #new_intercept = {}
    relu_idx = -1

    for layer_idx, layer in reversed(list(enumerate(layers))):
        if type(layer) is nn.Linear:
            ratio = ratio.unsqueeze(-1)
            w_temp = layer.weight.detach()
            ratio = torch.t(w_temp) @ ratio
            ratio = ratio.view(-1)
            #import pdb; pdb.set_trace()

        elif type(layer) is nn.ReLU:
            #compute KW ratio
            ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]], upper_bounds[pre_relu_indices[relu_idx]])
            # Intercept
            intercept_temp = torch.clamp(ratio, max = 0)
            intercept_candidate = intercept_temp*ratio_temp_1
            intercept_tb.insert(0, intercept_candidate.view(-1)*mask[relu_idx])
            
            #Bias
            b_temp = layers[layer_idx-1].bias.detach()
            if type(layers[layer_idx-1]) is nn.Conv2d:
                b_temp = b_temp.unsqueeze(-1).unsqueeze(-1)
            ratio_1 = ratio*(ratio_temp_0-1)
            bias_candidate_1 = b_temp*ratio_1
            ratio = ratio*ratio_temp_0
            bias_candidate_2 = b_temp*ratio
            bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)
            #test = (intercept_candidate!=0).float()
            # ???if the intercept_candiate at a node is 0, we should skip this node 
            #    (intuitively no relaxation triangle is introduced at this node)
            #    score_candidate = test*bias_candidate + intercept_candidate
            score_candidate = bias_candidate + intercept_candidate
            score.insert(0, abs(score_candidate).view(-1)*mask[relu_idx])
            

            relu_idx -= 1
        
        elif type(layer) is nn.Conv2d:
            #import pdb; pdb.set_trace()
            ratio = ratio.unsqueeze(0)
            ratio = F.conv_transpose2d(ratio, layer.weight, stride = layer.stride, padding = layer.padding)
            ratio = ratio.squeeze(0)

        elif type(layer) is Flatten:
            #import pdb; pdb.set_trace()
            ratio = ratio.reshape(lower_bounds[layer_idx].size())
        else:
            raise NotImplementedError
    
    max_info = [torch.max(i,0) for i in score]
    decision_layer = max_info.index(max(max_info))
    decision_index = max_info[decision_layer][1].item()
    if decision_layer!=0 and max_info[decision_layer][0].item()>0.001:
        #temp = torch.zeros(score[decision_layer].size())
        #temp[decision_index]=1
        #decision_index = torch.nonzero(temp.reshape(mask[decision_layer].shape))[0].tolist()
        decision = [decision_layer, decision_index]

    else:
        min_info = [[i,torch.min(intercept_tb[i],0)] for i in range(len(intercept_tb)) if torch.min(intercept_tb[i])<-1e-4]
        intercept_layer = min_info[-1][0]
        intercept_index = min_info[-1][1][1].item()
        #import pdb; pdb.set_trace()
        if icp_score_counter<2:
            icp_score_counter +=1 
            #inter_temp = torch.zeros(intercept_tb[intercept_layer].size())
            #inter_temp[intercept_index]=1
            #intercept_index = torch.nonzero(inter_temp.reshape(mask[intercept_layer].shape))[0].tolist()
            decision = [intercept_layer, intercept_index]
            if intercept_layer!=0:
                icp_score_counter = 0
            print('\tusing intercept score')
        else:
            print('\t using a random choice')
            if len(mask[2].nonzero())!= 0:
                decision = [2, mask[2].nonzero()[0].item()]
            elif len(mask[1].nonzero())!=0:
                decision = [1, mask[1].nonzero()[0].item()]
            else:
                decision = [0, mask[0].nonzero()[0].item()]
            icp_score_counter = 0
    if gt is False:
        return decision, icp_score_counter
    else:
        return decision, icp_score_counter, score



