import torch
from torch import nn
from torch.nn import functional as F
from convex_adversarial import DualNetwork
from convex_adversarial.dual_layers import DualLinear, DualReLU, DualConv2d, DualReshape
from convex_adversarial.dual_inputs import InfBallBounded

from plnn.network_linear_approximation import LinearizedNetwork
from plnn.model import reluify_maxpool, simplify_network
from plnn.modules import Flatten


class LooseDualNetworkApproximation(LinearizedNetwork):
    def __init__(self, layers, x = None, ball_eps = None):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.x = x
        self.ball_eps = ball_eps

    def remove_maxpools(self, domain, no_opt=False):
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain, no_opt))
            self.layers = new_layers

    def get_lower_bounds(self, domains):
        '''
        Create the linear approximation for `domains` of the network and use it
        to compute a lower bound on the minimum of the output.

        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
                Batch_idx x dimension x bound_type (0 -> lb, 1 -> ub)
        '''
        batch_size = domains.size(0)
        dual = self.build_approximation(domains)

        # since we have already encoded the property as a network layer,
        # our c vector in the kw sense is just a constant.
        bound_val = dual(torch.ones(batch_size, 1, 1))

        bound_val = bound_val.squeeze_(1)
        del dual

        return bound_val
    
    def init_kw_bounds_full(self, domain):
        lower_bounds = [domain[:,0]]
        upper_bounds = [domain[:,1]]
        
        dual = DualNetwork(self.net, self.x, self.ball_eps, bounded_input=True)
        for layer in dual.dual_net[1:]:
            if type(layer) is DualReLU:
                # K&W has this as input bounds to the ReLU, but our
                # codes reasons in terms of output bounds of a layer
                # We get this bounds and enqueue them, they correspond
                # to the output of the ReLU from before.
                
                lower_bounds[-1] = layer.zl.squeeze()
                upper_bounds[-1]= layer.zu.squeeze()
                lower_bounds.append(F.relu(lower_bounds[-1]))
                upper_bounds.append(F.relu(upper_bounds[-1]))
                
            elif Flatten:
                lower_bounds.append(lower_bounds[-1].view(-1))
                upper_bounds.append(upper_bounds[-1].view(-1))
            else:
                lower_bounds.append([])
                upper_bounds.append([])


        # Also add the bounds on the final thing
        lower_bounds.append(dual(torch.ones(1,1,1)).view(-1))
        upper_bounds.append(-dual(-torch.ones(1,1,1)).view(-1))
        #self.nf = dual.nf 
        return lower_bounds, upper_bounds


    def init_kw_bounds(self, pre_lb_all=None, pre_ub_all=None, decision=None, choice=None):
        '''
        ALWAYS create a new DualNetwork instance

        Returing intermediate kw bounds

        Input: mask and pre-computed intermedidate bounds consisted only of linear layers
        right before relu layers. 
        Since changing the relu mask is only going to affect bounds for layers after 
        the changed relu layer, we directly pass these values and only compute bounds 
        for later layers. However, we do create a new dual instance every time when we 
        compute bounds.

        '''
        if pre_lb_all is None and pre_ub_all is None:        
            lower_bounds = []
            upper_bounds = []
            
            dual = DualNetwork(self.net, self.x, self.ball_eps, bounded_input=True)
            self.pre_relu_indices = []
            idx = -1
            for layer in dual.dual_net[0:]:
                if type(layer) is DualReLU:
                    # K&W has this as input bounds to the ReLU, but our
                    # codes reasons in terms of output bounds of a layer
                    # We get this bounds and enqueue them, they correspond
                    # to the output of the ReLU from before.
                    lower_bounds[-1] = layer.zl.squeeze()
                    upper_bounds[-1]= layer.zu.squeeze()
                    lower_bounds.append([])
                    upper_bounds.append([])
                    self.pre_relu_indices.append(idx)
                    

                else:
                    lower_bounds.append([])
                    upper_bounds.append([])
                idx += 1

            # Also add the bounds on the final thing
            lower_bounds.append(dual(torch.ones(1,1,1)).view(-1))
            upper_bounds.append(-dual(-torch.ones(1,1,1)).view(-1))
            #self.nf = dual.nf 
            dual_info = [dual.dual_net, dual.last_layer]
            self.dual_info = dual_info
            return lower_bounds, upper_bounds, self.pre_relu_indices, dual_info

        else:
            pre_relu_lb = [pre_lb_all[i].clone() for i in self.pre_relu_indices]
            pre_relu_ub = [pre_ub_all[i].clone() for i in self.pre_relu_indices]
            if decision is not None:
                # upper_bound for the corresponding relu is forced to be 0
                if choice ==0:
                    pre_relu_ub[decision[0]].view(-1)[decision[1]] = 0
                else:
                    pre_relu_lb[decision[0]].view(-1)[decision[1]] = 0
            dual = DualNetwork(self.net, self.x, self.ball_eps, bounded_input=True, provided_zl=pre_relu_lb, provided_zu=pre_relu_ub)
        #    #dual = DualNetwork(self.net, self.x, self.ball_eps, bounded_input=True, mask =mask)
            lower_bounds = [pre_lb_all[0]]
            upper_bounds = [pre_ub_all[0]]
            #bounds_index = 0
            #changed_bounds_index = self.pre_relu_indices[-1]
            #bounds_unchanged = True
            for layer in dual.dual_net[1:]:
                if type(layer) is DualReLU:
                    # K&W has this as input bounds to the ReLU, but our
                    # codes reasons in terms of output bounds of a layer
                    # We get this bounds and enqueue them, they correspond
                    # to the output of the ReLU from before.
                    lower_bounds[-1] = layer.zl.squeeze()
                    upper_bounds[-1]= layer.zu.squeeze()
                    lower_bounds.append(F.relu(lower_bounds[-1]))
                    upper_bounds.append(F.relu(upper_bounds[-1]))
                    
                elif Flatten:
                    lower_bounds.append(lower_bounds[-1].view(-1))
                    upper_bounds.append(upper_bounds[-1].view(-1))
                else:
                    lower_bounds.append([])
                    upper_bounds.append([])
                #bounds_index += 1

            # Also add the bounds on the final thing
            lower_bounds.append(max(dual(torch.ones(1,1,1)).view(-1), pre_lb_all[-1]))
            upper_bounds.append(min(-dual(-torch.ones(1,1,1)).view(-1), pre_ub_all[-1]))
            dual_info = [dual.dual_net, dual.last_layer]
            return lower_bounds, upper_bounds, dual_info
    
    def update_kw_bounds(self, dual_info, change_idx, pre_lb_all = None, pre_ub_all =None, decision =None, choice=None, de_dual = None):
        #first update pre_lb_all and pre_ub_all if decision is not None
        upper_bounds = [i.clone() for i in pre_ub_all[:change_idx+1]]
        lower_bounds = [i.clone() for i in pre_lb_all[:change_idx+1]]
        if decision is not None:
            # upper_bound for the corresponding relu is forced to be 0
            if choice ==0:
                upper_bounds[change_idx].view(-1)[decision[1]]=0
                
            else:
                lower_bounds[change_idx].view(-1)[decision[1]] = 0

        # Recreate previous unchanged dual-layers
        new_dual_net = []
        for idx, dual_layer in enumerate(dual_info[0][:change_idx+1]):
            if type(dual_layer) is InfBallBounded:
                new_dual_layer = InfBallBounded(X = self.x, epsilon=self.ball_eps,l=dual_layer.l, u=dual_layer.u,
                                                nu_1=dual_layer.nu_1[:change_idx+1-idx], 
                                                nu_x = dual_layer.nu_x[:change_idx+1-idx])

            elif type(dual_layer) is DualConv2d:
                new_dual_layer = DualConv2d(layer=dual_layer.layer, 
                                            out_features=dual_layer.out_features, 
                                            bias=dual_layer.bias[:change_idx+1-idx])

            elif type(dual_layer) is DualReLU:
                if dual_layer.I_empty is True:
                    new_dual_layer = DualReLU(zl=dual_layer.zl, zu=dual_layer.zu, 
                                              I=dual_layer.I, I_ind=None, 
                                              I_empty=dual_layer.I_empty, 
                                              I_collapse = None, 
                                              d=dual_layer.d, nus=None)

                else:
                    new_dual_layer = DualReLU(zl=dual_layer.zl, zu=dual_layer.zu, 
                                              I=dual_layer.I, I_ind=dual_layer.I_ind, 
                                              I_empty=dual_layer.I_empty, 
                                              I_collapse = dual_layer.I_collapse, 
                                              d=dual_layer.d, nus=dual_layer.nus[:change_idx+1-idx])

            elif type(dual_layer) is DualLinear:
                new_dual_layer = DualLinear(layer=dual_layer.layer, 
                                            out_features=dual_layer.out_features, 
                                            bias = dual_layer.bias[:change_idx+1-idx])

            elif type(dual_layer) is DualReshape:
                new_dual_layer = DualReshape(in_f=dual_layer.in_f, 
                                             out_f=dual_layer.out_f, copy=True)

            new_dual_net.append(new_dual_layer)
        # Update dual_info for layers after change_idx
        #import pdb; pdb.set_trace()
        for i, dual_layer in enumerate(dual_info[0][change_idx+1:]):
            #print(i, dual_layer)
            if i==0:
                assert type(dual_layer) is DualReLU, "change idx is wrong"
                new_dual_layer = DualReLU(zl=lower_bounds[change_idx].unsqueeze(0),
                                          zu= upper_bounds[change_idx].unsqueeze(0))
            else:
                if type(dual_layer) is DualLinear:
                    new_dual_layer = DualLinear(layer=dual_layer.layer, 
                                                out_features=dual_layer.out_features)

                elif type(dual_layer) is DualConv2d:
                    new_dual_layer = DualConv2d(layer=dual_layer.layer, 
                                                out_features=dual_layer.out_features)

                elif type(dual_layer) is DualReLU:
                    zl,zu = zip(*[l.bounds() for l in new_dual_net])
                    zl,zu = sum(zl), sum(zu)
                    zu = torch.min(zu, pre_ub_all[i+change_idx]) 
                    zl = torch.max(zl, pre_lb_all[i+change_idx])
                    new_dual_layer = DualReLU(zl=zl,zu=zu)

                elif type(dual_layer) is DualReshape:
                    new_dual_layer= DualReshape(in_f=dual_layer.in_f,
                                                out_f=dual_layer.out_f, copy=True)

                else:
                    raise ValueError("No module for layer {}".format(str(dual_layer.__class__.__name__)))
            for l in new_dual_net:
                l.apply(new_dual_layer)
            new_dual_net.append(new_dual_layer)
        
        #import pdb; pdb.set_trace()
        lower_bounds.append(F.relu(lower_bounds[-1]))
        upper_bounds.append(F.relu(upper_bounds[-1]))
        #extract all new_updated_bounds
        for layer in new_dual_net[change_idx+2:]:
            if type(layer) is DualReLU:
                # K&W has this as input bounds to the ReLU, but our
                # codes reasons in terms of output bounds of a layer
                # We get this bounds and enqueue them, they correspond
                # to the output of the ReLU from before.
                lower_bounds[-1] = layer.zl.squeeze()
                upper_bounds[-1]= layer.zu.squeeze()
                lower_bounds.append(F.relu(lower_bounds[-1]))
                upper_bounds.append(F.relu(upper_bounds[-1]))
                
            elif type(layer) is DualReshape:
                lower_bounds.append(lower_bounds[-1].view(-1))
                upper_bounds.append(upper_bounds[-1].view(-1))
            else:
                lower_bounds.append([])
                upper_bounds.append([])
        # deal with the final one
        lower_bounds.append(max(self.last_layer_objective(torch.ones(1,1,1), new_dual_net,dual_info[1]).view(-1),pre_lb_all[-1]))
        upper_bounds.append(min(-self.last_layer_objective(-torch.ones(1,1,1),new_dual_net, dual_info[1]).view(-1), pre_ub_all[-1]))

        new_dual_info = [new_dual_net, dual_info[1]]
        return lower_bounds, upper_bounds, new_dual_info




    def last_layer_objective(self, c, dual_net, last_layer):
        """ For the constructed given dual network, compute the objective for
        some given vector c """
        nu = [-c]
        nu.append(last_layer.T(*nu))
        for l in reversed(dual_net[1:]): 
            nu.append(l.T(*nu))
        dual_net = dual_net + [last_layer]
                
        return sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) for i,l in enumerate(dual_net))


        
            

    def Test_linear_bounds(self, net, domain, mask=None, pre_lb_all=None, pre_ub_all=None):
        '''
        random test: attempt to improve the bounds of all other 
                     ambiguous relu nodes on the relu layer we decide to
                     split on
        '''
        pre_relu_indices = [1,3,6]

        lower_bounds = []
        upper_bounds = []
        
        # Add the input bounds
        lower_bounds.append(domain[:, 0])
        upper_bounds.append(domain[:, 1])

        if mask is None:
            dual = DualNetwork(net, self.x, self.ball_eps, bounded_input=True)
        else:
            pre_relu_lb = [torch.tensor(pre_lb_all[i]).clone() for i in pre_relu_indices]
            pre_relu_ub = [torch.tensor(pre_ub_all[i]).clone() for i in pre_relu_indices]
            mask = [torch.tensor(i) for i in mask]
            dual = DualNetwork(net, self.x, self.ball_eps, bounded_input=True, mask =mask, provided_zl=pre_relu_lb, provided_zu=pre_relu_ub)

        for layer in dual.dual_net[1:]:
            if type(layer) is DualLinear:
                # Skip this one, we're going to get his bound
                # with the next DualReLU
                pass
            elif type(layer) is DualReLU:
                # K&W has this as input bounds to the ReLU, but our
                # codes reasons in terms of output bounds of a layer
                # We get this bounds and enqueue them, they correspond
                # to the output of the ReLU from before.
                lower_bounds.append(layer.zl.squeeze())
                upper_bounds.append(layer.zu.squeeze())

                # Let's also trivially determine what are the bounds on the
                # outputs of the ReLU
                lower_bounds.append(torch.clamp(layer.zl, 0).squeeze())
                upper_bounds.append(torch.clamp(layer.zu, 0).squeeze())
            else:
                raise NotImplementedError("Unplanned layer type.")

        # Also add the bounds on the final thing
        lower_bounds.append(dual(torch.ones(1,1,1)).squeeze())
        upper_bounds.append(-dual(-torch.ones(1,1,1)).squeeze())
        return lower_bounds, upper_bounds, dual



    def get_intermediate_bounds(self, domain, mask =None, dual=None):
        '''
        Create the linear approximation and return all the intermediate bounds.
        '''
        # Transferred to UNBOUNDED BALL
        if dual == None:
            batch_domain = domain.unsqueeze(0)
            dual = self.build_approximation(batch_domain, mask)

        # Let's get the intermediate bounds, in the same way that they are
        # obtained by the other methods.
        lower_bounds = []
        upper_bounds = []
        # Add the input bounds
        lower_bounds.append(domain[:, 0])
        upper_bounds.append(domain[:, 1])

        for layer in dual.dual_net[1:]:
            if isinstance(layer, DualLinear):
                # Skip this one, we're going to get his bound
                # with the next DualReLU
                pass
            elif isinstance(layer, DualReLU):
                # K&W has this as input bounds to the ReLU, but our
                # codes reasons in terms of output bounds of a layer
                # We get this bounds and enqueue them, they correspond
                # to the output of the ReLU from before.
                lower_bounds.append(layer.zl.squeeze())
                upper_bounds.append(layer.zu.squeeze())

                # Let's also trivially determine what are the bounds on the
                # outputs of the ReLU
                lower_bounds.append(torch.clamp(layer.zl, 0).squeeze())
                upper_bounds.append(torch.clamp(layer.zu, 0).squeeze())
            else:
                raise NotImplementedError("Unplanned layer type.")

        # Also add the bounds on the final thing
        lower_bounds.append(dual(torch.ones(1,1,1)).squeeze())
        upper_bounds.append(-dual(-torch.ones(1,1,1)).squeeze())
        return lower_bounds, upper_bounds

    def build_approximation(self, domains, mask =None):
        # Okay, this is a disgusting, disgusting hack. This DEFINITELY should
        # be replaced by something more proper in practice but I'm doing this
        # for a quick experiment.

        # The code from https://github.com/locuslab/convex_adversarial only
        # works in the case of adversarial examples, that is, it assumes the
        # domain is centered around a point, and is limited by an infinity norm
        # constraint. Rather than properly implementing the general
        # optimization, I'm just going to convert the problem into this form,
        # by adding a fake linear at the beginning. This is definitely not
        # clean :)
        batched = domains.shape[0] > 1

        domain_lb = domains.select(2, 0)
        domain_ub = domains.select(2, 1)

        with torch.no_grad():
            x = (domain_ub + domain_lb) / 2
            domain_radius = (domain_ub - domain_lb)/2

            #if batched:
            #    # Verify that we can use the same epsilon for both parts
            #    assert (domain_radius[0] - domain_radius[1]).abs().sum() < 1e-6
            #    # We have written the code assuming that the batch size would
            #    # be limited to 2, check that it is the case.
            #    assert domains.shape[0] <= 2

            domain_radius = domain_radius[0]

            # Disgusting hack number 2:
            # In certain case we don't want to allow a variable to move.
            # Let's just allow it to move a tiny tiny bit
            domain_radius[domain_radius == 0] = 1e-6

            bias = x[0].clone()
            x[0].fill_(0)
            if batched:
                x[1] = (x[1] - bias) / domain_radius

            inp_layer = nn.Linear(domains.size(1), domains.size(1), bias=True)
            inp_layer.weight.copy_(torch.diag(domain_radius))
            inp_layer.bias.copy_(bias)
            fake_net = nn.Sequential(*simplify_network([inp_layer]
                                                       + self.layers))

            dual = DualNetwork(fake_net, x, 1, mask = mask)

        return dual

