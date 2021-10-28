import torch
import logger

from torch import nn
from torch.nn import functional as F
from plnn.network_linear_approximation import LinearizedNetwork
from proxlp.base_problem import Problem
from proxlp.variable import Var
from proxlp.timer import timer

import matplotlib.pyplot as plt
import seaborn as sns

import time

DO_PLOTS = False
USE_CUDA = True

SOLVER_OPT = dict(
    debug_mode=DO_PLOTS, verbose_mode=True,
    max_outer_iter=int(100),
    max_inner_iter=int(5),

    int_primal_eval=False,
    lp_primal_eval=False,

    fench_dual_check=True,
    fench_dual_threshold=1,

    dual_change_check=False,
    dual_change_threshold=1e-3,

    # Inner iteration early stop based on optimal step size
    # value
    gamma_threshold=1e-3,

    # Outer iteration early bail
    primal_dual_check=False,
    primal_dual_threshold=1e-2
)

class Reporter(object):

    def __init__(self, env_name):
        visdom_opts = dict(server="http://localhost", port=8097,
                           unsafe_send=True)
        self.xp = logger.Experiment(env_name, log_git_hash=True,
                                    use_visdom=True, visdom_opts=visdom_opts,
                                    time_indexing=False)
        self.dual_metric = self.xp.ParentWrapper(tag="dual", name='parent',
                                                 children=(self.xp.SimpleMetric(name='primal_obj'),))
        self.prox_dual_metric = self.xp.ParentWrapper(tag="prox_dual", name='parent',
                                                      children=(self.xp.SimpleMetric(name='prox_obj'),))
        self.dual_prox_dual_metric = self.xp.ParentWrapper(tag="dual_prox_dual", name='parent',
                                                           children=(self.xp.SimpleMetric(name='prox_obj'),))
        self.gamma_metric = self.xp.ParentWrapper(tag="gamma", name='parent',
                                                  children=(self.xp.SimpleMetric(name='gamma'),))
        self.dual_change_metric = self.xp.ParentWrapper(tag="dual_change", name='parent',
                                                        children=(self.xp.SimpleMetric(name='dual_change'),))

    def report_metrics(self, debug_state):
        nb_it = len(debug_state["dual"])
        for step in range(nb_it):
            def log_primal(metric, val):
                if val is None:
                    return
                metric.update(primal_obj=val[-1])
                self.xp.log_metric(metric, idx=step)
                metric.reset()
            log_primal(self.dual_metric, debug_state["dual"][step])
            def log_prox(metric, val):
                if val is None:
                    return
                metric.update(prox_obj=val[-1])
                self.xp.log_metric(metric, idx=step)
                metric.reset()
            log_prox(self.prox_dual_metric, debug_state["prox_dual"][step])
            log_prox(self.dual_prox_dual_metric, debug_state["dual_of_prox_dual"][step])

            if debug_state["gamma"][step] is not None:
                self.gamma_metric.update(gamma=debug_state["gamma"][step].mean())
                self.xp.log_metric(self.gamma_metric, idx=step)
                self.gamma_metric.reset()

            if debug_state["dual_change"][step] is not None:
                self.dual_change_metric.update(dual_change=debug_state["dual_change"][step].mean())
                self.xp.log_metric(self.dual_change_metric, idx=step)
                self.dual_change_metric.reset()

        self.xp.plotter.wait_sending()

class Benchmarker(object):

    def __init__(self, exp_name):
        visdom_opts = dict(server="http://localhost", port=8097,
                           unsafe_send=True)
        self.xp = logger.Experiment(exp_name, log_git_hash=False,
                                    use_visdom=True, visdom_opts=visdom_opts,
                                    time_indexing=False)
        self.bound_metric = self.xp.ParentWrapper(tag="Bound", name='parent',
                                                  children=(self.xp.SimpleMetric(name='logbound'),
                                                            self.xp.SimpleMetric(name='runtime')))

    def do_benchmark(self, weights, obj_layer,
                     lower_bounds, upper_bounds,
                     solver_opt, target_lbs, target_ubs):
        nb_etas = 2 * len(weights)
        ranges = [ub - lb for ub, lb in zip(upper_bounds, lower_bounds)]

        nb_steps = 50

        power_coeff = torch.linspace(-4, 4, steps=nb_steps).tolist()
        scaling_factor = torch.logspace(-4, 4, steps=nb_steps).tolist()

        results = torch.zeros((nb_steps, nb_steps))
        runtimes = torch.zeros((nb_steps, nb_steps))
        for power_coeff_idx in range(len(power_coeff)):
            power = power_coeff[power_coeff_idx]
            for scaling_factor_idx in range(len(scaling_factor)):
                scale = scaling_factor[scaling_factor_idx]

                etas = []
                for rng in ranges:
                    eta_val = scale * torch.pow(rng, power).mean()
                    etas.extend([eta_val]*4)


                timer.start("local")
                lbs, ubs = solve_problem(weights, obj_layer,
                                         lower_bounds, upper_bounds,
                                         etas,
                                         solver_opt, False)
                timer.end("local")

                total_gap = ubs - target_ubs + (target_lbs - lbs)
                log_total_gap = total_gap.log()
                runtime = timer.get_val("local")
                print(f"\n\n\nPower: {power}")
                print(f"Etas: {etas}")
                print(f"Total gap: {total_gap}")
                print(f"Runtime: {runtime}")
                results[power_coeff_idx][scaling_factor_idx] = log_total_gap
                runtimes[power_coeff_idx][scaling_factor_idx] = runtime
                # self.bound_metric.update(runtime=runtime,
                #                          logbound=log_total_gap)
                # self.xp.log_metric(self.bound_metric, idx=coeff)
                # self.bound_metric.reset()
                timer.reset()

        # self.xp.plotter.wait_sending()
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        ax = sns.heatmap(results, yticklabels=power_coeff,
                         xticklabels=scaling_factor)
        ax.set_ylabel("Power coeff")
        ax.set_xlabel("Scaling factor")
        plt.subplot(2, 1, 2)
        ax = sns.heatmap(runtimes, yticklabels=power_coeff,
                         xticklabels=scaling_factor)
        ax.set_ylabel("Power coeff")
        ax.set_xlabel("Scaling factor")
        plt.show()


class LinearOp:
    LINEAR = 0
    CONV2D = 1

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.out_features = weights.shape[0]
        self.in_features = weights.shape[1]

        if weights.dim() == 2:
            self.op_type = LinearOp.LINEAR
        elif weights.dim() == 4:
            self.op_type = LinearOp.CONV2D
        else:
            raise NotImplementedError

    def normalize_outrange(self, lbs, ubs):
        if self.op_type is LinearOp.LINEAR:
            inv_range = 1.0 / (ubs - lbs)
            self.bias = inv_range * (2 * self.bias - ubs - lbs)
            self.weights = 2 * inv_range.unsqueeze(1) * self.weights
        elif self.op_type is LinearOp.CONV2D:
            # TODO: store the data so that we can do the operation this
            # correspond to.
            pass

    def add_prerescaling(self, pre_scales):
        if self.op_type is LinearOp.LINEAR:
            self.weights = self.weights * pre_scales.unsqueeze(0)
        elif self.op_type is LinearOp.CONV2D:
            # TODO: store the data so that we can do the operation this
            # correspond to.
            pass


    def forward(self, inp):
        if self.op_type is LinearOp.LINEAR:
            return inp @ self.weights.t() + self.bias
        return weights

    def backward(self, out):
        if self.op_type is LinearOp.LINEAR:
            return out @ self.weights

    def __repr__(self):
        if self.op_type is LinearOp.LINEAR:
            return f'<Linear: {self.in_features} -> {self.out_features}>'


def solve_problem(weights, obj_layer,
                  lower_bounds, upper_bounds,
                  etas,
                  solver_opt, do_plots):
    problem = PlanetProblem(**solver_opt)
    problem.init_parameters(weights, obj_layer,
                            lower_bounds, upper_bounds,
                            etas)
    problem.solve()

    if DO_PLOTS:
        # Setup the reporter for the plots
        reporter = Reporter(f"layer_{lay_idx}_optim")
        # Get the stored data
        debug_state = problem.solver.debug_state
        # Send it to Vizdom
        reporter.report_metrics(debug_state)

    # We can use the values that are in the dual variables as they
    # will provide proper bounds
    unbiased_bounds = problem.dval
    nb_units = unbiased_bounds.shape[0] // 2
    u_kp1 = unbiased_bounds[:nb_units] + obj_layer.bias
    l_kp1 = -unbiased_bounds[nb_units:] + obj_layer.bias
    assert (u_kp1 - l_kp1).min() >= 0

    return l_kp1, u_kp1

def generate_etas(weights,
                  lower_bounds, upper_bounds):
    '''
    Generate the eta per subproblem vector.

    We actually don't strictly enforce that there is a eta per subproblem
    but that's what I have written down.
    What is expected is simply one eta per tensors of the dual variable.
    '''
    nb_etas = 2 * len(weights)
    ranges = [ub - lb for ub, lb in zip(upper_bounds, lower_bounds)]

    # Do the pick of the eta
    ## OPTION 1: Constant eta
    # self.etas = 1e-2
    ## OPTION 2: Indexed on depth
    # etas = []
    # for i in range(nb_etas):
    #     val = 2.44 * (0.4 ** i)
    #     etas.extend([val, val])
    ## OPTION 3: As a function of average range
    etas = []
    for rng in ranges:
        val = (5/rng).mean()
        etas.extend([val]*4)

    return etas


class NetworkLP(LinearizedNetwork):

    def __init__(self, layers):
        '''
        layers: list of Pytorch layers
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

        for param in self.net.parameters():
            param.requires_grad = False


    def define_linear_approximation(self, input_domain):
        grb_start = time.time()
        LinearizedNetwork.define_linear_approximation(self, input_domain)
        grb_end = time.time()
        print(f"Gurobi timing: {grb_end - grb_start}")

        if USE_CUDA:
            input_domain = input_domain.cuda()
            self.layers = [layer.cuda() for layer in self.layers]

        prox_start = time.time()

        # Setup the bounds on the inputs
        l_0 = input_domain.select(-1, 0)
        u_0 = input_domain.select(-1, 1)

        # Setup the bounds after the first linear layer, because we have closed
        # form
        first_linear = self.layers[0]
        w_1 = first_linear.weight
        b_1 = first_linear.bias

        pos_w1 = torch.clamp(w_1, 0, None)
        neg_w1 = torch.clamp(w_1, None, 0)
        if isinstance(first_linear, nn.Linear):
            l_1 = pos_w1 @ l_0 + neg_w1 @ u_0 + b_1
            u_1 = pos_w1 @ u_0 + neg_w1 @ l_0 + b_1

            # Build the "conditioned" first layer
            range_0 = (u_0 - l_0)
            range_1 = (u_1 - l_1)
            cond_w_1 = (1/range_1).unsqueeze(1) * w_1 * range_0
            cond_b_1 = (1/range_1) * (2 * b_1 - (u_1 + l_1) + w_1 @ (u_0 + l_0))

            cond_first_linear = LinearOp(cond_w_1, cond_b_1)
        elif isinstance(first_linear, nn.Conv2d):
            l_1 = (F.conv2d(l_0, pos_w1, b_1,
                            first_linear.stride, first_linear.padding,
                            first_linear.dilation, first_linear.groups)
                   + F.conv2d(u_0, neg_w1, None,
                              first_linear.stride, first_linear.padding,
                              first_linear.dilation, first_linear.groups))
            u_1 = (F.conv2d(u_0, pos_w1, b_1,
                            first_linear.stride, first_linear.padding,
                            first_linear.dilation, first_linear.groups)
                   + F.conv2d(l_0, neg_w1, None,
                              first_linear.stride, first_linear.padding,
                              first_linear.dilation, first_linear.groups))

            import IPython; IPython.embed();
            import sys; sys.exit()

        lower_bounds = [l_1]
        upper_bounds = [u_1]

        weights = [cond_first_linear]

        # Now, iterate over the subsequent layers to get all the intermediate
        # bounds we want.
        for lay_idx, layer in enumerate(self.layers[1:]): # We have already dealt with the first
            if isinstance(layer, nn.Linear):
                # Create the objective function
                w_kp1 = layer.weight
                b_kp1 = layer.bias
                nb_units = b_kp1.shape[0]

                obj_layer = LinearOp(w_kp1, b_kp1)
                obj_layer.add_prerescaling(torch.clamp(upper_bounds[-1], 0, None))

                grb_ukp1 = obj_layer.bias.new_tensor(self.upper_bounds[lay_idx + 2])
                grb_lkp1 = obj_layer.bias.new_tensor(self.lower_bounds[lay_idx + 2])
                if lay_idx == 11:
                    # bm = Benchmarker("geom_parameter")
                    # bm.do_benchmark(weights, obj_layer,
                    #                 lower_bounds, upper_bounds,
                    #                 SOLVER_OPT, grb_lkp1, grb_ukp1)

                    etas = generate_etas(weights, lower_bounds, upper_bounds)
                    l_kp1, u_kp1 = solve_problem(weights, obj_layer,
                                                 lower_bounds, upper_bounds,
                                                 etas,
                                                 SOLVER_OPT, DO_PLOTS)
                    # Gurobi bounds: +1 for the fact that enumerate from 1, +1
                    # because the interface for gurobi stuff also counts initial

                    best_ub_gap = (u_kp1 - grb_ukp1).min()
                    worst_ub_gap = (u_kp1 - grb_ukp1).max()
                    best_lb_gap = (grb_lkp1 - l_kp1).min()
                    worst_lb_gap = (grb_lkp1 - l_kp1).max()
                    # GRB is the best (largest) lower bound.
                    # So the gap should always be positive
                    print(f"Layer {lay_idx} - best UB gap (should always be >0): {best_ub_gap}")
                    print(f"Layer {lay_idx} - best LB gap (should always be >0): {best_lb_gap}")
                    print(f"Layer {lay_idx} - worst UB gap (need as small as possible): {worst_ub_gap}")
                    print(f"Layer {lay_idx} - worst LB gap (need as small as possible): {worst_lb_gap}")
                    assert best_ub_gap >= -1e-4
                    assert best_lb_gap >= -1e-4

                obj_layer.normalize_outrange(grb_lkp1, grb_ukp1)

                lower_bounds.append(grb_lkp1)
                upper_bounds.append(grb_ukp1)


                weights.append(obj_layer)
                # TODO(rudy): Update the last layer to contain the rescaling
                # for its output and add it to `weights`

        prox_end = time.time()
        print(f"Prox timing: {prox_end - prox_start}")

        import IPython; IPython.embed();
        import sys; sys.exit()

    def compute_lower_bound(self, domain):
        pass

class PlanetProblem(Problem):
    '''
    This class is supposed to represent the problem to solve.
    It should have an `init_parameters` method to which we give the problem
    parameters.


    We want to solve:

    max c^T zhat_n
    such that l_0 <= z_0 <= u_0
              zhat_kp1 = W_kp1 z_k + b_kp1 for k=0..n-1
              z_k >= 0
              z_k >= zhat_k    for k=1..n-1
              z_k <= u_k / (u_k - l_k) * (zhat_k - l_k)

    We do the decomposition of the problems into subproblems. We use
    the following ones:

    (A) zhat_kp1 = W_kp1 z_k + b_kp1,    for k=0..n-1
        and bounds on z_k
    (B) z_k >= 0, z_k >= zhat_k, z_k <= u_k / (u_k - l_k) * (zhat_k - l_k)
        for k=1..n-1, and some bounds on zhat_k

    So for each intermediate activation, we need to have two copies of the
    pre-relu and of the post-relu.
    '''

    def get_eta(self, params):
        return self.etas

    def init_parameters(self, weights, obj_layer,
                        lower_bounds, upper_bounds,
                        etas):
        '''
        The responsability of this class are:
        -> To create xs, an object of the shape of a dual variable, containing
           the cost vector split over subproblem.
        -> To set initialized to True, to show that it has been setup properly

        Args:
          input_bounds: (tensor), 2-tuple containing upper and lower bounds
            over the input variables.
          weights: [(w, b)], list of linear layers, each being represented by
            a tuple containing weight and biases.
          {lower, upper}_bounds: Bounds for each activation.
        '''
        self.weights = weights
        self.obj_layer = obj_layer
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Easiest solution to get the xs is to actually set up a primal
        # variable containing the actual cost vector, and then calling
        # to_dual on it, to obtain its decomposition.
        # Let's create the objective functions as primal variables.
        batch_size = 2 * self.obj_layer.out_features
        dims = (weights[0].in_features, ) + tuple(lay.out_features
                                                  for lay in self.weights)
        params = (dims, self.weights, self.lower_bounds, self.upper_bounds)
        cost_vec = PlanetPrimalVar(batch_size, params,
                                   device=lower_bounds[0].device)
        cost_vec.reset_()
        # We have created it. Let's now fill it up with the appropriate value.
        # It's only the last layer of hidden variables that actually have an
        # objective, the other ones just have zero.
        obj = self.obj_layer.weights * self.upper_bounds[-1].unsqueeze(0)
        cost_vec.tensors[-1].copy_(torch.cat([self.obj_layer.weights,
                                              -self.obj_layer.weights], dim=0))

        self.xs = cost_vec.to_dual(with_renorm=True)
        self.initialized = True
        self.etas = etas


class PlanetPrimalVar(Var):
    '''
    This class represents the primal variables of our problem

    To create this, the instantiator needs to be given as first argument
    the batch size, and as second argument all that is necessary information.

    Our responsability is to provide:
    -> an `init_tensors` function, that fills in the `tensors_name`,
      `tensors_size` and `tensors` list.
    -> a `convert_to_primal_solution_` function, that inplace change the
      content of the tensors so as to respect the constraints of the primal
      problem.
    '''

    def init_mapper(self):
        self.mapper = PlanetPDMapper(self.n, device=self.device)

    def init_tensors(self):
        '''
        Use the information that was stored in self.n to specify variable dims.
        '''
        self.tensors_name.append("Input")
        self.tensors_size.append((self.batch, self.n[0][0]))
        self.tensors.append(None)

        for km1, dim in enumerate(self.n[0][1:]):
            k = km1 + 1
            self.tensors_name.append(f"zhat_{k}")
            self.tensors_size.append((self.batch, dim))
            self.tensors.append(None)

            self.tensors_name.append(f"z_{k}")
            self.tensors_size.append((self.batch, dim))
            self.tensors.append(None)

class PlanetDualVar(Var):
    '''
    This class represents the splitted version of our problem

    To create this, the instantiator needs to be given as first argument
    the batch size, and as second argument all that is necessary information.

    Our responsability is to provide:
    -> an `init_tensors` function, that fills in the `tensors_name`,
       `tensors_size` and `tensors` list.
    -> an `argmax` function, that returns the maximum of the linear function
       defined by the content of the tensor, subject to the general constraints
       of the problem (without the constraint that x = x_i for all i)
    '''

    def init_tensors(self):
        '''
        Use the information that was stored in self.n to specify variable dims.
        '''
        self.tensors_name.append("Input")
        self.tensors_size.append((self.batch, self.n[0][0]))
        self.tensors.append(None)

        for km1, dim in enumerate(self.n[0][1:]):
            k = km1 + 1
            self.tensors_name.append(f"zhat_{k}_front")
            self.tensors_size.append((self.batch, dim))
            self.tensors.append(None)
            self.tensors_name.append(f"zhat_{k}_back")
            self.tensors_size.append((self.batch, dim))
            self.tensors.append(None)

            self.tensors_name.append(f"z_{k}_front")
            self.tensors_size.append((self.batch, dim))
            self.tensors.append(None)
            if k < len(self.n[0])-1:
                self.tensors_name.append(f"z_{k}_back")
                self.tensors_size.append((self.batch, dim))
                self.tensors.append(None)

    def check_feasibility(self):
        # Verify that we have a proper, constraints respecting set of
        # variables.
        weights = self.n[1]
        lbs = self.n[2]
        ubs = self.n[3]

        vals = self.tensors

        # Check that the constraint on the input is respected
        x_0 = self.tensors[0]
        assert x_0.min() >= -1
        assert x_0.max() <= 1

        for k in range(0, len(weights)):
            # Check that the constraint of the linear layer is respected.
            lay_k = weights[k]
            zkm1_back_idx = 4 * k  # Can be input as well
            zhatk_front_idx = zkm1_back_idx + 1
            zkm1_back = self.tensors[zkm1_back_idx]
            zhatk_front = self.tensors[zhatk_front_idx]
            # Check the linear constraint
            assert (zhatk_front - lay_k.forward(zkm1_back)).abs().max() < 1e-6
            # Check that the inputs to the linear were in their range.
            if k != 0:
                l_km1 = torch.clamp(lbs[k-1], 0, None)
                u_km1 = torch.clamp(ubs[k-1], 0, None)
                assert (zkm1_back * u_km1 - l_km1).min() >= -1e-6
                assert zkm1_back.min() >= 0
                assert zkm1_back.max() <= 1
                # NOTE: Ideally, we would like to verify that zhatk_front is
                # included between -1 and 1 but we can't actually enforce it
                # (see note in argmax()), so we don't check it.


            # Check that the constraints of the ReLU are respected
            zhatk_back_idx = 4 * k + 2
            zk_front_idx = 4 * k + 3
            zhatk_back = self.tensors[zhatk_back_idx]
            zk_front = self.tensors[zk_front_idx]
            l_k = lbs[k]
            u_k = ubs[k]
            # Check that ReLU output is greater than zero
            assert zk_front.min() >= 0
            # Check that ReLU output is greater than its input
            relu_inp = 0.5 * ((u_k + l_k) + (u_k - l_k) * zhatk_back)
            assert ((u_k > 0).float()*(zk_front - relu_inp/u_k)).min() >= -1e-3
            # For unfixed ReLUs, check that it's smaller than the hull top
            assert (((l_k <= 0) * (u_k >= 0)).float()
                    * (0.5 * (1 + zhatk_back) - zk_front)).min() >= 0
            # For blocked ReLUs, assert output is zero
            assert ((u_k <= 0).float() * zk_front).abs().max() == 0
            # For passing ReLUs, assert output equals to inputs
            assert ((l_k >= 0).float() * (u_k * zk_front - relu_inp)).abs().max() <= 1e-6

            # Assert that the bounds on zhatk are respected.
            assert zhatk_back.min() >= -1
            assert zhatk_back.max() <= 1


    def argmax_v2(self):
        argmax = self.__class__(self).reset_()
        # We're going to use the info we have stuck into self.n to do the
        # correct maximization of the unconstrained problem.
        weights = self.n[1]
        lbs = self.n[2]
        ubs = self.n[3]

        objs = self.tensors

        #TODO(rudy): Update doc
        # Deal with the problem over the input + output of first linear
        # The constraint to respect are:
        #   -1 < x_0 < 1
        #   xhat_1 = condW_1 x_0 + condb_1
        # We consider that there is a cost c_0 over z_0 and a cost chat_1
        # over zhat_1
        # Our solution to this problem is to solve just for x_0, with a cost
        # of   c_0 + chat_1 W_1  , and then deduce xhat_1 from it.
        c_0 = objs[0]
        chat_1 = objs[1]
        lay_1 = weights[0]
        c_eq = c_0 + lay_1.backward(chat_1)
        argmax.tensors[0].copy_(c_eq.sign())
        argmax.tensors[1].copy_(lay_1.forward(argmax.tensors[0]))

        # Now deal with the problem involving ReLUs.
        # We alternate between solving the problem that are:
        #   Relu_convex_hull(zk_hat_back, zk_front)
        # and
        #   zkp1_hat_front = W_kp1 z_k_back + b_kp1
        for k in range(1, len(weights)+1):
            zhat_k_back_idx = 2 + 4*(k-1)
            zk_front_idx = zhat_k_back_idx + 1
            assert self.tensors_name[zhat_k_back_idx] == f"zhat_{k}_back"
            assert self.tensors_name[zk_front_idx] == f"z_{k}_front"
            chat_k = objs[zhat_k_back_idx]
            c_k = objs[zk_front_idx]
            l_k = lbs[k-1]
            u_k = ubs[k-1]
            # We're maximizing a linear function over a polytope. As a
            # result, it will necessarily belong to one of the vertex.
            # We will do the maximization on each of the three (or two) vertices,
            # and keep the best result.
            # Of course, if one of the vertex is not possible due to bounds
            # constraints, we won't use its result, by making the score for this
            # choice -inf
            ambiguous_zero_vertex = - chat_k * (u_k + l_k) / (u_k - l_k)
            ambiguous_zero_vertex.masked_fill_((u_k < 0) + (l_k > 0), -float('inf'))
            possible_passing = (u_k >= 0).float()
            clamped_ratio = torch.clamp(l_k / u_k, 0, None)
            vertex_vals = torch.stack((
                -chat_k + c_k * clamped_ratio * possible_passing,  # min vertex
                chat_k + c_k * possible_passing,                   # max vertex
                ambiguous_zero_vertex                              # 0_vertex
            ))
            _, choice = torch.max(vertex_vals, 0)
            argmax.tensors[zhat_k_back_idx].copy_(-(choice == 0).float() +
                                                  (choice == 1).float() +
                                                  (choice == 2).float() * -1 * (u_k+l_k) / (u_k - l_k))
            argmax.tensors[zk_front_idx].copy_((choice == 0).float() * clamped_ratio *possible_passing +
                                               (choice == 1).float() * possible_passing
                                               # + (choice == 2).float() * 0
                                               )
            assert argmax.tensors[zk_front_idx].min() >= 0
            if k == len(weights):
                # This is the last layer. There is not a following layer.
                break

            ## Otherwise, handle the problem
            ##   zkp1_hat = W_kp1 z_k + b_kp1
            ##   l_k <= z_k <= u_k
            ## where we have a cost of c_k over z_k, and a cost of
            ## chat_kp1 over zkp1_hat
            zk_back_idx = 1 + 3 + 4 * (k-1)
            zkp1_hat_front_idx = 1 + 4 * k
            assert self.tensors_name[zk_back_idx] == f"z_{k}_back"
            assert self.tensors_name[zkp1_hat_front_idx] == f"zhat_{k+1}_front"
            c_k = objs[zk_back_idx]
            chat_kp1 = objs[zkp1_hat_front_idx]
            # Get the bounds on the ReLU outputs
            postl_k = (1/u_k) * torch.clamp(l_k, 0, None)
            postu_k = (1/u_k) * torch.clamp(u_k, 0, None)
            # The solution is very similar to the one involving the linear
            # input. We solve for z_k, with a cost of c_k + chat_kp1 W_kp1
            # and then deduce zkp1_hat from it.
            lay_kp1 = weights[k]
            c_eq = c_k + lay_kp1.backward(chat_kp1)
            argmax.tensors[zk_back_idx].copy_((c_eq > 0).float() * postu_k
                                              + (c_eq <= 0).float() * postl_k)
            argmax.tensors[zkp1_hat_front_idx].copy_(lay_kp1.forward(argmax.tensors[zk_back_idx]))

            # NOTE: It is possible here that the content of
            # argmax.tensors[zkp1_hat_front_idx] will not satisfy our usually
            # expected constraints of being between -1 and 1.
            # The reason is that we're only looking at a subset of the problem,
            # where here we don't have previous constraints over the element of
            # zk_back_idx. As a result, we can get into situations where we
            # couldn't get when we actually computed the bounds.
            # So it's not possible to enforce that the values are between -1 and 1,
            # even though at convergence they should be.

        return argmax

    def argmax(self):
        v2 = self.argmax_v2()
        # v2.check_feasibility()

        # This checks for if we get the exact same results, but in case where
        # some elements of the linear function have a value of zero, this might
        # be too strict a check.
        # for i in range(len(v1.tensors)):
        #     tensor_name = v1.tensors_name[i]
        #     assert (v1.tensors[i] - v2.tensors[i]).abs().max() < 1e-6, f"Error on tensor {tensor_name}"

        # # Instead, just compare the value obtained by the argmax.
        # v1_score = self.dot(v1)
        # v2_score = self.dot(v2)
        # if (v2_score - v1_score).abs().max() > 1e-5:
        #     # There is non-negligible gap between the two values.
        #     # Let's assert that it is relatively small compared to the values.
        #     if ((v1_score - v2_score) / v1_score).abs().max() > 1e-3:
        #         raise Exception("Significant gap between the two methods.")
        return v2

class PlanetPDMapper:
    '''
    This class allows to jump from a primal version to a dual version of the
    problem.

    We need to have a `__init__` function, to setup the shape of the mapper and
    pass it any necessary arguments.
    It also needs to have a `to_primal` and a `to_dual` functions, which are
    terrible names for these function
    '''

    def __init__(self, n, device=None):
        if device is None:
            self.device = torch.rand(1).device
        self.n = n

    def split(self, constrained, with_renorm=True):
        '''
        Take the primal variable `constrained` and duplicate it into all the
        dual variables that it corresponds to.
        '''
        assert isinstance(constrained, PlanetPrimalVar)

        # If with_renorm is True, divide every value by the number of copies
        # there is.
        splitted = PlanetDualVar(constrained.batch, constrained.n,
                                 self, device=constrained.device)
        splitted.reset_()
        # The first set of variables consist in the input variables, who belong
        # to only a single set of constraint.
        splitted.tensors[0].copy_(constrained.tensors[0])
        # Then, all but the last set of post ReLUs are duplicated, once in a
        # linear constraint, once in a convex hull of ReLU constraint.
        act_idx = 1
        while act_idx < len(constrained.tensors):
            idx_in_primal = act_idx
            frontidx_in_dual = 1 + 2*(act_idx-1)
            backidx_in_dual = frontidx_in_dual + 1

            to_copy = constrained.tensors[act_idx]
            if with_renorm and (act_idx != len(constrained.tensors)-1):
                to_copy = to_copy / 2
            if (act_idx != len(constrained.tensors)-1):
                splitted.tensors[backidx_in_dual].copy_(to_copy)
            splitted.tensors[frontidx_in_dual].copy_(to_copy)
            act_idx += 1

        # Just making sure we've gone through it all
        assert backidx_in_dual == len(splitted.tensors)

        return splitted

    to_dual = split

    def gather(self, unrestricted):
        '''
        Take the dual variable `unconstrained`, containing all the different
        copies for each variable and return a primal variable where all the
        copies have been summed.
        '''
        assert isinstance(unrestricted, PlanetDualVar)

        constrained = PlanetPrimalVar(unrestricted.batch, unrestricted.n,
                                      self, device=unrestricted.device)
        constrained.reset_()

        # First tensor is the input variables which are not duplicated
        constrained.tensors[0].copy_(unrestricted.tensors[0])

        # All but the last tensors are duplicated
        act_idx = 1
        while act_idx < len(constrained.tensors):
            idx_in_primal = act_idx
            frontidx_in_dual = 1 + 2*(act_idx-1)
            backidx_in_dual = frontidx_in_dual + 1

            target = constrained.tensors[act_idx]
            if (act_idx < len(constrained.tensors)-1):
                target.copy_(unrestricted.tensors[frontidx_in_dual]
                             + unrestricted.tensors[backidx_in_dual])
            else:
                target.copy_(unrestricted.tensors[frontidx_in_dual])
            act_idx += 1
        assert backidx_in_dual == len(unrestricted.tensors)

        return constrained

    to_primal = gather
