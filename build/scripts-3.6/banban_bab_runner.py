#!/home/jodie/miniconda3/envs/verif/bin/python
import argparse

from plnn.branch_and_bound import bab
from plnn.banban_lp.banban_linear_approximation import NetworkLP
from plnn.model import load_and_simplify, load_adversarial_problem

def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")
    parser.add_argument('network_filename', type=str,
                        help='.rlv file to prove.')
    parser.add_argument('--reluify_maxpools', action='store_true')
    parser.add_argument('--smart_branching', action='store_true')
    args = parser.parse_args()

    if args.network_filename.endswith('.rlv'):
        rlv_infile = open(args.network_filename, 'r')
        network, domain = load_and_simplify(rlv_infile,
                                            NetworkLP)
        rlv_infile.close()
    else:
        network, domain = load_adversarial_problem(args.network_filename,
                                                   NetworkLP)

    epsilon = 1e-2
    decision_bound = 0
    min_lb, min_ub, ub_point, nb_visited_states = bab(network, domain,
                                                      epsilon, decision_bound)

    if min_lb >= 0:
        print("UNSAT")
    elif min_ub < 0:
        # Verify that it is a valid solution
        candidate_ctx = ub_point.view(1,-1)
        val = network.net(candidate_ctx)
        margin = val.squeeze().item()
        if margin > 0:
            print("Error")
        else:
            print("SAT")
        print(ub_point)
        print(margin)
    else:
        print("Unknown")
    print(f"Nb states visited: {nb_visited_states}")


if __name__ == '__main__':
    main()
