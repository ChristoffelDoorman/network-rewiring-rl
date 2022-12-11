import argparse
import csv
import json
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append('/relnet')
sys.path.append('/analyses')
sys.path.append(str(Path(__file__).parent.parent))

from analyses.utils import construct_graphs, generate_model_list, NumpyArrayEncoder
from analyses.utils import create_MDP, test_performance, ci
from analyses.attack_simulation.simulation_utils import grade_graph
from relnet.objective_functions.objective_functions import MERW, GlobalEntropy

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

if __name__ == '__main__':
    exp_date = '220525'
    n_graphs = 100

    parser = argparse.ArgumentParser(
        description="Script to compute objective functions for different budgets and graph sizes")
    parser.add_argument("--obj_name", required=True, choices=['Shannon', 'MERW'], help="Objective function", type=str)
    parser.add_argument("--n_nodes", help="Number of nodes. If not specified, default list is used", type=int)
    parser.add_argument("--graph_type", help="Type of the graph", type=str)
    parser.add_argument("--attack_scenario", action='store_true',
                        help="Indicate if attack scenario should be simulated and recorded")
    parser.add_argument("--budget", help="Budget. If not specified, default list is used", type=int)

    args = parser.parse_args()
    obj_nm = args.obj_name
    graph_type = args.graph_type
    attack_scenario = args.attack_scenario

    if args.n_nodes is not None:
        n_nodes_list = [args.n_nodes]
    else:
        n_nodes_list = [10, 20, 30, 40, 50]

    if args.budget is not None:
        budget_list = [args.budget]
    else:
        budget_list = [5, 10, 15, 20, 25]

    agent_name = f'{obj_nm}-DQN'
    exp_name = f'{obj_nm}_{graph_type}'
    dir_path = os.path.join(script_dir, f'models/{exp_name}/exp220518/')
    model_list, hyperparams_subst = generate_model_list(dir_path)

    appdx = f'{n_nodes_list[0]}'
    if len(n_nodes_list) > 1:
        appdx += f'-{n_nodes_list[-1]}n'
    else:
        appdx += 'n'

    if len(budget_list) == 1:
        appdx += f'_{budget_list[0]}b'
    if attack_scenario:
        appdx += '_wAttck'

    outfile = os.path.join(script_dir, f'results/exp{exp_date}/DQN_{exp_name}_{appdx}.csv')

    with open(outfile, 'w') as mkfile:
        writer = csv.writer(mkfile)
        if attack_scenario:
            writer.writerow(['costMN', 'costCI', 'graph', 'nodes', 'agent'])
            # writer.writerow(['time', 'entropy', 'RWscore', 'RWcost', 'RWattackers', 'RWdefenders', 'graph', 'method', 'seeds', 'nodes', 'budget', 'agent'])
        else:
            writer.writerow(['time', 'entropy', 'graph', 'method', 'seeds', 'nodes', 'budget', 'agent'])

    if obj_nm == 'MERW':
        obj_fn = MERW()
    elif obj_nm == 'Shannon':
        obj_fn = GlobalEntropy()

    for n_nodes in n_nodes_list:
        graphs_list = construct_graphs(n_graphs, n_nodes, graph_type, 'test')
        for budget in budget_list:
            print(f'Nodes: {n_nodes}, budget: {budget}, graph: {graph_type}, obj_fn: {obj_nm}')
            graphs = graphs_list.copy()
            delta_entr, org_graphs_multiple_seeds, rew_graphs_multiple_seeds, delta_t = test_performance(graphs, obj_fn,
                                                                                                         model_list,
                                                                                                         budget,
                                                                                                         hyperparams_subst,
                                                                                                         measure_time=True)
            scores_1seed, costs_1seed, attcks_1seed, defends_1seed = [], [], [], []

            if attack_scenario:
                print("Executing attack scenarios...")
                tot_costs = []
                for org_graph_single_seed, rew_single_seed in zip(org_graphs_multiple_seeds, rew_graphs_multiple_seeds):
                    org_graphs = [g.to_networkx() for g in org_graph_single_seed]
                    rew_graphs = [g.to_networkx() for g in rew_single_seed]
                    all_scores, all_costs, all_attcks, all_defends = [], [], [], []
                    for g_org, g_rew in zip(org_graphs, rew_graphs):
                        attacker, defender, scores, this_graph_all_costs = grade_graph(g_org, g_rew,
                                                                                       find_missing_nodes=True,
                                                                                       exhautive_search=False)
                        tot_graph_costs = []
                        i0 = 0
                        for c_idx in scores:
                            tot_graph_costs.append(sum(this_graph_all_costs[i0: i0 + c_idx]) / n_nodes)
                            i0 = i0 + c_idx

                        assert len(tot_graph_costs) == len(scores)
                        tot_costs += tot_graph_costs

                tot_norm_costs_mn = np.mean(tot_costs)
                tot_norm_costs_ci = ci(tot_costs)
                with open(outfile, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([tot_norm_costs_mn, tot_norm_costs_ci, graph_type, n_nodes, agent_name])

            else:
                with open(outfile, 'a') as f:
                    writer = csv.writer(f)
                    for seed in range(len(delta_entr)):
                        writer.writerow(
                            [delta_t[seed], delta_entr[seed], graph_type, obj_nm, seed, n_nodes, budget, agent_name])
