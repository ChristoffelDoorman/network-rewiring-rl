import argparse
import csv
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append('/relnet')
sys.path.append('/analyses')
sys.path.append(str(Path(__file__).parent.parent))

from analyses.attack_simulation.simulation_utils import grade_graph
from analyses.utils import create_MDP, construct_graphs, generate_model_list, rewire, NumpyArrayEncoder

sys.path.append(os.getcwd())
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

if __name__ == '__main__':
    n_graphs = 100
    exp_date = '220514'

    parser = argparse.ArgumentParser(description="Script to run attack scenario for given graph size and budget")
    parser.add_argument("--n_nodes", help="Number of nodes. If not specified, default list is used", type=int)
    parser.add_argument("--obj_name", required=True, choices=['Shannon', 'MERW'], help="Objective function", type=str)
    parser.add_argument("--budget_perc", help="Budget [percentage of number of edges]", type=int)
    parser.add_argument("--graph_type", help="Type of the graph", type=str)

    args = parser.parse_args()
    n_nodes = args.n_nodes
    obj_nm = args.obj_name
    budget_perct = args.budget_perc
    graph_type = args.graph_type

    exp_name = f'{obj_nm}_{graph_type}'
    dir_path = os.path.join(script_dir, f'./models/{obj_nm}_{graph_type}/')
    outfile = os.path.join(script_dir,
                           f'./results/exp{exp_date}/Attck_{obj_nm}_{graph_type}_{budget_perct}b_{n_nodes}n.csv')

    with open(outfile, 'w') as mkfile:
        writer = csv.writer(mkfile)
        writer.writerow(
            ['RWscore', 'RWcost', 'RWattackers', 'RWdefenders', 'graph', 'method', 'seeds', 'nodes', 'budget'])

    model_list, hyperparams_subst = generate_model_list(dir_path)
    graphs_list = construct_graphs(n_graphs, n_nodes, graph_type, data_set='test')

    print(f'Obj_fn: {obj_nm}, graph: {graph_type}, nodes: {n_nodes}, budget: {budget_perct}')
    graphs = graphs_list.copy()
    graphseed = 0
    for modelseed, model in enumerate(model_list):
        agent = create_MDP(model, modelseed, budget_perct, hyperparams_subst=hyperparams_subst)
        org_graphs, rew_graphs = rewire(agent, graphs, to_networkx=True)

        for org_g, rew_g in zip(org_graphs, rew_graphs):
            print(f"{graphseed + modelseed + 1}/{len(model_list) * n_graphs}...")
            attacker, defender, scores, this_graph_all_costs = grade_graph(org_g, rew_g, find_missing_nodes=True,
                                                                           exhautive_search=False)
            mean_score = 0 if len(scores) == 0 else np.mean(scores)
            mean_cost = 0 if len(this_graph_all_costs) == 0 else np.mean(this_graph_all_costs)

            with open(outfile, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [mean_score, mean_cost, attacker, defender, graph_type, obj_nm, modelseed + graphseed, n_nodes,
                     budget_perct])

            graphseed += 1

    print(f"{graphseed + modelseed + 1}/{len(model_list) * n_graphs}... Done!")
