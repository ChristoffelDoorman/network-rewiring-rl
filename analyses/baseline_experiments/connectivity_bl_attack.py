import argparse
import csv
import networkx as nx
import numpy as np
import os
import random
import scipy as sp
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append('/relnet')
sys.path.append('/analyses')
sys.path.append(str(Path(__file__).parent.parent.parent))

from analyses.time_evaluation.baselines import *
from analyses.utils import ci
from analyses.attack_simulation.simulation_utils import grade_graph

sys.path.append(os.getcwd())
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in


def get_laplacian(A):
    degs = A.sum(axis=1).reshape(-1)
    D = np.diagflat(degs)
    L = D - A
    return L


def compute_fiedler_vector(L, v0=None, k=2, maxiter=None):
    try:
        _, eigvs = sp.sparse.linalg.eigsh(L, k=2, which='SM', return_eigenvectors=True, v0=v0, maxiter=maxiter)
    except sp.sparse.linalg.ArpackNoConvergence:
        print("Fiedler vector error...")
        return compute_fiedler_vector(L, v0, k=k + 1, maxiter=5000)

    fiedler_vector = eigvs[:, 1]
    return fiedler_vector


def alg_connectivity_rewiring(g, budget, print_info=False):
    remaining_budget = int(budget)

    while remaining_budget >= 1:
        disconnects_graph = True

        edges = list(g.edges)
        non_edges = list(nx.non_edges(g))

        A = nx.to_numpy_matrix(G=g)
        L = get_laplacian(A)
        fiedler_vector = compute_fiedler_vector(L)

        edge_scores = list(map(lambda pair: (fiedler_vector[pair[0]] - fiedler_vector[pair[1]]) ** 2, edges))
        non_edge_scores = list(map(lambda pair: (fiedler_vector[pair[0]] - fiedler_vector[pair[1]]) ** 2, non_edges))

        # choose removal edge
        removal_edge_idx = np.argmax(edge_scores)
        bn1, bn2 = edges[removal_edge_idx][0], edges[removal_edge_idx][1]

        addible_edges = [idx for idx, e in enumerate(non_edges) if bn1 in e or bn2 in e]

        while disconnects_graph:

            # choose removal edge
            if len(addible_edges) == 0:
                if print_info:
                    print("Choosing new removal edge!")
                del edges[removal_edge_idx]
                del edge_scores[removal_edge_idx]
                if len(edges) == 0:
                    break
                removal_edge_idx = np.argmax(edge_scores)
                bn1, bn2 = edges[removal_edge_idx][0], edges[removal_edge_idx][1]
                addible_edges = [idx for idx, e in enumerate(non_edges) if bn1 in e or bn2 in e]

            max_addition_gain = np.inf
            for idx in addible_edges:
                non_edge = non_edges[idx]
                if bn1 in non_edge and non_edge_scores[idx] < max_addition_gain:
                    edge2add = non_edge
                    max_addition_gain = non_edge_scores[idx]
                    base_node = bn1
                    remv_node = bn2
                    chosen_idx = idx
                elif bn2 in non_edge and non_edge_scores[idx] < max_addition_gain:
                    edge2add = non_edge
                    max_addition_gain = non_edge_scores[idx]
                    base_node = bn2
                    remv_node = bn1
                    chosen_idx = idx

            # get addition node
            targ_node = edge2add[0] if base_node == edge2add[1] else edge2add[1]

            g_temp = g.copy()
            g_temp = rewire_graph(g_temp, base_node, remv_node, targ_node)
            if nx.is_connected(g_temp):
                disconnects_graph = False
                g = g_temp
            else:
                if print_info:
                    print(
                        f"This move disconnects the graph! Remaining budget: {int(remaining_budget)}/{budget}.   {len(non_edges)}")
                addible_edges.remove(chosen_idx)

        remaining_budget -= 1

    rew_acts = budget - remaining_budget

    if print_info:
        print("Rewiring operations:", int(rew_acts))

    return rew_acts, g


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Algebraic Connectivity baseline on synthetic graphs')
    parser.add_argument('-g', '--graph_type', help='Graph type', type=str, choices=['ba', 'ba1', 'er', 'ws'],
                        required=True)
    parser.add_argument('-n', '--n_nodes', help='Number of nodes', type=int, required=True)
    parser.add_argument('-r', '--node_range', help='Run over multiple nodes', type=bool, required=False)
    args = parser.parse_args()

    exp_date = '221103'
    n_graphs = 100

    budget_perct = 15

    graph_type = args.graph_type
    nodes = args.n_nodes

    if args.node_range:
        node_range = [10, 20, 30, 40, 50, 75, 100]
        outfile = os.path.join(script_dir,
                               f'./results/exp{exp_date}/ConnectivityBL_Attack_{graph_type}_15b_{min(node_range)}-{max(node_range)}n.csv')
    else:
        node_range = [nodes]
        outfile = os.path.join(script_dir,
                               f'./results/exp{exp_date}/ConnectivityBL_Attack_{graph_type}_15b_{nodes}n.csv')

    with open(outfile, 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['costMN', 'costCI', 'graph', 'nodes', 'agent'])

    for n_nodes in node_range:
        if graph_type == 'ba':
            graph_gen = nx.generators.random_graphs.barabasi_albert_graph
            graph_params = {'m': 2}
        elif graph_type == 'er':
            graph_gen = nx.generators.gnp_random_graph
            graph_params = {'p': 0.15}
        elif graph_type == 'ws':
            graph_gen = nx.generators.connected_watts_strogatz_graph
            graph_params = {'k': 4, 'p': 0.1}
        elif graph_type == 'ba1':
            graph_gen = nx.generators.random_graphs.barabasi_albert_graph
            graph_params = {'m': 1}

        F = ObjFn('Shannon')

        print(f"Running Random agent on {graph_type} (n={n_nodes}) with budget {budget_perct}...")
        tot_costs = []
        for s in tqdm(range(n_graphs), position=0, leave=True):
            # print(f"Seed: {s+1}/{n_graphs}")
            for extra_seed in range(10, 20):
                graph_params['n'] = n_nodes
                graph_params['seed'] = s + 800
                g_org = graph_gen(**graph_params)
                m = g_org.number_of_edges()

                budget = int(m * budget_perct / 100)

                # do rewiring
                _, g_rew = alg_connectivity_rewiring(g_org.copy(), budget, print_info=False)

                # attack simulation evaluation
                attacker, defender, scores, this_graph_all_costs = grade_graph(g_org, g_rew, find_missing_nodes=True,
                                                                               exhautive_search=False)
                tot_graph_costs = []
                i0 = 0
                for c_idx in scores:
                    tot_graph_costs.append(sum(this_graph_all_costs[i0: i0 + c_idx]) / n_nodes)
                    i0 = i0 + c_idx

                assert len(tot_graph_costs) == len(scores)

                tot_costs += tot_graph_costs

        with open(outfile, 'a') as of:
            writer = csv.writer(of)
            writer.writerow([np.mean(tot_costs), ci(tot_costs), graph_type, n_nodes, 'Connectivity'])
