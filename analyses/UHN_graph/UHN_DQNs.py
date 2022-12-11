import argparse
import csv
import networkx as nx
import numpy as np
import os
import random
import sys
from pathlib import Path

sys.path.append('/relnet')
sys.path.append('/analyses')
sys.path.append(str(Path(__file__).parent.parent))

from analyses.utils import create_MDP, rewire, generate_model_list, ci
from analyses.attack_simulation.simulation_utils import grade_graph
from analyses.baselines import *

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

if __name__ == '__main__':
    exp_date = '220518'
    obj_names = ['Shannon', 'MERW']
    graph_types = ['ba', 'ba1', 'er', 'ws']
    budget = 15

    G_LAN = nx.read_graphml(os.path.join(script_dir, './LAN_processed.graphml'), node_type=int)
    N = G_LAN.number_of_nodes()
    outfile = os.path.join(script_dir, f'LAN_results.csv')

    with open(outfile, 'w') as mkfile:
        writer = csv.writer(mkfile)
        writer.writerow(['agent', 'method', 'RWTotalNormCost', 'RWTotalNormCostCI95'])

    random_results = []
    dqn_results = {}
    rnd_seed = 0
    for obj_nm in obj_names:
        g = G_LAN.copy()

        # Do and measure random rewiring
        obj_fn = ObjFn(obj_nm)
        RND_entr_0, RND_entr_T, _, RND_rewired_graph = random_rewiring(g, (N * budget / 100), obj_fn, rnd_seed,
                                                                       print_info=False)
        rnd_attck, rnd_dfdr, rnd_scores, rnd_costs = grade_graph(G_LAN, RND_rewired_graph, find_missing_nodes=True,
                                                                 exhautive_search=True)
        rnd_seed += 1
        rnd_tot_costs = []
        i0 = 0
        for c_idx in rnd_scores:
            rnd_tot_costs.append(sum(rnd_costs[i0: i0 + c_idx]) / N)
            i0 = i0 + c_idx

        assert len(rnd_tot_costs) == len(rnd_scores)
        random_results += rnd_tot_costs

        for g_type in graph_types:
            print(f"Running {exp_name}...")
            exp_name = f'{obj_nm}_{g_type}'
            g = G_LAN.copy()
            dir_path = os.path.join(script_dir, f'models/{exp_name}/')
            model_list, hyperparams_subst = generate_model_list(dir_path)
            dqn_results[exp_name] = []
            for seed, model in enumerate(model_list):
                print(f'seed {seed + 1}/{len(model_list)}...')
                agent = create_MDP(model, seed, budget, hyperparams_subst)
                DQN_rewired_graph, dqn_conn = agent.execute_rewiring_strategy([g], convert_from_networkx=True)

                if len(dqn_conn) == 0:
                    print("DQN rewiring disconnected the graph!")
                    continue

                DQN_rewired_graph = DQN_rewired_graph[0].to_networkx()
                print("Attack simulation...")
                dqn_attck, dqn_dfdr, dqn_scores, dqn_costs = grade_graph(G_LAN, DQN_rewired_graph,
                                                                         find_missing_nodes=True,
                                                                         exhautive_search=True)
                dqn_tot_costs = []
                i0 = 0
                for c_idx in dqn_scores:
                    dqn_tot_costs.append(sum(dqn_costs[i0: i0 + c_idx]) / N)
                    i0 = i0 + c_idx

                assert len(dqn_tot_costs) == len(dqn_scores)
                dqn_results[exp_name] += dqn_tot_costs

            RWC_DQN_mn = np.mean(dqn_results[exp_name])
            RWC_DQN_ci = ci(dqn_results[exp_name])

            print('--' * 30)
            print(f"{exp_name}-DQN Random Walk Costs: {RWC_DQN_mn} ± {RWC_DQN_ci}")
            print(f"Random Agent Random Walk Costs: {np.mean(random_results)} ± {ci(random_results)}")
            print('--' * 30)

            with open(outfile, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([exp_name, obj_nm, RWC_DQN_mn, RWC_DQN_ci])

    with open(outfile, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['RND', obj_nm, np.mean(random_results), ci(random_results)])
