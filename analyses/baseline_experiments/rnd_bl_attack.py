import sys
from pathlib import Path

sys.path.append('/relnet')
sys.path.append('/analyses')
sys.path.append(str(Path(__file__).parent.parent.parent))

from analyses.baselines import *

from analyses.utils import ci

from analyses.attack_simulation.simulation_utils import grade_graph

sys.path.append(os.getcwd())
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

if __name__ == '__main__':
    exp_date = '220518'
    n_graphs = 100

    budget_perct = 15

    outfile = os.path.join(script_dir, f'./results/exp{exp_date}/RandomBL_Attack_30n_15b_all.csv')
    with open(outfile, 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['costMN', 'costCI', 'graph', 'nodes', 'agent'])

    for graph_type in ['er']:
        for n_nodes in [10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300]:
            if graph_type == 'er' and n_nodes > 100:
                continue

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
                    _, _, _, g_rew = random_rewiring(g_org.copy(), budget, F, seed=s * extra_seed, print_info=False)

                    # attack simulation evaluation
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

            with open(outfile, 'a') as of:
                writer = csv.writer(of)
                writer.writerow([np.mean(tot_costs), ci(tot_costs), graph_type, n_nodes, 'RND'])
