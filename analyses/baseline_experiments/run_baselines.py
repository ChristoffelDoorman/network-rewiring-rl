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
    exp_date = '220525'
    n_graphs = 10

    parser = argparse.ArgumentParser(
        description="Script to compute time complexity, entropy gain and random walk costs of baselines")
    parser.add_argument("--n_nodes", help="Number of nodes. If not specified, default list is used", type=int)
    parser.add_argument("--obj_name", required=True, choices=['Shannon', 'MERW'], help="Objective function", type=str)
    parser.add_argument("--budget_perc", help="Budget [percentage of number of edges]", type=int)
    parser.add_argument("--graph_type", help="Type of the graph", type=str)
    parser.add_argument("--baseline", choices=['random', 'heuristic'], help="Baseline to compute", type=str)
    parser.add_argument("--attack_scenario", action='store_true',
                        help="Indicate if attack scenario should be simulated and recorded")

    args = parser.parse_args()
    n_nodes = args.n_nodes
    obj_nm = args.obj_name
    budget_perct = args.budget_perc
    graph_type = args.graph_type
    baseline = args.baseline
    attack_scenario = args.attack_scenario

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

    F = ObjFn(obj_nm)

    appdx = f'_wAttack' if attack_scenario else '_noAttack'
    outfile = os.path.join(script_dir,
                           f'./results/exp{exp_date}/{baseline}BL_{obj_nm}_{graph_type}_{budget_perct}b_{n_nodes}n{appdx}.csv')
    with open(outfile, 'w') as fd:
        writer = csv.writer(fd)
        if attack_scenario:
            writer.writerow(
                ['time', 'entropy', 'RWscoreMN', 'RWcostCI', 'RWattackers', 'RWdefenders', 'graph', 'method', 'seeds',
                 'nodes', 'budget', 'agent'])
        else:
            writer.writerow(['time', 'entropy', 'graph', 'method', 'seeds', 'nodes', 'budget', 'agent'])

    print(f"Running {baseline}-{obj_nm} baseline on {graph_type} (n={n_nodes}) with budget {budget_perct}...")
    tot_costs = []
    for s in tqdm(range(n_graphs), position=0, leave=True):
        print(f"Seed: {s + 1}/{n_graphs}")
        graph_params['n'] = n_nodes
        graph_params['seed'] = s + 800
        g_org = graph_gen(**graph_params)
        m = g_org.number_of_edges()

        budget = int(m * budget_perct / 100)

        # do rewiring
        if baseline == 'heuristic':
            start = time.time()
            entr_0, entr_T, rew_acts, g_rew = exhaustive_rewiring(g_org.copy(), budget, F, print_info=False)
            end = time.time()
        elif baseline == 'random':
            start = time.time()
            entr_0, entr_T, rew_acts, g_rew = random_rewiring(g_org.copy(), budget, F, seed=s, print_info=False)
            end = time.time()

        delta_t = end - start
        delta_entr = entr_T - entr_0

        # attack simulation evaluation
        tot_graph_costs = []
        if attack_scenario:
            attacker, defender, scores, this_graph_all_costs = grade_graph(g_org, g_rew, find_missing_nodes=True,
                                                                           exhautive_search=False)
            for c_idx in scores:
                tot_graph_costs.append(sum(this_graph_all_costs[i0: i0 + c_idx]) / n_nodes)
                i0 = i0 + c_idx

            assert len(tot_graph_costs) == len(scores)
            tot_costs += tot_graph_costs

            with open(outfile, 'a') as of:
                writer = csv.writer(of)
                writer.writerow(
                    [delta_t, delta_entr, np.mean(tot_costs), ci(tot_costs), attacker, defender, graph_type, obj_nm, s,
                     n_nodes, budget_perct, baseline])

        # without attack sim eval
        else:
            with open(outfile, 'a') as of:
                writer = csv.writer(of)
                writer.writerow([delta_t, delta_entr, graph_type, obj_nm, s, n_nodes, budget_perct, baseline])
