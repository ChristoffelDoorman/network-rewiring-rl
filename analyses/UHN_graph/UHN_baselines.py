import os
import sys
from pathlib import Path

sys.path.append('/relnet')
sys.path.append('/analyses')
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.getcwd())

from analyses.attack_simulation.simulation_utils import grade_graph
from analyses.baselines import *
from analyses.baseline_experiments.connectivity_bl_attack import *

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to compute random walk costs for UHN graphs")
    parser.add_argument("--obj_name", required=True, choices=['Shannon', 'MERW'], help="Objective function", type=str)
    parser.add_argument("--budget_perc", help="Budget [percentage of number of edges]", type=int)
    parser.add_argument("--graph_name", help="Type of the graph", type=str)
    parser.add_argument("--baseline", choices=['random', 'heuristic', 'MinConnectivity'], help="Baseline to compute",
                        type=str)
    parser.add_argument("--attack_scenario", choices=['yes', 'no'],
                        help="Indicate if attack scenario should be simulated and recorded", type=str)

    args = parser.parse_args()
    obj_nm = args.obj_name
    budget_perct = args.budget_perc
    graph_name = args.graph_name
    baseline = args.baseline
    attack_scenario = args.attack_scenario

    F = ObjFn(obj_nm)

    if attack_scenario == 'yes':
        outfile = os.path.join(script_dir,
                               f'./results/LAN_results/{baseline}BL_withAttck_{obj_nm}_LAN{graph_name}_{budget_perct}b.csv')
        with open(outfile, 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(
                ['time', 'RWscore', 'RWcost', 'RWattackers', 'RWdefenders', 'graph', 'method', 'nodes',
                 'edges', 'budget', 'rewireActions'])
    else:
        outfile = os.path.join(script_dir,
                               f'./results/LAN_results/{baseline}BL_{obj_nm}_LAN{graph_name}_{budget_perct}b.csv')
        with open(outfile, 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(
                ['time', 'graph', 'method', 'nodes', 'edges', 'budget', 'rewireActions'])

    g_org = nx.readwrite.read_graphml(path=os.path.join(script_dir, f'./lan_graphs/lan{graph_name}.graphml'))
    m = g_org.number_of_edges()
    n = g_org.number_of_nodes()

    budget = int(m * budget_perct / 100)

    # do rewiring
    if baseline == 'heuristic':
        start = time.time()
        _, _, rew_acts, g_rew = exhaustive_rewiring(g_org.copy(), budget, F, print_info=False,
                                                    allow_tails=True)
        end = time.time()
    elif baseline == 'random':
        start = time.time()
        _, _, rew_acts, g_rew = random_rewiring(g_org.copy(), budget, F, print_info=False)
        end = time.time()
    elif baseline == 'MinConnectivity':
        start = time.time()
        rew_acts, g_rew = alg_connectivity_rewiring(g_org.copy(), budget, print_info=False)
        end = time.time()

    print("time: ", delta_t)
    print("--" * 20)

    # attack simulation evaluation
    if attack_scenario == 'yes':
        attacker, defender, scores, this_graph_all_costs = grade_graph(g_org, g_rew, find_missing_nodes=True,
                                                                       exhautive_search=True)
        mean_score = np.mean(scores)
        mean_cost = np.mean(this_graph_all_costs)
        print("Attack simulation")
        print(r'Score: {:.4f} ± {:.4f}'.format(mean_score, np.std(scores)))
        print(r'Cost: {:.4f} ± {:.4f}'.format(mean_cost, np.std(this_graph_all_costs)))

    with open(outfile, 'a') as of:
        writer = csv.writer(of)
        if attack_scenario == 'yes':
            writer.writerow(
                [delta_t, mean_score, mean_cost, attacker, defender, graph_name, obj_nm, n, m, budget_perct,
                 rew_acts])
        else:
            writer.writerow([delta_t, graph_name, obj_nm, n, m, budget_perct, rew_acts])
