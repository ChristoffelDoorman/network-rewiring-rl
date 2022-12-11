import json
import os
import pickle
import time
from json import JSONEncoder
from pathlib import Path

from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.objective_functions.objective_functions import *
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WSNetworkGenerator, \
    ERNetworkGenerator, BA1NetworkGenerator


def construct_graphs(n_graphs, nodes, graph_type, data_set):
    if data_set == 'train':
        seeds, _, _ = NetworkGenerator.construct_network_seeds(n_graphs, 200, 200)

    elif data_set == 'validation':
        _, seeds, _ = NetworkGenerator.construct_network_seeds(600, n_graphs, 200)

    elif data_set == 'test':
        _, _, seeds = NetworkGenerator.construct_network_seeds(600, 200, n_graphs)

    if graph_type == 'ba':
        graph_params = {'m_ba': 2}
        gen = BANetworkGenerator(**{'store_graphs': True, 'graph_storage_root': Path('/experiment_data/stored_graphs')})
    elif graph_type == 'ws':
        graph_params = {'k_ws': 4, 'p_ws': 0.1}
        gen = WSNetworkGenerator(**{'store_graphs': True, 'graph_storage_root': Path('/experiment_data/stored_graphs')})
    elif graph_type == 'er':
        graph_params = {'p_er': 0.15}
        gen = ERNetworkGenerator(**{'store_graphs': True, 'graph_storage_root': Path('/experiment_data/stored_graphs')})
    elif graph_type == 'ba1':
        graph_params = {'m_ba': 1}
        gen = BA1NetworkGenerator(
            **{'store_graphs': True, 'graph_storage_root': Path('/experiment_data/stored_graphs')})

    graph_params['n'] = nodes
    graph_params['draw_graphs'] = False
    graphs = gen.generate_many(graph_params, seeds)
    return graphs


def create_MDP(model_path, seed, budget, hyperparams_subst=None):
    # create environment
    env = GraphEdgeEnv(NoObjFn(), {"random_seed": seed}, budget, None)  # create environment
    env.training = False

    # create agent
    agent = RNetDQNAgent(env)

    # set hyper parameters for this sweep
    default_hyperparams = agent.get_default_hyperparameters()
    hyperparams = default_hyperparams.copy()

    if hyperparams_subst is not None:
        # print("Setting hyperparams...")
        hyperparams.update(hyperparams_subst)

    agent.setup({"random_seed": seed}, hyperparams)
    agent.initialize_from_model(model_path)

    return agent


def rewire(agent, graphs, random=False, to_networkx=False):
    # rewire
    if not random:
        rewired_full, conn = agent.execute_rewiring_strategy(graphs)
    else:
        rewired_full, conn = agent.execute_random_rewiring_strategy(graphs)

    # throw away disconnected and convert to networkx
    if to_networkx:
        rewired = [rg.to_networkx() for i, rg in enumerate(rewired_full) if i in conn]
        originals = [og.to_networkx() for j, og in enumerate(graphs) if j in conn]

    # only throw away disconnected
    else:
        rewired = [rg for i, rg in enumerate(rewired_full) if i in conn]
        originals = [og for j, og in enumerate(graphs) if j in conn]

    return originals, rewired


def measure(graphs, metric, n_graphs):
    vals = []
    for g in graphs:
        vals.append(metric.compute(g))

    for pad in range(n_graphs - len(vals)):
        vals.append(np.nan)

    return vals


def ci(values, acc=1.96, axis=None):
    """ Compute confidence interval of array """
    if axis is not None:
        # return acc * np.nanstd(values, axis=axis) / np.sqrt(values.shape[axis])
        return acc * np.nanstd(values, axis=axis) / np.sqrt(np.count_nonzero(~np.isnan(values), axis=axis))
    else:
        return acc * np.nanstd(values) / np.sqrt(np.count_nonzero(~np.isnan(values)))


def test_performance(graphs, metric, model_list, budget, hyperparams_subs, measure_time=False):
    entropy_gains = []
    entropy_gains_total = []
    entropy_ci = []
    time_measurements = []

    original_graphs_multiple_seeds = []
    rewired_graphs_multiple_seeds = []
    for seed, model in enumerate(model_list):
        print(f'seed {seed + 1}/{len(model_list)}...')
        agent = create_MDP(model, seed, budget, hyperparams_subs)

        start = time.time()
        original_graphs, rewired_graphs = rewire(agent, graphs, random=False, to_networkx=False)
        end = time.time()

        time_p_graph = (end - start) / len(graphs)
        time_measurements.append(time_p_graph)

        if len(rewired_graphs) == 0:
            entropy_gains.append(np.nan)
            continue
        else:
            delta_entr = []
            for g_0, g_T in zip(original_graphs, rewired_graphs):
                entr_0 = metric.compute(g_0)
                entr_T = metric.compute(g_T)
                delta_entr.append(entr_T - entr_0)
            entropy_gains.append(np.mean(delta_entr))

            entropy_gains_total += delta_entr

            original_graphs_multiple_seeds.append(original_graphs)
            rewired_graphs_multiple_seeds.append(rewired_graphs)

    print(f"Entropy gain: {np.mean(entropy_gains_total)} ± {ci(entropy_gains_total)}")
    return entropy_gains, original_graphs_multiple_seeds, rewired_graphs_multiple_seeds, time_measurements


def rewire_and_measure(agent, graphs, metric, random):
    originals, rewired = rewire(agent, graphs, random)
    org_vals = measure(originals, metric)
    rew_vals = measure(rewired, metric)
    return org_vals, rew_vals


def calculate_random_performance(graphs, metric, model, seeds, budget):
    avg_perfs = []

    for seed in seeds:
        this_agent = create_MDP(model, seed, budget, None)
        org, new = rewire_and_measure(this_agent, graphs, metric, random=True)
        avg_perfs.append(new - org)

    print("{} after random rewiring: {:.6f} ± {:.6f}".format(metric.name, np.mean(avg_perfs), ci(avg_perfs)))

    return avg_perfs


def generate_model_list(dir_path):
    model_list = []
    default_hyperparams = True
    for filename in os.listdir(dir_path):
        if filename.endswith('.model'):
            model_list.append(f'{dir_path}/{filename}')
        if filename.endswith('.json'):
            with open(f'{dir_path}/{filename}', 'r') as json_file:
                info = json.load(json_file)
                hyperparams_subst = info["hyperopt_params"]
                default_hyperparams = False

    if default_hyperparams:
        print("Using agent's default hyperparameters!")
        hyperparams_subst = None

    return model_list, hyperparams_subst


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)
