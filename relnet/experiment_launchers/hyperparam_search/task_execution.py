import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append('/relnet')
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WSNetworkGenerator, \
    ERNetworkGenerator, BA1NetworkGenerator, LANNetworkGenerator
from relnet.objective_functions.objective_functions import *
from relnet.experiment_launchers.hyperparam_search.hyperopt_setup import get_gen_params, get_file_paths, get_options


def main(exp_id, obj_fn, graph_type, graph_name, restore_model=False):
    # open json file with task information
    with open(os.path.join(script_dir, f'tasks/in_files/{obj_fn}_{graph_type}_{exp_id}.json'), 'r') as f:
        config = json.load(f)

    # read task information
    task_id = config['task_id']
    param_keys = config['param_keys']
    ss_chunk = config['ss_chunk']
    ms_chunk = config['ms_chunk']

    # create list for result dumps
    config['results'] = []

    for hyperparams_id, combination in ss_chunk:
        hyperparams = {}

        # make hyperparams dictionary hyperparams={'param_key': param_value}
        for idx, param_value in enumerate(tuple(combination)):
            param_key = param_keys[idx]
            hyperparams[param_key] = param_value

        for model_seed in ms_chunk:
            nodes = 30
            num_training_steps = 120000
            num_train_graphs = 600
            num_validation_graphs = 200
            num_test_graphs = 200
            edge_percentage = 15

            # set experiment parameters
            objF = MERW() if obj_fn == 'MERW' else GlobalEntropy()

            metrics = {}  # don't record any metrics other than objective function
            gen_params = get_gen_params(nodes)
            gen_params['draw_graphs'] = False

            if graph_type == 'ba':
                gen = BANetworkGenerator(
                    **{'store_graphs': True, 'graph_storage_root': Path('/experiment_data/stored_graphs')})
            elif graph_type == 'ws':
                gen = WSNetworkGenerator(
                    **{'store_graphs': True, 'graph_storage_root': Path('/experiment_data/stored_graphs')})
            elif graph_type == 'er':
                gen = ERNetworkGenerator(
                    **{'store_graphs': True, 'graph_storage_root': Path('/experiment_data/stored_graphs')})
            elif graph_type == 'ba1':
                gen = BA1NetworkGenerator(
                    **{'store_graphs': True, 'graph_storage_root': Path('/experiment_data/stored_graphs')})

            # dir_name = f'{exp_id}_{hyperparams_id}_{model_seed}'
            dir_name = f'results'
            file_paths = get_file_paths(dir_name, objF.name, gen.name)

            # construct network seeds for consistent results
            train_graph_seeds, validation_graph_seeds, test_graph_seeds = NetworkGenerator.construct_network_seeds(
                num_train_graphs, num_validation_graphs, num_test_graphs)

            # set log options
            options = get_options(file_paths, exp_id, hyperparams_id, model_seed, nodes, objF.name, gen.name)

            # generate len(train_graph_seeds) graphs
            train_graphs = gen.generate_many(gen_params, train_graph_seeds)
            validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
            test_graphs = gen.generate_many(gen_params, test_graph_seeds)

            # create environment
            obj_fun_kwargs = {"random_seed": model_seed}
            targ_env = GraphEdgeEnv(objF, obj_fun_kwargs, edge_percentage, metrics, gen.name)

            # create agent
            agent = RNetDQNAgent(targ_env)

            # restore from last step
            if restore_model:
                options['restore_model'] = True

            # set hyper parameters for this sweep
            default_hyperparams = agent.get_default_hyperparameters()
            hyperparams_sweep = default_hyperparams.copy()
            hyperparams_sweep.update(hyperparams)
            agent.setup(options, hyperparams_sweep)

            # train agent
            print(f"yahoo, running with seed {model_seed} and hyperparams {hyperparams}")
            agent.train(train_graphs, validation_graphs, num_training_steps)

            # log losses
            best_val_loss = agent.eval(validation_graphs)
            test_loss = agent.eval(test_graphs)
            config['results'].append({'hyperparams_id': hyperparams_id,
                                      'model_seed': model_seed,
                                      'best_val_loss': best_val_loss,
                                      'test_loss': test_loss})

    with open(os.path.join(script_dir, f'tasks/out_files/{exp_id}_{objF.name}_{gen.name}_{nodes}n.json'), 'w') as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", help="experiment id corresponding to exp_id.json file", type=int)
    parser.add_argument("--obj_fn", choices=['Shannon', 'MERW'], help="objective function to train on", type=str)
    parser.add_argument("--graph_type", choices=['ba', 'er', 'ws', 'ba1', 'lan'],
                        help="graph type to run experiments on", type=str)
    parser.add_argument("--graph_name", help="name of LAN graph", type=int)
    parser.add_argument("--restore", action='store_true', help="Restore model from last checkpoint")

    args = parser.parse_args()

    obj_fn = args.obj_fn
    graph_type = args.graph_type

    if args.graph_name is not None:
        graph_name = args.graph_name
    else:
        graph_name = None

    restore = args.restore

    exp_id = args.exp_id - 1

    main(exp_id, obj_fn, graph_type, graph_name, restore)
