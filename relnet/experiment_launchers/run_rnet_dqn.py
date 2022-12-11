import sys
from pathlib import Path

sys.path.append('/relnet')

from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import CriticalFractionRandom
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator


def get_gen_params():
    """ Set number of nodes and edges """
    gp = {}
    gp['n'] = 30  # number of nodes
    gp['m_ba'] = 2  # barabasi: number of edges to attach from a new node to existing nodes
    gp['m_percentage_er'] = 20  # percentage of max possible edges ( n*(n-1)/2 )
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])  # number of edges
    return gp


def get_options(file_paths):
    """ Set log options """
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "restore_model": False}
    return options


def get_file_paths():
    parent_dir = '/experiment_data'
    experiment_id = 'development'
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths


if __name__ == '__main__':
    num_training_steps = 5000
    num_train_graphs = 100
    num_validation_graphs = 20
    num_test_graphs = 20

    # set up desired graph properties and file locations
    gen_params = get_gen_params()
    file_paths = get_file_paths()

    options = get_options(file_paths)  # set log options
    storage_root = Path('/experiment_data/stored_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}
    gen = BANetworkGenerator(**kwargs)  # create 'empty' network generator

    # construct network seeds for consistent results
    train_graph_seeds, validation_graph_seeds, test_graph_seeds = NetworkGenerator.construct_network_seeds(
        num_train_graphs, num_validation_graphs, num_test_graphs)

    # generate len(train_graph_seeds) graphs
    train_graphs = gen.generate_many(gen_params, train_graph_seeds)  # return value = list([S2V states])
    validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)  # return value = list([S2V states])
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)  # return value = list([S2V states])

    edge_percentage = 2.5  # BUDGET (percentage of maximal possible edges)
    obj_fun_kwargs = {"random_seed": 42, "num_mc_sims": gen_params['n'] * 2}  # num_mc_sims monte carlo simulations?
    targ_env = GraphEdgeEnv(CriticalFractionRandom(), obj_fun_kwargs, edge_percentage)  # create environment

    agent = RNetDQNAgent(targ_env)
    agent.setup(options, agent.get_default_hyperparameters())  # set options and hyperparameters
    agent.train(train_graphs, validation_graphs, num_training_steps)  #
    avg_perf = agent.eval(test_graphs)
