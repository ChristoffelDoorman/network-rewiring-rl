from relnet.evaluation.file_paths import FilePaths
from relnet.state.network_generators import NetworkGenerator


def get_gen_params(nodes):
    """ Set number of nodes and edges """
    gp = {}
    gp['n'] = nodes  # number of nodes
    gp['m_ba'] = 2  # barabasi: number of edges to attach from a new node to existing nodes
    gp['m_percentage_er'] = 15  # percentage of number of edges
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])  # number of edges
    gp['k_ws'] = 4
    gp['p_ws'] = 0.1
    gp['p_er'] = 0.15
    return gp


def get_options(file_paths, exp_id, hyper_id, seed, n_nodes, obj_fn, graph_type):
    """ Set log options """
    # optional string to put at end of filename
    exp_name = f'{exp_id}_{hyper_id}_{seed}_{obj_fn}_{graph_type}_{n_nodes}n'
    options = {"log_progress": True,
               "experiment_name": exp_name,
               "log_filename": str(file_paths.construct_log_filepath(exp_name)),
               "log_tf_summaries": False,
               "random_seed": seed,
               "validation_check_interval": 10,
               "models_path": file_paths.models_dir,
               "restore_model": False,
               "data_filename": str(file_paths.construct_metrics_csv_filepath(exp_name)),
               "prefix": f'train_{exp_id}_{hyper_id}_{seed}_{obj_fn}_{graph_type}_{n_nodes}n',
               "model_identifier_prefix": f'{exp_id}_{hyper_id}_{seed}_{obj_fn}_{graph_type}_{n_nodes}n'}
    return options


def get_file_paths(exp_id, objFn, graph_type):
    parent_dir = f'/experiment_data/hyperopt/{objFn}_{graph_type}'
    experiment_id = str(exp_id)
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths
