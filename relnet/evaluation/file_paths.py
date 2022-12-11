import os
import subprocess
from copy import copy
from pathlib import Path


class FilePaths:
    DATE_FORMAT = "%Y-%m-%d-%H-%M-%S"
    GRAPH_STORAGE_DIR_NAME = 'stored_graphs'
    REAL_WORLD_GRAPHS_DIR_NAME = 'real_world_graphs'

    MODELS_DIR_NAME = 'models'
    CHECKPOINTS_DIR_NAME = 'checkpoints'
    SUMMARIES_DIR_NAME = 'summaries'
    EVAL_HISTORIES_DIR_NAME = 'eval_histories'
    HYPEROPT_RESULTS_DIR_NAME = 'hyperopt_results'

    FIGURES_DIR_NAME = 'figures'
    LOGS_DIR_NAME = 'logs'

    DATA_DIR_NAME = 'csv_data'

    DEFAULT_MODEL_PREFIX = 'default'

    def __init__(self, parent_dir, experiment_id, setup_directories=True):
        self.parent_dir = parent_dir

        self.graph_storage_dir = Path(self.parent_dir) / self.GRAPH_STORAGE_DIR_NAME
        self.graph_storage_dir.mkdir(parents=True, exist_ok=True)

        self.real_world_graphs_dir = Path(self.parent_dir) / self.REAL_WORLD_GRAPHS_DIR_NAME

        self.logs_dir = Path(self.parent_dir) / self.LOGS_DIR_NAME
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = Path(self.parent_dir) / self.DATA_DIR_NAME
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_dir = self.get_dir_for_experiment_id(experiment_id)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.experiment_dir / self.MODELS_DIR_NAME
        self.checkpoints_dir = self.models_dir / self.CHECKPOINTS_DIR_NAME
        self.summaries_dir = self.models_dir / self.SUMMARIES_DIR_NAME
        self.eval_histories_dir = self.models_dir / self.EVAL_HISTORIES_DIR_NAME
        self.hyperopt_results_dir = self.models_dir / self.HYPEROPT_RESULTS_DIR_NAME

        self.figures_dir = self.experiment_dir / self.FIGURES_DIR_NAME

        if setup_directories:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.summaries_dir.mkdir(parents=True, exist_ok=True)
            self.eval_histories_dir.mkdir(parents=True, exist_ok=True)
            self.hyperopt_results_dir.mkdir(parents=True, exist_ok=True)
            self.figures_dir.mkdir(parents=True, exist_ok=True)

            self.set_group_permissions()

    def set_group_permissions(self):
        try:
            for dir in [self.graph_storage_dir, self.logs_dir, self.experiment_dir, self.data_dir]:
                abspath = str(dir.absolute())
                subprocess.run(["chmod", "-R", "g+rwx", abspath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def get_dir_for_experiment_id(self, experiment_id):
        return Path(self.parent_dir) / f'{experiment_id}'

    def __str__(self):
        asdict = self.__dict__
        target = copy(asdict)
        for param_name, corresp_path in asdict.items():
            target[param_name] = str(corresp_path.absolute())

        return str(target)

    def __repr__(self):
        return self.__str__()

    def construct_metrics_csv_filepath(self, exp_name):
        return self.data_dir / self.construct_metrics_csv_filename(exp_name)

    @staticmethod
    def construct_metrics_csv_filename(exp_name):
        return f'{exp_name}.csv'

    def construct_log_filepath(self, title=None):
        return self.logs_dir / self.construct_log_filename(title)

    @staticmethod
    def construct_log_filename(title=None):
        hostname = os.getenv("HOSTNAME", "unknown")
        if title is not None:
            return f'experiments_{title}.log'
        else:
            return f'experiments_{hostname}.log'

    @staticmethod
    def construct_model_identifier_prefix(agent_name, obj_fun_name, network_generator_name, model_seed, hyperparams_id,
                                          graph_id=None):
        model_identifier_prefix = f"{agent_name}-{obj_fun_name}-{network_generator_name}-{(graph_id + '-') if graph_id is not None else ''}" \
                                  f"{model_seed}-{hyperparams_id}"
        return model_identifier_prefix

    @staticmethod
    def construct_history_file_name(model_identifier_prefix):
        return f"{model_identifier_prefix}_history.csv"

    @staticmethod
    def construct_best_validation_file_name(model_identifier_prefix):
        return f"{model_identifier_prefix}_best.csv"
