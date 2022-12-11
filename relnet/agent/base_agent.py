import numpy as np
import random
import torch
from abc import ABC, abstractmethod
from copy import deepcopy

from relnet.evaluation.eval_utils import eval_on_dataset, get_values_for_g_list, add_list_to_csv
from relnet.state.graph_state import S2VGraph
from relnet.utils.config_utils import get_logger_instance, setup_metrics_to_csv


class Agent(ABC):
    def __init__(self, environment):
        self.environment = environment

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        pass

    def eval(self, g_list,
             initial_obj_values=None,
             validation=False,
             make_action_kwargs=None):
        eval_nets = [deepcopy(g) for g in g_list]
        initial_obj_values, final_obj_values = get_values_for_g_list(self, eval_nets, initial_obj_values, validation,
                                                                     make_action_kwargs)
        return eval_on_dataset(initial_obj_values, final_obj_values)

    @abstractmethod
    def make_actions(self, t, **kwargs):
        pass

    def setup(self, options, hyperparams):
        self.options = options
        if 'log_filename' in options:
            self.log_filename = options['log_filename']
        if 'log_progress' in options:
            self.log_progress = options['log_progress']
        else:
            self.log_progress = False
        if self.log_progress:
            self.logger = get_logger_instance(self.log_filename)
            self.environment.pass_logger_instance(self.logger)
        else:
            self.logger = None

        if 'experiment_name' in options:
            self.experiment_name = options['experiment_name']

        # setup outfile
        if 'data_filename' in options:
            self.data_filename = options['data_filename']
            headers = ['step', 'val_loss']
            if self.environment.metrics:
                for gm in self.environment.metrics.keys():
                    if gm == 'local_metrics':
                        for lm in self.environment.metrics['local_metrics']['kwargs']['metrics']:
                            headers.append(lm)
                    else:
                        headers.append(gm)
            else:
                headers.append(self.environment.objective_function.name)

            if 'restore_model' not in options or options['restore_model'] == False:
                self.data_filename = setup_metrics_to_csv(self.data_filename, headers)

        if 'random_seed' in options:
            self.set_random_seeds(options['random_seed'])
        else:
            self.set_random_seeds(42)
        self.hyperparams = hyperparams

        if 'prefix' in options:
            self.prefix = f"{self.random_seed}_{self.environment.objective_function.name}_{self.environment.network_name}_{options['prefix']}_"
        else:
            self.prefix = f"{self.random_seed}_{self.environment.objective_function.name}_{self.environment.network_name}_"

    @abstractmethod
    def finalize(self):
        pass

    def pick_random_actions(self, i):
        g = self.environment.g_list[i]
        banned_first_nodes = g.banned_actions

        first_valid_acts = self.environment.get_valid_actions(g, banned_first_nodes)
        if len(first_valid_acts) == 0:
            return -1, -1, -1

        first_node = self.local_random.choice(tuple(first_valid_acts))
        rem_budget = self.environment.get_remaining_budget(i)
        banned_second_nodes = g.get_invalid_edge_ends(first_node, rem_budget)

        second_valid_acts = self.environment.get_valid_actions(g, banned_second_nodes)

        if second_valid_acts is None or len(second_valid_acts) == 0:
            second_valid_acts = g.get_disconnected_nodes(first_node)

        if second_valid_acts is None or len(second_valid_acts) == 0:
            if self.logger is not None:
                self.logger.error(f"caught an illegal state: allowed first actions disagree with second")
                self.logger.error(f"first node valid acts: {first_valid_acts}")
                self.logger.error(f"second node valid acts: {second_valid_acts}")
                self.logger.error(f"the remaining budget: {rem_budget}")
                self.logger.error(f"first_node selection: {first_node}")
                self.logger.error(f"{g.k_nns[first_node]}")
                self.logger.error(f"{g.to_networkx().edges()}")
            return -1, -1, -1

        else:
            second_node = self.local_random.choice(tuple(second_valid_acts))

            g_temp, _ = g.add_edge(first_node, second_node)

            banned_third_nodes = g_temp.get_invalid_removal(first_node, second_node, rem_budget)
            third_valid_acts = self.environment.get_valid_actions(g_temp, banned_third_nodes)

            if third_valid_acts is None or len(third_valid_acts) == 0:
                temp_second_node = first_node
                first_node = second_node
                second_node = temp_second_node
                banned_third_nodes = g_temp.get_invalid_removal(first_node, second_node, rem_budget)
                third_valid_acts = self.environment.get_valid_actions(g_temp, banned_third_nodes)

            if third_valid_acts is None or len(third_valid_acts) == 0:
                if self.logger is not None:
                    self.logger.error(f"caught an illegal state: allowed first actions disagree with third")
                    self.logger.error(f"first node valid acts: {first_valid_acts}")
                    self.logger.error(f"second node valid acts: {second_valid_acts}")
                    self.logger.error(f"third node valid acts: {third_valid_acts}")
                    self.logger.error(f"the remaining budget: {rem_budget}")
                    self.logger.error(f"first_node selection: {first_node}")
                    self.logger.error(f"second_node selection: {second_node}")
                    self.logger.error(f"edge_list: {g_temp.edge_pairs}")
                return -1, -1, -1

            else:
                third_node = self.local_random.choice(tuple(third_valid_acts))

                # g_temp2, _ = g.remove_edge(first_node, third_node)
                #
                # if not g_temp2.is_connected():
                #     banned_third_nodes = g_temp.get_invalid_removal(first_node, second_node, rem_budget)
                #     third_valid_acts = self.environment.get_valid_actions(g_temp, banned_third_nodes)
                #     third_node = self.local_random.choice(tuple(third_valid_acts))

                return first_node, second_node, third_node

    @staticmethod
    def say_hello():
        print(f"Hello World!")

    def set_random_seeds(self, random_seed):
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

    def mid_evaluation(self, step_nr, val_loss):
        """
        :param agent:
        :param metrics:
        :param eval_list:
        :return: metrics
        """

        eval_list = [deepcopy(g) for g in self.validation_g_list]

        metric_values = [step_nr, val_loss]

        self.environment.setup_mid_evaluation(eval_list)

        t = 0
        # take actions until terminal
        while not self.environment.is_terminal():
            list_at = self.make_actions(t)  # e.g. greedy / exploration

            connected = self.environment.execute(list_at)
            t += 1

        connected_eval_list = [gval for i, gval in enumerate(eval_list) if i in connected]

        for m in self.environment.metrics.values():
            final_vals = self.environment.get_metric_values(connected_eval_list, m)

            init_vals = m['init_vals'][connected]

            performance = eval_on_dataset(init_vals, final_vals)

            if isinstance(performance, np.ndarray):
                metric_values += list(performance)
            elif isinstance(performance, float):
                metric_values.append(performance)

        add_list_to_csv(self.data_filename, metric_values)

    def execute_rewiring_strategy(self, graphs, convert_from_networkx=False):

        if convert_from_networkx:
            graph_list = []
            for g in graphs:
                state = S2VGraph(g)
                state.populate_banned_actions()
                graph_list.append(state)

        else:
            graph_list = [deepcopy(g) for g in graphs]

        self.environment.setup_rewiring_execution(graph_list)

        t = 0
        connected_and_done = []
        # take actions until terminal
        while not self.environment.is_terminal():
            list_at = self.make_actions(t)  # e.g. greedy / exploration

            connected = self.environment.execute(list_at)
            connected_and_done += list(connected)
            t += 1

        return graph_list, connected_and_done

    def execute_random_rewiring_strategy(self, graphs, convert_from_networkx=False):

        if convert_from_networkx:
            graph_list = []
            for g in graphs:
                state = S2VGraph(g)
                state.populate_banned_actions()
                graph_list.append(state)

        else:
            graph_list = [deepcopy(g) for g in graphs]
        self.environment.setup_rewiring_execution(graph_list)

        t = 0
        # take actions until terminal
        while not self.environment.is_terminal():
            list_at = self.make_random_actions(t)  # e.g. greedy / exploration

            connected = self.environment.execute(list_at)
            t += 1

        return graph_list, connected
