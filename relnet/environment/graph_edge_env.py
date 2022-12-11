import math
import numpy as np
from copy import deepcopy, copy

from relnet.state.network_generators import NetworkGenerator


class GraphEdgeEnv(object):
    def __init__(self, objective_function, objective_function_kwargs,
                 edge_budget_percentage, metrics_dict=None, network_name=None):

        self.original_objective_function = objective_function
        self.original_objective_function_kwargs = objective_function_kwargs
        self.objective_function = self.original_objective_function
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)

        self.metrics = metrics_dict
        self.network_name = network_name
        self.edge_budget_percentage = edge_budget_percentage  # percentage is of total possible edges
        self.reward_eps = 1e-4

        if objective_function.name == 'MERW':
            self.reward_scale_multiplier = 100
            self.penalty = -10

        elif objective_function.name == 'Shannon':
            self.reward_scale_multiplier = 10
            self.penalty = -10

        elif objective_function.name == 'BZ2':
            self.reward_scale_multiplier = 100
            self.penalty = -20

        elif objective_function.name == 'BDM':
            self.reward_scale_multiplier = 100
            self.penalty = -10

        elif objective_function.name == 'NoObjFn':
            self.reward_scale_multiplier = 0
            self.penalty = 0

    def setup(self, g_list, initial_objective_function_values, training=False):
        self.g_list = g_list
        self.n_steps = 0

        self.edge_budgets = self.compute_edge_budgets(self.g_list, self.edge_budget_percentage)
        self.used_edge_budgets = np.zeros(len(g_list), dtype=np.float)
        self.exhausted_budgets = np.zeros(len(g_list), dtype=np.bool)

        for i in range(len(self.g_list)):
            g = g_list[i]
            g.first_node = None
            g.second_node = None
            g.populate_banned_actions(self.edge_budgets[i])

        self.training = training

        self.objective_function_values = np.zeros((2, len(self.g_list)), dtype=np.float)
        self.objective_function_values[0, :] = initial_objective_function_values
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)
        self.rewards = np.zeros(len(g_list), dtype=np.float)

        if self.training:
            self.objective_function_values[0, :] = np.multiply(self.objective_function_values[0, :],
                                                               self.reward_scale_multiplier)

    def setup_mid_evaluation(self, g_list):
        """ Setup for mid evaluation of all metrics"""
        self.g_list = g_list
        self.n_steps = 0

        self.edge_budgets = self.compute_edge_budgets(self.g_list, self.edge_budget_percentage)
        self.used_edge_budgets = np.zeros(len(g_list), dtype=np.float)
        self.exhausted_budgets = np.zeros(len(g_list), dtype=np.bool)

        for i in range(len(self.g_list)):
            g = g_list[i]
            g.first_node = None
            g.second_node = None
            g.populate_banned_actions(self.edge_budgets[i])

        for m in self.metrics.values():
            # calculate initial values
            m['init_vals'] = self.get_metric_values(g_list, m)

    def setup_rewiring_execution(self, g_list):
        """ Setup for rewiring without measuring"""
        self.g_list = g_list
        self.n_steps = 0

        self.edge_budgets = self.compute_edge_budgets(self.g_list, self.edge_budget_percentage)
        self.used_edge_budgets = np.zeros(len(g_list), dtype=np.float)
        self.exhausted_budgets = np.zeros(len(g_list), dtype=np.bool)

        for i in range(len(self.g_list)):
            g = g_list[i]
            g.first_node = None
            g.second_node = None
            g.populate_banned_actions(self.edge_budgets[i])

    def pass_logger_instance(self, logger):
        self.logger_instance = logger

    def get_final_values(self):
        return self.objective_function_values[-1, :]

    def get_objective_function_value(self, s2v_graph):
        """ Single graph: compute the objective function """
        obj_function_value = self.objective_function.compute(s2v_graph, **self.objective_function_kwargs)
        return obj_function_value

    def get_objective_function_values(self, s2v_graphs):
        """ List of graphs """
        return np.array([self.get_objective_function_value(g) for g in s2v_graphs])

    def get_metric_value(self, s2v_graph, metric):
        """ Single graph: compute metric value """
        metric_values = metric['fn'].compute(s2v_graph, **metric['kwargs'])
        return metric_values

    def get_metric_values(self, s2v_graphs, metric):
        """ List of graphs: compute array of metric value """
        return np.array([self.get_metric_value(g, metric) for g in s2v_graphs])

    def get_remaining_budget(self, i):
        return self.edge_budgets[i] - self.used_edge_budgets[i]

    @staticmethod
    def compute_edge_budgets(g_list, edge_budget_percentage):
        edge_budgets = np.zeros(len(g_list), dtype=np.float)

        for i in range(len(g_list)):
            g = g_list[i]
            m = g.num_edges
            edge_budgets[i] = NetworkGenerator.compute_number_edges(m, edge_budget_percentage)

        return edge_budgets

    @staticmethod
    def get_valid_actions(g, banned_actions):
        all_nodes_set = g.all_nodes_set
        valid_nodes = all_nodes_set - banned_actions
        return valid_nodes

    @staticmethod
    def apply_action(g, action, remaining_budget, copy_state=False):
        if g.first_node is None:
            if copy_state:
                g_ref = g.copy()
            else:
                g_ref = g
            g_ref.first_node = action
            g_ref.populate_banned_actions(remaining_budget)
            return g_ref, remaining_budget

        elif g.second_node is None:
            new_g, _ = g.add_edge(g.first_node, action)
            new_g.first_node = g.first_node
            new_g.second_node = action
            new_g.populate_banned_actions(remaining_budget)

            if new_g.banned_actions == new_g.all_nodes_set:
                new_g.first_node = action
                new_g.second_node = g.first_node
                new_g.populate_banned_actions(remaining_budget)

            return new_g, remaining_budget
        else:
            new_g, edge_cost = g.remove_edge(g.first_node, action)
            updated_budget = remaining_budget - edge_cost
            new_g.populate_banned_actions(updated_budget)
            return new_g, updated_budget

    @staticmethod
    def apply_action_in_place(g, action, remaining_budget):
        if g.first_node is None:
            g.first_node = action
            g.populate_banned_actions(remaining_budget)
            return remaining_budget

        elif g.second_node is None:
            g.second_node = action
            _ = g.add_edge_dynamically(g.first_node, action)
            g.populate_banned_actions(remaining_budget)
            return remaining_budget

        else:
            edge_cost = g.remove_edge_dynamically(g.first_node, action)

            g.first_node = None
            g.second_node = None

            updated_budget = remaining_budget - edge_cost
            g.populate_banned_actions(updated_budget)
            return updated_budget

    def step(self, actions):
        """ If possible, take action. Compute obj_fn value and rewards. """
        connected = []
        # loop through list of graphs
        for i in range(len(self.g_list)):
            if not self.exhausted_budgets[i]:
                if actions[i] == -1:
                    if self.logger_instance is not None:
                        self.logger_instance.warn("budget not exhausted but trying to apply dummy action!")
                        self.logger_instance.error(f"the remaining budget: {self.get_remaining_budget(i)}")
                        g = self.g_list[i]  # graph
                        self.logger_instance.error(f"first_node selection: {g.first_node}")

                remaining_budget = self.get_remaining_budget(i)

                # take action
                self.g_list[i], updated_budget = self.apply_action(self.g_list[i], actions[i], remaining_budget)
                self.used_edge_budgets[i] += (remaining_budget - updated_budget)

                # after both actions (n1 and n2) are done?
                if self.n_steps % 3 == 2:

                    # if all actions/nodes are banned actions, compute obj_fn values and rewards
                    if self.g_list[i].banned_actions == self.g_list[
                        i].all_nodes_set:  # all_nodes_set = set(self.node_labels)
                        self.exhausted_budgets[i] = True

                        # check if graph is still connected, and if not give penalty
                        if not self.g_list[i].is_connected():
                            self.rewards[i] = self.penalty

                        # if graph is connected, give positive reward
                        else:
                            connected.append(i)

                            # only objective function
                            objective_function_value = self.get_objective_function_value(self.g_list[i])

                            if self.training:
                                objective_function_value = objective_function_value * self.reward_scale_multiplier

                            # update last obj_val of graph i
                            self.objective_function_values[-1, i] = objective_function_value

                            # compute reward as difference with initial obj_fn value
                            reward = objective_function_value - self.objective_function_values[0, i]
                            if abs(reward) < self.reward_eps:
                                reward = 0

                            self.rewards[i] = reward

        self.n_steps += 1

        return np.array(connected)

    def execute(self, actions):
        """ Execute rewiring during test time without evaluations """
        # loop through list of graphs

        connected = []
        for i in range(len(self.g_list)):
            if not self.exhausted_budgets[i]:
                if actions[i] == -1:
                    if self.logger_instance is not None:
                        self.logger_instance.warn("budget not exhausted but trying to apply dummy action!")
                        self.logger_instance.error(f"the remaining budget: {self.get_remaining_budget(i)}")
                        g = self.g_list[i]  # graph
                        self.logger_instance.error(f"first_node selection: {g.first_node}")

                remaining_budget = self.get_remaining_budget(i)

                # take action
                self.g_list[i], updated_budget = self.apply_action(self.g_list[i], actions[i], remaining_budget)
                self.used_edge_budgets[i] += (remaining_budget - updated_budget)

                # after both actions (n1 and n2) are done?
                if self.n_steps % 3 == 2:

                    # if all actions/nodes are banned actions, compute obj_fn values and rewards
                    if self.g_list[i].banned_actions == self.g_list[
                        i].all_nodes_set:  # all_nodes_set = set(self.node_labels)
                        self.exhausted_budgets[i] = True

                        # check if graph is still connected, and if not give penalty
                        if self.g_list[i].is_connected():
                            connected.append(i)

        self.n_steps += 1

        return np.array(connected)

    def exploratory_actions(self, agent_exploration_policy):
        act_list_t0, act_list_t1, act_list_t2 = [], [], []
        for i in range(len(self.g_list)):
            first_node, second_node, third_node = agent_exploration_policy(i)

            act_list_t0.append(first_node)
            act_list_t1.append(second_node)
            act_list_t2.append(third_node)

        return act_list_t0, act_list_t1, act_list_t2

    def get_max_graph_size(self):
        max_graph_size = np.max([g.num_nodes for g in self.g_list])
        return max_graph_size

    def is_terminal(self):
        return np.all(self.exhausted_budgets)

    def get_state_ref(self):
        cp_first = [g.first_node for g in self.g_list]
        cp_second = [g.second_node for g in self.g_list]
        b_list = [g.banned_actions for g in self.g_list]
        return zip(self.g_list, cp_first, cp_second, b_list)

    def clone_state(self, indices=None):
        if indices is None:
            cp_first = [g.first_node for g in self.g_list][:]
            cp_second = [g.second_node for g in self.g_list][:]
            b_list = [g.banned_actions for g in self.g_list][:]
            return list(zip(deepcopy(self.g_list), cp_first, cp_second, b_list))
        else:
            cp_g_list = []
            cp_first = []
            cp_second = []
            b_list = []

            for i in indices:
                cp_g_list.append(deepcopy(self.g_list[i]))
                cp_first.append(deepcopy(self.g_list[i].first_node))
                cp_second.append(deepcopy(self.g_list[i].second_node))
                b_list.append(deepcopy(self.g_list[i].banned_actions))

            return list(zip(cp_g_list, cp_first, cp_second, b_list))

    def change_obj_fn(self, new_obj_fn, new_kwargs):
        """ Changes the objective function """
        self.objective_function = new_obj_fn
        self.objective_function_kwargs = deepcopy(new_kwargs)

    def restore_original_obj_fn(self):
        """ Changes the objective function back to original """
        self.objective_function = self.original_objective_function
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)
