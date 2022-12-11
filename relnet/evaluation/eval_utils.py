import csv
import numpy as np
from copy import deepcopy
from itertools import product

from relnet.utils.config_utils import local_seed


def generate_search_space(parameter_grid,
                          random_search=False,
                          random_search_num_options=20,
                          random_search_seed=42):
    combinations = list(product(*parameter_grid.values()))
    search_space = {i: combinations[i] for i in range(len(combinations))}

    if random_search:
        if not random_search_num_options > len(search_space):
            reduced_space = {}
            with local_seed(random_search_seed):
                random_indices = np.random.choice(len(search_space), random_search_num_options, replace=False)
                for random_index in random_indices:
                    reduced_space[random_index] = search_space[random_index]
            search_space = reduced_space
    return search_space


def get_values_for_g_list(agent, g_list, initial_obj_values, validation, make_action_kwargs):
    """ Perform actions and register obj_fn vals and rewards until termination """

    # compute current objective function values
    if initial_obj_values is None:
        obj_values = agent.environment.get_objective_function_values(g_list)
    else:
        obj_values = initial_obj_values

    agent.environment.setup(g_list, obj_values, training=False)

    t = 0
    # take actions until terminal
    while not agent.environment.is_terminal():
        action_kwargs = (make_action_kwargs or {})
        list_at = agent.make_actions(t, **action_kwargs)  # e.g. greedy / exploration
        if not validation:
            agent.environment.objective_function_kwargs["random_seed"] += 1

        connected = agent.environment.step(list_at)
        t += 1

    # compute final obj_functions after termination
    final_obj_values = agent.environment.get_final_values()

    if connected.size == 0:
        print("All validation graphs have been disconnected!")
        connected = np.arange(0, obj_values.size)

    return obj_values[connected], final_obj_values[connected]


def eval_on_dataset(initial_objective_function_values,
                    final_objective_function_values):
    return np.mean(final_objective_function_values - initial_objective_function_values, axis=0)


def record_episode_histories(agent, g_list):
    states, actions, rewards, initial_values = [], [], [], []

    nets = [deepcopy(g) for g in g_list]
    initial_values = agent.environment.get_objective_function_values(nets)

    agent.environment.setup(nets, initial_values, training=False)
    t = 0
    while not agent.environment.is_terminal():
        list_st = deepcopy(agent.environment.g_list)
        list_at = agent.make_actions(t, **{})

        states.append(list_st)
        actions.append(list_at)
        rewards.append([0] * len(list_at))

        agent.environment.step(list_at)
        t += 1

    final_states = deepcopy(agent.environment.g_list)
    states.append(final_states)
    final_acts = [None] * len(final_states)
    actions.append(final_acts)

    final_obj_values = agent.environment.get_final_values()
    rewards.append(final_obj_values - initial_values)

    return states, actions, rewards, initial_values


def add_list_to_csv(filename, metric_list):
    """ Write line of data or header to metric data csv """
    with open(filename, 'a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(metric_list)
        f_object.close()
