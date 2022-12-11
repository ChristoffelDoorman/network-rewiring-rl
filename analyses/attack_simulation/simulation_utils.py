import networkx as nx
import numpy as np
import random
from tqdm import tqdm

from analyses.utils import create_MDP, ci


def create_edge_map(graph, source_node):
    """ Create a local edge map of 2 hops from source node """
    edges = []
    for n1 in graph[source_node]:
        for n2 in graph[n1]:
            edges.append((n1, n2))
    return source_node, edges


def create_subgraph(graph, source_node, option='vein', cutoff=2, draw=False):
    """ Make local map. vein only includes two-hop edges, spider_web creates subgraph given two-hop nodes """

    if option == 'vein':
        _, edge_map = create_edge_map(graph, source_node)
        sub_g = nx.Graph()
        sub_g.add_edges_from(edge_map)
    elif option == 'spider_web':
        nodes = nx.single_source_shortest_path(graph, source_node, cutoff=cutoff)
        sub_g = nx.subgraph(graph, nodes.keys())
    if draw:
        nx.draw(sub_g, with_labels=True)

    return source_node, sub_g


def create_trajectory_map(sub_g, source_n):
    """ Find all trajectories in local map from entry point to every target node """
    trajectory_map = dict()
    # loop through all nodes except entry point
    for target_n in sub_g:
        if target_n == source_n:
            continue
        trajectories = nx.all_simple_paths(sub_g, source_n, target_n)
        trajectory_map[target_n] = [*trajectories]

    return source_n, trajectory_map


def find_uncreachable_nodes(trajectory_map, new_graph):
    """ Given the trajectories in the original map, find which nodes have become unreachable after rewiring """
    new_edges = new_graph.edges()
    unreachable_nodes = []

    # loop through every node in the map
    for target in trajectory_map.items():
        end_node = target[0]
        org_paths = target[1]
        target_reachable = False
        p = 0
        while not target_reachable and p < len(org_paths):
            path = org_paths[p]
            for i in range(len(path) - 1):
                v1, v2 = path[i], path[i + 1]
                # check if original edge still exists
                if tuple(sorted([v1, v2])) in new_edges:
                    # if node is found, break
                    if v2 == end_node:
                        target_reachable = True
                        break
                    else:
                        continue
                else:
                    continue
            # next path
            p += 1
        # add end_node if couldn't be reached
        if not target_reachable:
            unreachable_nodes.append(end_node)

    return unreachable_nodes


def make_maps_and_find_missing_nodes(org_graph, new_graph, source_node, option='spider_web', draw=False):
    """
    Create a map from a source node in an original map, and find missing nodes in new map

    :param org_graph (networkx): original graph
    :param new_graph (networkx): graph after rewiring
    :param source_node (int): entry point from which to create a local map
    :param option (str): map type (vein or spider_web)
    :param draw (bool): draw subgraph or not
    :return unreachable_n (list[int]), sub_g (networkx): list of unreachable nodes after rewiring and local map
    """
    source_node, old_map = create_subgraph(org_graph, source_node, option=option, draw=draw)
    old_map_nodes = list(old_map.nodes())
    new_map = new_graph.subgraph(old_map_nodes)
    intersection_graph = nx.intersection(old_map, new_map)
    components = list(nx.connected_components(intersection_graph))
    unreachable_nodes = []
    if len(components) > 1:
        for target_node in old_map:
            if target_node == source_node:
                continue
            else:
                target_reachable = False
                for node_set in components:
                    if source_node in node_set and target_node in node_set:
                        target_reachable = True
                        break

                if not target_reachable:
                    unreachable_nodes.append(target_node)
    return unreachable_nodes, old_map


def random_walk_search(new_graph, old_map, source_node, target_node):
    """
    Given a local old_map with entry point soure_node, find the target_node
    through a random walks in the new_graph. The cost of the random walk is
    computed as follows: every edge that is not included in the old_map and
    has not been visited before has to be cracked and therefore counts as one
    cost. Edges existing in the old_map or which have been visited (and therefore
    have been cracked) have cost zero.

    Returns the random walk cost of finding the missing target node.
    """
    hacked_links = list(old_map.edges())
    cost = 0
    found = False

    # perform first step
    first_nn = list(new_graph[source_node])
    next_node = random.choice(first_nn)
    prev_node = source_node

    while not found:
        edge = tuple(sorted([prev_node, next_node]))
        # hack edge if not seen before
        if edge not in hacked_links:
            cost += 1
            hacked_links.append(edge)
        if next_node == target_node:
            found = True
            break
        # go to next node
        pos = next_node
        # find new neighbours
        nn = list(new_graph[pos])
        nn.remove(prev_node)
        # if no neighbours, go back to previous node
        if len(nn) == 0:
            next_node = prev_node
        # else, choose random next node
        else:
            next_node = random.choice(nn)
        # update previous node
        prev_node = pos

    return cost


def grade_graph(org_graph, new_graph, find_missing_nodes=False, exhautive_search=False):
    attacker = 0
    defender = 0
    scores = []
    this_graph_all_costs = []
    n_nodes = org_graph.number_of_nodes()
    nodes = np.array(org_graph.nodes)
    if n_nodes > 30 and not exhautive_search:
        sampled_nodes = np.random.choice(nodes, size=30, replace=False)
    else:
        sampled_nodes = nodes

    i = 0
    for source_node in sampled_nodes:
        missing_nodes, old_map = make_maps_and_find_missing_nodes(org_graph, new_graph, source_node,
                                                                  option='spider_web')
        i += 1
        N_missing = len(missing_nodes)
        if N_missing == 0:
            attacker += 1
        else:
            defender += 1
            scores.append(N_missing)
            if find_missing_nodes:
                this_node_all_costs = []
                for missing_n in missing_nodes:
                    cost = random_walk_search(new_graph, old_map, source_node, missing_n)
                    this_node_all_costs.append(cost)
                this_graph_all_costs += this_node_all_costs

    return attacker, defender, scores, this_graph_all_costs
