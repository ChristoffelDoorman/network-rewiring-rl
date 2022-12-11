import math
import networkx as nx
import numpy as np
import warnings
import xxhash
from copy import deepcopy
from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
from networkx.algorithms.connectivity import local_edge_connectivity
from networkx.algorithms.flow import build_residual_network

budget_eps = 1e-5


class S2VGraph(object):
    def __init__(self, g, k_nns=None):  # g is an instance of a networkx graph structure
        self.num_nodes = g.number_of_nodes()
        self.node_labels = np.arange(self.num_nodes)
        self.all_nodes_set = set(self.node_labels)

        self.rewire_range = 2

        if k_nns is not None:
            self.k_nns = k_nns
        else:
            self.k_nns = self.find_k_nns(g, self.rewire_range)

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = np.ravel(self.edge_pairs)  # [[x1, y1], [x2, y2], ...] --> [x1,y1,x2,y2,...]

        self.node_degrees = np.array([deg for (node, deg) in sorted(g.degree(), key=lambda deg_pair: deg_pair[0])])
        self.first_node = None
        self.second_node = None
        self.dynamic_edges = None

    def find_k_nns(self, g, k):
        """ Returns a dict of the nearest neigbours, and a dict of neigbhours k-hops away"""
        k_nn = dict()
        return k_nn

    def add_edge(self, first_node, second_node):
        nx_graph = self.to_networkx()  # convert S2V graph to NetworkX graph
        nx_graph.add_edge(first_node, second_node)  # add edge
        s2v_graph = S2VGraph(nx_graph, self.k_nns)  # convert NetworkX graph back to S2V graph
        return s2v_graph, 1

    def add_edge_dynamically(self, first_node, second_node):
        self.dynamic_edges.append((first_node, second_node))
        self.node_degrees[first_node] += 1
        self.node_degrees[second_node] += 1
        return 1

    def remove_edge(self, first_node, second_node):
        nx_graph = self.to_networkx()
        nx_graph.remove_edge(first_node, second_node)
        s2v_graph = S2VGraph(nx_graph, self.k_nns)
        return s2v_graph, 1

    def remove_edge_dynamically(self, first_node, second_node):
        self.dynamic_edges.remove((first_node, second_node))
        self.node_degrees[first_node] -= 1
        self.node_degrees[second_node] -= 1
        return 1

    def populate_banned_actions(self, budget=None):
        """" Find and store invalid actions for first and second nodes """
        if budget is not None:
            if budget < budget_eps:
                self.banned_actions = self.all_nodes_set
                return

        # invalid actions (base_node) for first node
        if self.first_node is None:
            self.banned_actions = self.get_invalid_first_nodes(budget)

        # invalid actions (add_edge) for second node given first node
        elif self.second_node is None:
            self.banned_actions = self.get_invalid_edge_ends(self.first_node, budget)

        # invalid actions (remove_edge) for third node given first and second node
        else:
            self.banned_actions = self.get_invalid_removal(self.first_node, self.second_node, budget)

    def get_invalid_first_nodes(self, budget=None):
        return set([node_id for node_id in self.node_labels if
                    self.node_degrees[node_id] == (self.num_nodes - 1) or self.node_degrees[node_id] == 0])

    def get_invalid_edge_ends(self, query_node, budget=None):
        results = self.get_connected_nodes(query_node)
        results.add(query_node)

        return results

    def get_connected_nodes(self, query_node):
        results = set()
        existing_edges = self.edge_pairs.reshape(-1, 2)
        existing_left = existing_edges[existing_edges[:, 0] == query_node]
        results.update(np.ravel(existing_left[:, 1]))
        existing_right = existing_edges[existing_edges[:, 1] == query_node]
        results.update(np.ravel(existing_right[:, 0]))

        return results

    def get_disconnected_nodes(self, query_node, budget=None):
        connected_nodes = self.get_connected_nodes(query_node)
        disconnected_nodes = self.all_nodes_set - connected_nodes - {query_node}
        return disconnected_nodes

    def get_invalid_removal(self, query_node, new_connected_node, budget=None):
        """
        First find the neighbors of the query_node and ban the rest. Also ban nodes that can become
        isolated (with node degree euqals 1). Then, ban nodes that can disconnect the graph.
        :param query_node: selected base node of rewiring operation
        """

        connected_nodes = self.get_connected_nodes(query_node)
        banned = self.all_nodes_set - connected_nodes
        banned.add(new_connected_node)

        return banned

    def get_invalid_removal_with_disconnect(self, query_node, new_connected_node, budget=None):
        """
        First find the neighbors of the query_node and ban the rest. Also ban nodes that can become
        isolated (with node degree euqals 1). Then, ban nodes that can disconnect the graph.
        :param query_node: selected base node of rewiring operation
        """

        H = build_auxiliary_edge_connectivity(self.to_networkx())
        R = build_residual_network(H, "capacity")
        banned = np.ones(self.num_nodes, dtype=np.int32)
        allowed = [node_id for node_id in self.get_connected_nodes(query_node)
                   if node_id != new_connected_node
                   and local_edge_connectivity(self.to_networkx(), query_node, node_id,
                                               auxiliary=H, residual=R, cutoff=2) != 1]

        banned[allowed] = 0
        result = set(banned.nonzero()[0])
        return result

    def init_dynamic_edges(self):
        self.dynamic_edges = []

    def apply_dynamic_edges(self):
        nx_graph = self.to_networkx()
        for edge in self.dynamic_edges:
            nx_graph.add_edge(edge[0], edge[1])
        return S2VGraph(nx_graph)

    def to_networkx(self):
        """ Convert S2V graph to NetworkX graph """
        edges = self.convert_edges()
        g = nx.Graph()
        g.add_edges_from(edges)
        g.add_nodes_from(self.all_nodes_set)
        return g

    def is_connected(self):
        """ Check is graph is connected """
        return nx.is_connected(self.to_networkx())

    def convert_edges(self):
        """ Convert edge list to NetworkX interpretable edge construction """
        return np.reshape(self.edge_pairs, (self.num_edges, 2))

    def display(self, ax=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw_shell(nx_graph, with_labels=True, ax=ax)

    def display_with_positions(self, node_positions, ax=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw(nx_graph, pos=node_positions, with_labels=True, ax=ax)

    def draw_to_file(self, filename):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size_length = self.num_nodes / 5
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.display(ax=ax)
        fig.savefig(filename)
        plt.close()

    def get_adjacency_matrix(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            adj_matrix = np.asarray(nx.convert_matrix.to_numpy_matrix(nx_graph, nodelist=self.node_labels))

        return adj_matrix

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        gh = get_graph_hash(self, size=32, include_first=True)
        return f"Graph State with hash {gh}"


def get_graph_hash(g, size=32, include_first=False):
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    if include_first:
        if g.first_node is not None:
            hash_instance.update(np.array([g.first_node]))
        else:
            hash_instance.update(np.zeros(g.num_nodes))

    hash_instance.update(g.edge_pairs)
    graph_hash = hash_instance.intdigest()
    return graph_hash
