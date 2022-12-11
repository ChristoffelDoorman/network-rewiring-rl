import ast
import json
import math
import networkx as nx
import numpy as np
import random
from abc import ABC, abstractmethod
from pathlib import Path

from relnet.evaluation.file_paths import FilePaths
from relnet.state.graph_state import S2VGraph
from relnet.utils.config_utils import get_logger_instance


class NetworkGenerator(ABC):
    enforce_connected = True

    def __init__(self, store_graphs=False, graph_storage_root=None, logs_file=None):
        super().__init__()

        # if desired, set locations to store graphs
        self.store_graphs = store_graphs
        if self.store_graphs:
            self.graph_storage_root = graph_storage_root
            self.graph_storage_dir = graph_storage_root / self.name
            self.graph_storage_dir.mkdir(parents=True, exist_ok=True)

        # location for log files
        if logs_file is not None:
            self.logger_instance = get_logger_instance(logs_file)
        else:
            self.logger_instance = None

    def generate(self, gen_params, random_seed):

        # read/write graphs from/to files
        if self.store_graphs:
            filename = self.get_data_filename(gen_params, random_seed)
            filepath = self.graph_storage_dir / filename

            should_create = True
            if filepath.exists():
                try:
                    instance = self.read_graphml_with_ordered_int_labels(filepath)  # read off networkX graph
                    state = self.post_generate_instance(
                        instance)  # convert networkx to S2V and populate invalid actions
                    should_create = False
                except Exception:
                    should_create = True

            if should_create:
                instance = self.generate_instance(gen_params, random_seed)  # generate new networkX graph
                state = self.post_generate_instance(instance)  # convert networkx to S2V and populate invalid actions
                nx.readwrite.write_graphml(instance, filepath.resolve())  # write networkX graph to file

                # draw graph to file
                drawing_filename = self.get_drawing_filename(gen_params, random_seed)
                drawing_path = self.graph_storage_dir / drawing_filename
                if 'draw_graphs' not in gen_params or gen_params['draw_graphs'] == True:
                    state.draw_to_file(drawing_path)

        # do not read/write graphs from/to files
        else:
            instance = self.generate_instance(gen_params, random_seed)  # generate new networkX graph
            state = self.post_generate_instance(instance)  # convert networkx to S2V and populate invalid actions

        return state

    def read_graphml_with_ordered_int_labels(self, filepath):
        """ Read a graph from file and label, then return networkx graph """
        instance = nx.readwrite.read_graphml(filepath.resolve())
        num_nodes = len(instance.nodes)
        relabel_map = {str(i): i for i in range(num_nodes)}
        nx.relabel_nodes(instance, relabel_map, copy=False)

        G = nx.Graph()
        G.add_nodes_from(sorted(instance.nodes(data=True)))
        G.add_edges_from(instance.edges(data=True))

        return G

    def generate_many(self, gen_params, random_seeds):
        return [self.generate(gen_params, random_seed) for random_seed in random_seeds]

    @abstractmethod
    def generate_instance(self, gen_params, random_seed):
        pass

    @abstractmethod
    def post_generate_instance(self, instance):
        pass

    def get_data_filename(self, gen_params, random_seed):
        n = gen_params['LAN'] if 'LAN' in gen_params else gen_params['n']
        filename = f"{n}-{random_seed}.graphml"
        return filename

    def get_drawing_filename(self, gen_params, random_seed):
        n = gen_params['LAN'] if 'LAN' in gen_params else gen_params['n']
        filename = f"{n}-{random_seed}.png"
        return filename

    @staticmethod
    def compute_number_edges(m, edge_percentage):
        total_possible_edges = m
        return int(math.ceil((total_possible_edges * edge_percentage / 100)))

    @staticmethod
    def construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs):
        train_seeds = list(range(0, num_train_graphs))
        validation_seeds = list(range(num_train_graphs, num_train_graphs + num_validation_graphs))
        offset = num_train_graphs + num_validation_graphs
        test_seeds = list(range(offset, offset + num_test_graphs))
        return train_seeds, validation_seeds, test_seeds


class OrdinaryGraphGenerator(NetworkGenerator, ABC):
    def post_generate_instance(self, instance):
        """
        This function converts an :instance: (networkx graph structure with nodes and edges)
        into a :state: (S2V graph) and returns it with invalid first-node actions
        """
        state = S2VGraph(instance)
        state.populate_banned_actions()  # find and store invalid first-node actions
        return state


class ERNetworkGenerator(OrdinaryGraphGenerator):
    name = 'er'
    num_tries = 10000

    def generate_instance(self, gen_params, random_seed):
        number_vertices = gen_params['n']
        edge_prob = gen_params['p_er']

        if not self.enforce_connected:
            random_graph = nx.generators.random_graphs.gnp_random_graph(n=number_vertices, p=edge_prob,
                                                                        seed=random_seed)
            return random_graph
        else:
            for try_num in range(0, self.num_tries):
                random_graph = nx.generators.random_graphs.gnp_random_graph(n=number_vertices, p=edge_prob,
                                                                            seed=(random_seed + (try_num * 1000)))
                if nx.is_connected(random_graph):
                    return random_graph
                else:
                    continue
            raise ValueError("Maximum number of tries exceeded, giving up...")


class LANNetworkGenerator(OrdinaryGraphGenerator):
    name = 'lan'

    def generate_instance(self, gen_params, random_seed):
        print("Can't generate this graph, should be read from file...")
        return 0


class BANetworkGenerator(OrdinaryGraphGenerator):
    name = 'ba'

    def generate_instance(self, gen_params, random_seed):
        n, m = gen_params['n'], gen_params['m_ba']
        ba_graph = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=random_seed)
        return ba_graph


class BA1NetworkGenerator(OrdinaryGraphGenerator):
    name = 'ba1'

    def generate_instance(self, gen_params, random_seed):
        n = gen_params['n']
        m = 1
        ba_graph = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=random_seed)
        return ba_graph


class WSNetworkGenerator(OrdinaryGraphGenerator):
    name = 'ws'

    def generate_instance(self, gen_params, random_seed):
        n, k, p = gen_params['n'], gen_params['k_ws'], gen_params['p_ws']
        ws_graph = nx.generators.connected_watts_strogatz_graph(n=30, k=k, p=p, tries=1000, seed=random_seed)
        return ws_graph


class GNMNetworkGenerator(OrdinaryGraphGenerator):
    name = 'gnm'
    num_tries = 10000

    def generate_instance(self, gen_params, random_seed):
        number_vertices = gen_params['n']
        number_edges = gen_params['m']

        if not self.enforce_connected:
            random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges, seed=random_seed)
            return random_graph
        else:
            for try_num in range(0, self.num_tries):
                random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges,
                                                                            seed=(random_seed + (try_num * 1000)))
                if nx.is_connected(random_graph):
                    return random_graph
                else:
                    continue
            raise ValueError("Maximum number of tries exceeded, giving up...")


class RealWorldNetworkGenerator(OrdinaryGraphGenerator, ABC):
    def __init__(self, store_graphs=False, graph_storage_root=None, logs_file=None, original_dataset_dir=None):
        super().__init__(store_graphs=store_graphs, graph_storage_root=graph_storage_root, logs_file=logs_file)

        if original_dataset_dir is None:
            raise ValueError(f"{original_dataset_dir} cannot be None")
        self.original_dataset_dir = original_dataset_dir

        graph_metadata_file = original_dataset_dir / self.name / 'dataset_metadata.json'
        with open(graph_metadata_file.resolve(), "r", encoding='UTF-8') as fh:
            content = fh.read()
            graph_metadata = ast.literal_eval(content)
            self.num_graphs, \
                self.graph_names, \
                self.graph_props = graph_metadata['num_graphs'], graph_metadata['graph_names'], graph_metadata[
                'graph_props']

    def generate_instance(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)

        filepath = self.original_dataset_dir / self.name / f"{graph_name}.graphml"

        nx_graph = self.read_graphml_with_ordered_int_labels(filepath)
        return nx_graph

    def generate_all_original(self, gen_params):
        return [self.generate(gen_params, random_seed) for random_seed in range(self.num_graphs)]

    def get_num_graphs(self):
        return self.num_graphs

    def get_graph_name(self, random_seed):
        graph_idx = random_seed % self.num_graphs
        graph_name = self.graph_names[graph_idx]
        return graph_name

    def get_data_filename(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filename = f"{random_seed}-{graph_name}.graphml"
        return filename

    def get_drawing_filename(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filename = f"{random_seed}-{graph_name}.png"
        return filename

    @staticmethod
    def get_default_kwargs():
        storage_root = Path('/experiment_data/stored_graphs')
        original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
        kwargs = {'store_graphs': True, 'graph_storage_root': storage_root,
                  'original_dataset_dir': original_dataset_dir}
        return kwargs

    def get_data_filename(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filename = f"{random_seed}-{graph_name}.graphml"
        return filename

    def get_drawing_filename(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filename = f"{random_seed}-{graph_name}.png"
        return filename
