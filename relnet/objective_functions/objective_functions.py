import networkx as nx
import numpy as np
import pybdm
import random

pybdm.options.set(raise_if_zero=False)
import bz2

from relnet.state.graph_state import get_graph_hash, S2VGraph


def extract_kwargs(kwargs):
    num_mc_sims = 20
    random_seed = 42
    if 'num_mc_sims' in kwargs:
        num_mc_sims = kwargs['num_mc_sims']
    if 'random_seed' in kwargs:
        random_seed = kwargs['random_seed']
    return num_mc_sims, random_seed


class NoObjFn(object):
    name = "NoObjFn"
    upper_limit = 0.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        return 0.


class MERW(object):
    name = "MERW"
    upper_limit = 100.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        # retrieve adjacency matrix
        adjacency_mtx = s2v_graph.get_adjacency_matrix()
        eigv, _ = np.linalg.eigh(adjacency_mtx)
        # largest eigenvalue
        L = eigv.max()
        # return MERW
        return np.log(L)


class GlobalEntropy(object):
    name = "Shannon"
    upper_limit = 100.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        g = s2v_graph.to_networkx()
        deg = nx.degree_histogram(g)
        N = g.number_of_nodes()
        s = 0
        for d in deg:
            if d != 0:
                prob = d / N
                s -= prob * np.log2(prob)
        return s


class BZ2CompressRatio(object):
    name = "BZ2"
    upper_limit = 100.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        g = s2v_graph.to_networkx()
        N = g.number_of_nodes()
        adjacency_mtx = np.asarray(nx.adjacency_matrix(g).todense())
        triu_idx = np.triu_indices(N, 1)
        binary_str = bytes(''.join(map(str, adjacency_mtx[triu_idx])), 'utf-8')
        c = bz2.compress(binary_str)
        c_ratio = len(binary_str) / len(c)
        max_val = c_ratio

        if 'num_shuffles' in kwargs:
            n_shuffles = kwargs['num_shuffles']
            shuffled = np.arange(N)
        else:
            n_shuffles = 0

        for _ in range(n_shuffles):
            np.random.shuffle(shuffled)
            adjacency_mtx = np.asarray(nx.adjacency_matrix(g, shuffled).todense(), dtype=int)
            binary_str = bytes(''.join(map(str, adjacency_mtx[triu_idx])), 'utf-8')
            c = bz2.compress(binary_str)
            c_ratio = len(binary_str) / len(c)

            if c_ratio > max_val:
                max_val = c_ratio

        return max_val


class BDM(object):
    name = "BDM"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        # retrieve adjacency matrix
        g = s2v_graph.to_networkx()
        N = g.number_of_nodes()
        bdm = pybdm.BDM(ndim=2)
        if 'num_shuffles' in kwargs:
            n_shuffles = kwargs['num_shuffles']
        else:
            n_shuffles = 0

        shuffled = np.arange(N)
        adjacency_mtx = np.asarray(nx.adjacency_matrix(g).todense())
        min_val = bdm.nbdm(adjacency_mtx)

        for _ in range(n_shuffles):
            # shuffle adjacency matrix
            np.random.shuffle(shuffled)
            adjacency_mtx = np.asarray(nx.adjacency_matrix(g, shuffled).todense())
            val = bdm.nbdm(adjacency_mtx)

            if val < min_val:
                min_val = val

        return min_val


class LocalMetric(object):
    name = "local_metrics"
    upper_limit = 100.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        g_copy = s2v_graph.copy()
        # convert s2v to networkx
        g = g_copy.to_networkx()
        n_nodes = g.number_of_nodes()
        # adjacency matrix
        A = np.asarray(nx.adjacency_matrix(g).todense())
        # set number of subgraphs
        if 'n_subg' in kwargs:
            n_subg = kwargs['n_subg']
        else:
            n_subg = 5

        if 'subg_frac' in kwargs:
            subg_size = kwargs['subg_frac'] * n_nodes
        else:
            subg_size = n_nodes / 3

        if 'metrics' in kwargs:
            metrics = kwargs['metrics']
        else:
            metrics = ['MERW']

        # calculate merw of subgraphs
        subg_val = np.zeros((n_subg, len(metrics)))

        # tot = 0
        for this_subg in range(n_subg):
            m = 0  # metric index

            # choose random entry node and add to set
            n0 = np.random.choice(list(g))
            map_nodes = {n0}

            # add 1st degree neighbors of entry node
            N1 = g.neighbors(n0)
            for n1 in N1:
                map_nodes.add(n1)

            # fill set with 2nd degree nodes until full
            iter_count = 0
            while len(map_nodes) < subg_size and iter_count < 200:
                n1 = np.random.choice(np.nonzero(A[n0, :])[0])
                N2 = np.copy(A[:, n1])
                N2[n0] = 0
                if np.sum(N2) == 0:
                    continue
                n2 = np.random.choice(np.nonzero(N2)[0])
                map_nodes.add(n2)
                iter_count += 1

            subg = g.subgraph(list(map_nodes))

            # calculate MERW
            if 'local_MERW' in metrics:
                adjacency_mtx = nx.adjacency_matrix(subg).todense()  # retrieve adjacency matrix
                eigv, _ = np.linalg.eigh(adjacency_mtx)  # compute eigenvalues (numpy implementation)
                L = eigv.max()  # largest eigenvalue
                subg_val[this_subg, m] = np.log(L)
                m += 1

            if 'local_Shannon' in metrics:
                deg = nx.degree_histogram(subg)
                N = subg.number_of_nodes()

                s = 0
                for d in deg:
                    if d != 0:
                        prob = d / N
                        s -= prob * np.log2(prob)

                subg_val[this_subg, m] = s

                m += 1

            if 'local_BZ2' in metrics:

                N = subg.number_of_nodes()
                adjacency_mtx = np.asarray(nx.adjacency_matrix(subg).todense())
                triu_idx = np.triu_indices(N, 1)
                binary_str = bytes(''.join(map(str, adjacency_mtx[triu_idx])), 'utf-8')

                c = bz2.compress(binary_str)
                c_ratio = len(binary_str) / len(c)

                max_val = c_ratio

                if 'num_shuffles' in kwargs:
                    n_shuffles = kwargs['num_shuffles']
                    shuffled = np.arange(N)
                else:
                    n_shuffles = 0

                for _ in range(n_shuffles):
                    np.random.shuffle(shuffled)
                    adjacency_mtx = np.asarray(nx.adjacency_matrix(subg, shuffled).todense(), dtype=int)
                    binary_str = bytes(''.join(map(str, adjacency_mtx[triu_idx])), 'utf-8')
                    c = bz2.compress(binary_str)
                    c_ratio = len(binary_str) / len(c)

                    if c_ratio > max_val:
                        max_val = c_ratio

                subg_val[this_subg, m] = max_val

                m += 1

            if 'local_BDM' in metrics:
                adjacency_mtx = np.asarray(nx.adjacency_matrix(subg).todense())

                if adjacency_mtx.size > 9:
                    N = subg.number_of_nodes()
                    bdm = pybdm.BDM(ndim=2)
                    if 'num_shuffles' in kwargs:
                        n_shuffles = kwargs['num_shuffles']
                        shuffled = np.arange(N)
                    else:
                        n_shuffles = 0

                    min_val = bdm.nbdm(adjacency_mtx)

                    for _ in range(n_shuffles):
                        # shuffle adjacency matrix
                        np.random.shuffle(shuffled)
                        adjacency_mtx = np.asarray(nx.adjacency_matrix(subg, shuffled).todense(), dtype=int)
                        val = bdm.nbdm(adjacency_mtx)
                        if val < min_val:
                            min_val = val
                else:
                    min_val = 0.

                subg_val[this_subg, m] = min_val
                m += 1

        return np.mean(subg_val, axis=0)
