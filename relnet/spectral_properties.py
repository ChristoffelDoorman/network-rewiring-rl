import networkx as nx
import numpy as np
import scipy as sp


def get_laplacian(A):
    degs = A.sum(axis=1).reshape(-1)
    D = np.diagflat(degs)
    L = D - A
    return L


def compute_fiedler_vector(L, v0=None):
    _, eigvs = sp.sparse.linalg.eigsh(L, k=2, which='SM', return_eigenvectors=True, v0=v0)
    fiedler_vector = eigvs[:, 1]
    return fiedler_vector


def compute_effective_resistance(L):
    eigs = np.sort(np.abs(sp.linalg.eigvalsh(L)))[::-1][:-1]
    # as described in Wang et al. 2014, divide by total number of possible edges
    N = L.shape[0]
    total_possible_edges = (N * (N - 1)) / 2
    return (N * np.sum(1 / eigs)) / total_possible_edges


def get_pseudoinverse(M):
    return np.linalg.pinv(M)
