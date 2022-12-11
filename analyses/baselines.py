import argparse
import csv
import networkx as nx
import numpy as np
import os
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append('/relnet')
sys.path.append('/analyses')
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.getcwd())

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in


class ObjFn(object):
    def __init__(self, objfn_name):
        self.name = objfn_name

        if objfn_name == 'Shannon':
            # self.obj_fn = GlobalEntropy()
            self.compute = self.compute_Shannon
        elif objfn_name == 'MERW':
            # self.obj_fn = MERW()
            self.compute = self.compute_MERW

    def compute_Shannon(self, g):
        deg = nx.degree_histogram(g)
        N = g.number_of_nodes()
        s = 0
        for d in deg:
            if d != 0:
                prob = d / N
                s -= prob * np.log2(prob)
        return s

    def compute_MERW(self, g):
        adjacency_mtx = nx.to_numpy_matrix(g)
        eigv, _ = np.linalg.eigh(adjacency_mtx)
        L = eigv.max()
        return np.log(L)


def rewire_graph(graph, base_n, removal_node, target_node):
    graph.remove_edge(base_n, removal_node)
    graph.add_edge(base_n, target_node)
    return graph


def random_rewiring(G, budget, obj_fn, seed, print_info=True):
    remaining_budget = budget
    entr_0 = obj_fn.compute(G)
    random.seed(seed)
    while remaining_budget > 0:
        disconnects_graph = True
        valid_base = [n_d[0] for n_d in G.degree() if n_d[1] != (G.number_of_nodes() - 1) or n_d[1] > 0]
        tries = 0
        while disconnects_graph:
            # choose base node
            base_node = random.choice(valid_base)

            # choose addition node
            valid_targ = [n for n in G if n not in G.neighbors(base_node) and n != base_node]
            if len(valid_targ) == 0:
                break
            targ_node = random.choice(valid_targ)

            # choose removal node
            valid_remv = [n for n in G.neighbors(base_node) if n != targ_node]
            if len(valid_remv) == 0:
                break
            remv_node = random.choice(valid_remv)

            G_temp = G.copy()
            G_temp = rewire_graph(G_temp, base_node, remv_node, targ_node)
            if nx.is_connected(G_temp):
                disconnects_graph = False
                G = G_temp
            else:
                tries += 1
                if tries > 1000:
                    disconnects_graph = False
                    remaining_budget = 0

        remaining_budget -= 1

    entr_T = obj_fn.compute(G)
    rew_acts = budget - remaining_budget

    if print_info:
        print("Rewiring operations:", int(rew_acts))
        print(f'{obj_fn.name} entropy after rewiring: {entr_T}')
        print(f'{obj_fn.name} entropy increase: {entr_T - entr_0}')

    return entr_0, entr_T, rew_acts, G


def exhaustive_rewiring(G, budget, obj_fn, print_info=True):
    remaining_budget = budget
    entr_0 = obj_fn.compute(G)
    while remaining_budget > 0:
        entr = obj_fn.compute(G)
        G_complement = nx.complement(G)
        rew_operations = []
        for e in G.edges:
            for e_complement in G_complement.edges:
                if e[0] in e_complement:
                    bs_n = e[0]
                    rm_n = e[1]
                    if G.degree(rm_n) > 1:
                        bs_idx = e_complement.index(bs_n)
                        tg_n = e_complement[~bs_idx]
                        g_rew = G.copy()
                        g_rew = rewire_graph(g_rew, bs_n, rm_n, tg_n)
                        rew_entr = obj_fn.compute(g_rew)
                        if rew_entr > entr:
                            rew_operations.append((bs_n, rm_n, tg_n, (rew_entr - entr)))

                elif e[1] in e_complement:
                    bs_n = e[1]
                    rm_n = e[0]
                    if G.degree(rm_n) > 1:
                        bs_idx = e_complement.index(bs_n)
                        tg_n = e_complement[~bs_idx]
                        g_rew = G.copy()
                        g_rew = rewire_graph(g_rew, bs_n, rm_n, tg_n)
                        rew_entr = obj_fn.compute(g_rew)
                        if rew_entr > entr:
                            rew_operations.append((bs_n, rm_n, tg_n, (rew_entr - entr)))
                else:
                    continue

        if len(rew_operations) == 0:
            break

        ordered_rewiring = sorted(rew_operations, reverse=True, key=lambda item: item[3])
        disconnects_graph = True
        i = 0
        while disconnects_graph:
            if i == len(ordered_rewiring):
                remaining_budget = 0
                break
            base_node = ordered_rewiring[i][0]
            remv_node = ordered_rewiring[i][1]
            targ_node = ordered_rewiring[i][2]
            G_temp = G.copy()
            G_temp = rewire_graph(G_temp, base_node, remv_node, targ_node)
            if nx.is_connected(G_temp):
                disconnects_graph = False
                G = G_temp
            else:
                i += 1

        remaining_budget -= 1

    entr_T = obj_fn.compute(G)
    rew_acts = budget - remaining_budget
    if print_info:
        print("Rewiring operations:", int(rew_acts))
        print(f'{obj_fn.name} entropy after rewiring: {entr_T}')

    return entr_0, entr_T, rew_acts, G
