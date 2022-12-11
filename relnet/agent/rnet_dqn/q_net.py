import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from relnet.agent.fnapprox.gnn_regressor import GNNRegressor
from torch.autograd import Variable

from relnet.utils.config_utils import get_device_placement

sys.path.append('/usr/lib/pytorch_structure2vec/s2v_lib')
from pytorch_util import weights_init


def jmax(arr, prefix_sum):
    """
    Compute lists of actions and values where each element corresponds to a graph in batch
    :param arr: q-values
    :param prefix_sum: cumulative sum of nodes/graph
    :return: two lists (actions and values) of each graph in batch
    """
    actions = []
    values = []
    # for every graph in batch
    for i in range(len(prefix_sum[0:, ])):
        if i == 0:
            start_index = 0
            end_index = prefix_sum[i]
        else:
            start_index = prefix_sum[i - 1]
            end_index = prefix_sum[i]

        # q-values per graph
        arr_vals = arr[start_index:end_index]

        # action is max q-value for that graph
        act = np.argmax(arr_vals)
        val = arr_vals[act]

        actions.append(act)
        values.append(val)

    actions_tensor = torch.LongTensor(actions)
    values_tensor = torch.Tensor(values)

    return actions_tensor, values_tensor


def greedy_actions(q_values, v_p, banned_list):
    """
    Choose greedy actions from q_values in batched structure
    :param q_values:
    :param v_p: prefix_sum (cumulative list of nodes/graph in batch: [80, 180, 380, 450, ..]
    :param banned_list:
    :return: two lists (actions and values) of each graph in batch
    """
    actions = []
    offset = 0
    banned_acts = []
    prefix_sum = v_p.data.cpu().numpy()
    for i in range(len(prefix_sum)):  # number of graphs in batch
        # nodes in graph i
        n_nodes = prefix_sum[i] - offset

        if banned_list is not None and banned_list[i] is not None:
            for j in banned_list[i]:
                banned_acts.append(offset + j)
        offset = prefix_sum[i]

    q_values = q_values.data.clone()
    q_values.resize_(len(q_values))

    banned = torch.LongTensor(banned_acts)
    device_placement = get_device_placement()
    if device_placement == 'GPU':
        banned = banned.cuda()

    # if there are banned actions
    if len(banned_acts):
        min_tensor = torch.tensor(float(np.finfo(np.float32).min))
        if device_placement == 'GPU':
            min_tensor = min_tensor.cuda()
        q_values.index_fill_(0, banned, min_tensor)

    q_vals_cpu = q_values.data.cpu().numpy()

    # return lists of actions and values where each element corresponds to a graph in batch
    return jmax(q_vals_cpu, prefix_sum)


class QNet(GNNRegressor, nn.Module):
    def __init__(self, hyperparams, s2v_module):
        super().__init__(hyperparams, s2v_module)

        embed_dim = hyperparams['latent_dim']

        self.linear_1 = nn.Linear(embed_dim * 2, hyperparams['hidden'])
        self.linear_out = nn.Linear(hyperparams['hidden'], 1)

        # apply batchnormalization
        self.bn = nn.BatchNorm1d(num_features=hyperparams['hidden'])

        weights_init(self)

        self.num_node_feats = 3
        self.num_edge_feats = 0

        if s2v_module is None:
            self.s2v = self.model(latent_dim=embed_dim,
                                  output_dim=0,
                                  num_node_feats=self.num_node_feats,
                                  num_edge_feats=self.num_edge_feats,
                                  max_lv=hyperparams['max_lv'])
        else:
            self.s2v = s2v_module

    def add_offset(self, actions, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        shifted = []
        for i in range(len(prefix_sum)):
            if i > 0:
                offset = prefix_sum[i - 1]
            else:
                offset = 0
            shifted.append(actions[i] + offset)

        return shifted

    def rep_global_embed(self, graph_embed, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        rep_idx = []
        for i in range(len(prefix_sum)):
            if i == 0:
                n_nodes = prefix_sum[i]
            else:
                n_nodes = prefix_sum[i] - prefix_sum[i - 1]
            rep_idx += [i] * n_nodes

        rep_idx = Variable(torch.LongTensor(rep_idx))
        if get_device_placement() == 'GPU':
            rep_idx = rep_idx.cuda()
        graph_embed = torch.index_select(graph_embed, 0, rep_idx)
        return graph_embed

    def prepare_node_features(self, batch_graph, picked_first_nodes, picked_second_nodes):
        """
        Return information about nodes/graph and picked_nodes/graph for a batch of graphs
        :param batch_graph: batch containing graphs
        :param picked_nodes: list of integers denoting the number of picked nodes
        :return:
        - node_feat is list of [1 0] or [0 1] indicating number of picked nodes of graphs
            in batch, [0 1] indicating the picked nodes
        - prefix_sum is list with cumulative sum of nodes. If every graph in batch has
            120 nodes, it would look like [120, 240, 360, ...]
        """
        n_nodes = 0  # total number of nodes in BATCH ( sum(graph_nodes) )
        prefix_sum = []  # list of cumulative number of nodes of each graph in batch
        picked_first_ones = []  # list of number of picked nodes added to prefix_sum
        picked_second_ones = []  # list of number of picked nodes added to prefix_sum
        for i in range(len(batch_graph)):
            if picked_first_nodes is not None and picked_first_nodes[i] is not None:
                assert picked_first_nodes[i] >= 0 and picked_first_nodes[i] < batch_graph[i].num_nodes
                picked_first_ones.append(n_nodes + picked_first_nodes[i])

            if picked_second_nodes is not None and picked_second_nodes[i] is not None:
                assert picked_second_nodes[i] >= 0 and picked_second_nodes[i] < batch_graph[i].num_nodes
                picked_second_ones.append(n_nodes + picked_second_nodes[i])

            n_nodes += batch_graph[i].num_nodes  # add number of nodes of graph i to n_nodes
            prefix_sum.append(n_nodes)

        # list to indicate how many picked nodes every graph has
        node_feat = torch.zeros(n_nodes, self.num_node_feats)
        node_feat[:, 0] = 1.0

        # [1 0 0] nothing, [0 1 0] base node, [0 0 1] target node
        if len(picked_first_ones):
            node_feat.numpy()[picked_first_ones, 2] = 0.0
            node_feat.numpy()[picked_first_ones, 1] = 1.0
            node_feat.numpy()[picked_first_ones, 0] = 0.0

        if len(picked_second_ones):
            node_feat.numpy()[picked_second_ones, 2] = 1.0
            node_feat.numpy()[picked_second_ones, 1] = 0.0
            node_feat.numpy()[picked_second_ones, 0] = 0.0

        return node_feat, torch.LongTensor(prefix_sum)

    def forward(self, states, actions, greedy_acts=False):
        batch_graph, picked_first_nodes, picked_second_nodes, banned_list = zip(*states)

        # prefix_sum is cumulative sum of nodes/graph
        # node_feat indicated number of picked nodes/graph
        node_feat, prefix_sum = self.prepare_node_features(batch_graph, picked_first_nodes, picked_second_nodes)
        embed, graph_embed, prefix_sum = self.run_s2v_embedding(batch_graph, node_feat, prefix_sum)

        prefix_sum = Variable(prefix_sum)
        if actions is None:  # give me the q-values for all of the possible values
            graph_embed = self.rep_global_embed(graph_embed, prefix_sum)
        else:
            shifted = self.add_offset(actions, prefix_sum)
            embed = embed[shifted, :]

        embed_s_a = torch.cat((embed, graph_embed), dim=1)

        # embed_s_a = F.relu(self.linear_1(embed_s_a))

        # apply batch normalization
        embed_s_a = F.relu(self.bn(self.linear_1(embed_s_a)))

        raw_pred = self.linear_out(embed_s_a)

        if greedy_acts:
            actions, _ = greedy_actions(raw_pred, prefix_sum, banned_list)

        return actions, raw_pred, prefix_sum


class NStepQNet(nn.Module):
    def __init__(self, hyperparams, num_steps):
        super(NStepQNet, self).__init__()

        list_mod = [QNet(hyperparams, None)]

        for i in range(1, num_steps):
            list_mod.append(QNet(hyperparams, list_mod[0].s2v))

        self.list_mod = nn.ModuleList(list_mod)  # list of t QNets
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts=False):
        """ Returns the forward pass of QNet at time t
        :param time_t
        """
        assert time_t >= 0 and time_t < self.num_steps

        return self.list_mod[time_t](states, actions, greedy_acts)
