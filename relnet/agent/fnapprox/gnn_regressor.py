import numpy as np
import torch

from relnet.state.graph_embedding import EmbedMeanField, EmbedLoopyBP
from relnet.utils.config_utils import get_device_placement


class GNNRegressor(object):
    def __init__(self, hyperparams, s2v_module):
        super(GNNRegressor, self).__init__()
        self.hyperparams = hyperparams

        if hyperparams['embedding_method'] == 'mean_field':
            self.model = EmbedMeanField
        elif hyperparams['embedding_method'] == 'loopy_bp':
            self.model = EmbedLoopyBP
        else:
            raise ValueError(f"unknown embedding method {hyperparams['embedding_method']}")


    def run_s2v_embedding(self, batch_graph, node_feat, prefix_sum):
        if get_device_placement() == 'GPU':
            node_feat = node_feat.cuda()
            prefix_sum = prefix_sum.cuda()
        edge_feat = None
        embed, graph_embed = self.s2v(batch_graph, node_feat, edge_feat, pool_global=True)
        return embed, graph_embed, prefix_sum