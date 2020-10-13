import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import sys
cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
gra_dir  = os.path.dirname(par_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)
from math23k.src.GCN import GCN


# Graph Module
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff ,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Graph_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        # indim:  hidden_size(512)
        # hiddim: hidden_size(512)
        # outdim: hidden_size(512)
        super(Graph_Module, self).__init__()
        '''
        ## Variables:
        - indim: dimensionality of input node features
        - hiddim: dimensionality of the joint hidden embedding
        - outdim: dimensionality of the output node features
        - combined_feature_dim: dimensionality of the joint hidden embedding for graph
        - K: number of graph nodes/objects on the image
        '''
        self.in_dim = indim  # 512
        self.h = 4  # multiGCN: head = 4
        self.d_k = outdim // self.h  # 128

        # 4层GCN网络
        self.graph = clones(module=GCN(in_feat_dim=indim,      # 512
                                       nhid=hiddim,            # 512
                                       out_feat_dim=self.d_k,  # 128
                                       dropout=dropout),
                            N=4)

        self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
        self.norm = LayerNorm(outdim)

    def get_adj(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''

        # graph_nodes: [batch_size, seq_len, hidden_size]
        self.K = graph_nodes.size(1)  # seq_len
        graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)
        # graph_nodes: [(batch_size*seq_len), hidden_size]

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)

        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))

        adjacency_matrix = self.b_normal(adjacency_matrix)

        return adjacency_matrix

    def normalize(self, A, symmetric=True):
        '''
        ## Inputs:
        - adjacency matrix (K, K) : A
        ## Returns:
        - adjacency matrix (K, K) 
        '''
        A = A + torch.eye(A.size(0)).cuda().float()
        d = A.sum(1)
        if symmetric:
            # D = D^{-1/2}
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else :
            D = torch.diag(torch.pow(d, -1))
            return D.mm(A)

    def b_normal(self, adj):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i])
        return adj

    def forward(self, graph_nodes, graph):
        """
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - graph_encode_features (batch_size, K, out_feat_dim)
        """
        # graph_nodes: [seq_len, batch_size, hidden_size]
        # graph:       [batch_size, 5, seq_len, seq_len]
        nbatches = graph_nodes.size(0)
        mbatches = graph.size(0)
        if nbatches != mbatches:
            graph_nodes = graph_nodes.transpose(0, 1)
        # graph_nodes: [batch_size, seq_len, hidden_size]

        # adj (batch_size, K, K): adjacency matrix

        # graph.numel(): 返回数组中的元素个数
        if not bool(graph.numel()):
            adj = self.get_adj(graph_nodes)
            adj_list = [adj, adj, adj, adj]
        else:
            adj = graph.float()
            # adj: [batch_size, 5, seq_len, seq_len]
            # adj[:, 1, :]: Quantity Comparison Graph
            # adj[:, 4, :]: Quantity Cell Graph
            adj_list = [adj[:, 1, :], adj[:, 1, :], adj[:, 4, :], adj[:, 4, :]]

        g_feature = tuple([l(graph_nodes, x) for l, x in zip(self.graph, adj_list)])
        g_feature = self.norm(torch.cat(g_feature, dim=2)) + graph_nodes  # Norm & Add => Z => Z^
        # g_feature: [batch_size, seq_len, hidden_size]

        graph_encode_features = self.feed_foward(g_feature) + g_feature   # Norm & Add => Z-
        # graph_encode_features: [batch_size, seq_len, hidden_size]

        # adj: [batch_size, 5, seq_len, seq_len]
        return adj, graph_encode_features
