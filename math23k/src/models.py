import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.attn  = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        # hidden:         [batch_size, 1,                      2*hidden_size]
        # num_embeddings: [batch_size, num_size + constant_size, hidden_size]
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len

        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # hidden: [batch_size, num_size + constant_size, 2*hidden_size]

        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)

        # 1. Top-Down Goal Decomposition
        # hidden:         (q; c)
        # num_embeddings: e(y|P)
        energy_in = torch.cat((hidden, num_embeddings), 2)
        # energy_in: (q; c; e(y|P))
        # energy_in: [batch_size, num_size + constant_size, 3*hidden_size]

        energy_in = energy_in.view(-1, self.input_size + self.hidden_size)
        # energy_in: [batch_size * (num_size + constant_size), 3*hidden_size]

        # 1. Top-Down Goal Decomposition
        # self.attn:  W_{s} (q; c; e(y|P))
        # torch.tanh: tanh( W_{s} (q; c; e(y|P)) )
        # self.score: w_{n}^{T} tanh( W_{s} (q; c; e(y|P)) )

        # score: s(y|q; c; P) = s(y|q; c; e(y|P))
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        # score: [batch_size * (num_size + constant_size), 1]

        score = score.squeeze(1)
        # score: [batch_size * (num_size + constant_size)]

        score = score.view(this_batch_size, -1)  # B x O
        # score: [batch_size, num_size + constant_size]
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.attn  = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        # hidden:          [      1, batch_size, hidden_size]
        # encoder_outputs: [seq_len, batch_size, hidden_size]
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        # hidden: [seq_len, batch_size, hidden_size]
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2)
        # energy_in: [seq_len, batch_size, 2*hidden_size]

        energy_in = energy_in.view(-1, self.input_size + self.hidden_size)
        # energy_in: [seq_len * batch_size, 2*hidden_size]

        score_feature = torch.tanh(self.attn(energy_in))
        # score_feature: [seq_len * batch_size, hidden_size]

        # 1. Top-Down Goal Decomposition
        # attn_energies: score(q; h_s^{p})

        # attn_energies: [seq_len * batch_size, 1]
        attn_energies = self.score(score_feature)  # (S x B) x 1

        # attn_energies: [seq_len * batch_size]
        attn_energies = attn_energies.squeeze(1)

        # attn_energies: [batch_size, seq_len]
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S

        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)

        # attn_energies: [batch_size, seq_len]
        # 1. Top-Down Goal Decomposition
        # attn_energies: a_{s}
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size     = input_size      # input_vocab size
        self.embedding_size = embedding_size  # 128
        self.hidden_size    = hidden_size     # 512
        self.n_layers       = n_layers        # 2
        self.dropout        = dropout         # 0.5

        self.embedding  = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade  = nn.GRU(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=n_layers,
                                dropout=dropout,
                                bidirectional=True)
        self.gcn       = Graph_Module(indim=hidden_size,
                                      hiddim=hidden_size,
                                      outdim=hidden_size)  # 4层GCN网络

    def forward(self,
                input_seqs,
                input_lengths,
                batch_graph,
                hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)

        # input_seqs:    [seq_len, batch_size]
        # input_lengths: [batch_size]
        # batch_graph:   [batch_size, 5, seq_len, seq_len]

        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        # embedded:   [seq_len, batch_size, embedding_size]

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden

        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)
        # pade_outputs: [seq_len,   batch_size, 2*hidden_size]
        # pade_hidden:  [2*n_layer, batch_size,   hidden_size]

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs   = pade_outputs[ :, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        # problem_output: [batch_size, hidden_size]
        # pade_outputs:   [seq_len, batch_size, hidden_size]

        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        # pade_outputs: [batch_size, seq_len, hidden_size]

        pade_outputs = pade_outputs.transpose(0, 1)
        # pade_outputs: [seq_len, batch_size, hidden_size]

        return pade_outputs, problem_output


# 预测运算符和运算数的分类分数
class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size # 512
        self.input_size  = input_size  # input_size: 2(数据集中出现的常数)
        self.op_nums     = op_nums     # op_nums:    5(数据集中的运算符个数)

        # Define layers
        self.dropout = nn.Dropout(dropout)

        # constant embedding
        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l  = nn.Linear(hidden_size,     hidden_size)
        self.concat_r  = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size,     hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn  = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self,
                node_stacks,
                left_childs,
                encoder_outputs,
                num_pades,
                padding_hidden,
                seq_mask,
                mask_nums):

        current_embeddings = []
        for st in node_stacks:  # node_stacks记录节点的goal vector q
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        # ** left_childs:          subtree embedding t
        # ** current_embeddings:   goal vector q
        # len(left_childs):        batch_size
        # len(current_embeddings): batch_size
        # left_childs item:        [1, hidden_size]
        # current_embeddings item: [1, hidden_size]
        # 初始时，current_embeddings为problem_output

        current_node_temp = []  # updated goal vector q
        for l, c in zip(left_childs, current_embeddings):
            if l is None:  # left sub-tree embedding is None, generate left child node
                # 在初始化根节点时，h_l为problem_text

                # 2. Left Sub-Goal Generation
                # 若此时左子树为空，则生成左孩子节点
                c = self.dropout(c)                   # h_l
                g = torch.tanh(   self.concat_l(c))   # Q_{le}
                t = torch.sigmoid(self.concat_lg(c))  # g_l
                current_node_temp.append(g * t)       # q_l

            else:  # left sub-tree embedding is not None, generate right child node

                # 3. Right Sub-Goal Generation
                # 若此时左子树不为空，则生成右孩子节点
                # 当左孩子为叶子节点时，sub-tree embedding为embedding matrix，否则由sub_tree embedding 由merge后的结果得到
                ld = self.dropout(l)                                       # ld = sub-tree left tree embedding
                c  = self.dropout(c)                                       # h_r
                g  = torch.tanh(   self.concat_r( torch.cat((ld, c), 1)))  # Q_{re}
                t  = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))  # g_r
                current_node_temp.append(g * t)                            # q_r
        # len(current_node_temp): batch_size
        # current_node_temp item: [1, hidden_size]

        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node)
        # current_node: goal vector q
        # current_node: [batch_size, 1, hidden_size]

        # current_embeddings: goal vector q
        # encoder_outputs:    final hidden state h_{s}^{p}

        # current_embeddings: [1,       batch_size, hidden_size]
        # encoder_outputs:    [seq_len, batch_size, hidden_size]
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        # 1. Top-Down Goal Decomposition
        # current_attn: [batch_size, 1, seq_len]

        # current_attn:    a_{s}
        # encoder_outputs: final hidden state h_{s}^{p}
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        # 1. Top-Down Goal Decomposition
        # current_context: context vector c
        # current_context: [batch_size, 1, hidden_size]

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size

        # constant_size = input_size
        # self.embedding_weight: [         1, constant_size, hidden_size]
        # embedding_weight:      [batch_size, constant_size, hidden_size]
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N

        # embedding_weight:   [batch_size, constant_size, hidden_size]
        # num_pades:          [batch_size, num_size,      hidden_size]
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N
        # embedding_weight:   [batch_size, num_size + constant_size, hidden_size]

        # 1. Top-Down Goal Decomposition
        # current_node:    goal    vector q
        # current_context: context vector c
        leaf_input = torch.cat((current_node, current_context), 2)
        # leaf_input: concat(q; c)
        # leaf_input: [batch_size, 1, 2*hidden_size]

        leaf_input = leaf_input.squeeze(1)
        # leaf_input: [batch_size, 2*hidden_size]

        leaf_input = self.dropout(leaf_input)
        # leaf_input: [batch_size, 2*hidden_size]

        # max pooling the embedding_weight
        # embedding_weight: [batch_size, num_size + constant_size, hidden_size]

        # embedding_weight_: token embedding e(y|P)
        embedding_weight_ = self.dropout(embedding_weight)
        # embedding_weight_: [batch_size, num_size + constant_size, hidden_size]

        # leaf_input.unsqueeze(1): [batch_size, 1, 2*hidden_size]
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)
        # num_score: [batch_size, num_size + constant_size]

        # op: [batch_size, op_num]
        op = self.ops(leaf_input)
        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size  # 128
        self.hidden_size    = hidden_size     # 512

        self.embeddings  = nn.Embedding(op_nums, embedding_size)
        self.em_dropout  = nn.Dropout(dropout)
        self.generate_l  = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r  = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        # node_embedding:  [batch_size, 1, hidden_size]
        # node_label:      [batch_size]
        # current_context: [batch_size, 1, hidden_size]

        node_label_ = self.embeddings(node_label)
        # node_label_: [batch_size, embedding_size]

        node_label = self.em_dropout(node_label_)
        # node_label:  [batch_size, embedding_size]

        node_embedding  = node_embedding.squeeze(1)
        # node_embedding:  [batch_size, hidden_size]

        current_context = current_context.squeeze(1)
        # current_context: [batch_size, hidden_size]

        node_embedding  = self.em_dropout(node_embedding)
        # node_embedding:  [batch_size, hidden_size]

        current_context = self.em_dropout(current_context)
        # current_context: [batch_size, hidden_size]

        # 2. Left Sub-Goal Generation
        # node_embedding:  parent goal    vector q
        # current_context: parent context vector c
        # node_label:      parent token embedding e(y^|P)

        l_child   = torch.tanh(   self.generate_l( torch.cat((node_embedding, current_context, node_label), 1)))  # o_l
        # l_child:   [batch_size, hidden_size]
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))  # C_l
        # l_child_g: [batch_size, hidden_size]
        l_child   = l_child * l_child_g  # h_l
        # l_child:   [batch_size, hidden_size]

        # 3. Right Sub-Goal Generation
        # node_embedding:  parent goal    vector q
        # current_context: parent context vector c
        # node_label:      parent token embedding e(y^|P)

        r_child   = torch.tanh(   self.generate_r( torch.cat((node_embedding, current_context, node_label), 1)))  # o_r
        # l_child:   [batch_size, hidden_size]
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))  # C_r
        # l_child_g: [batch_size, hidden_size]
        r_child   = r_child * r_child_g  # h_r
        # l_child:   [batch_size, hidden_size]
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge   = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        # node_embedding: operator token embedding e(y^|P) = [1, embedding_size]
        # sub_tree_1:     left  sub-tree embedding t_l     = [1, hidden_size]
        # sub_tree_2:     right sub-tree embedding t_r     = [1, hidden_size]
        sub_tree_1     = self.em_dropout(sub_tree_1)
        sub_tree_2     = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        # 4. Subtree Embedding via Recursive Neural Network
        sub_tree   = torch.tanh(   self.merge(  torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))  # c_t
        # sub_tree:   [1, hidden_size]
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))  # g_t
        # sub_tree_g: [1, hidden_size]
        sub_tree = sub_tree * sub_tree_g  # t
        # sub_tree:   [1, hidden_size]

        # operator sub_tree embedding t
        return sub_tree


# 在Graph2Tree中额外添加的图神经网络
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
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Graph_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
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
        self.d_k = outdim//self.h  # 128

        # 4层GCN网络
        self.graph = clones(GCN(in_feat_dim=indim, nhid=hiddim, out_feat_dim=self.d_k, dropout=dropout), 4)
        
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


class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


# Graph_Conv
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  # input * weight
        output  = torch.matmul(adj, support)        # adj * input * weight
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
