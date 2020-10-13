import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
gra_dir  = os.path.dirname(par_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)
from math23k.src.Graph_Module import Graph_Module


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
        self.hidden_size = hidden_size  # 512
        self.input_size  = input_size   # input_size: 2(数据集中出现的常数)
        self.op_nums     = op_nums      # op_nums:    5(数据集中的运算符个数)

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
                # 说明此时存在sub-tree embedding

                # 3. Right   -Goal Generation
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
        leaf_input = torch.cat((current_node, current_context), dim=2)
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

        # 得到当前operator的token embedding
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
