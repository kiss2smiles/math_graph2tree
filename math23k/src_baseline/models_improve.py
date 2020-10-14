import torch
import torch.nn as nn


# 预测数字的概率
class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.attn  = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        # hidden:         [batch_size, 1, 2*hidden_size]
        # num_embeddings: [batch_size, num_size + constant_size, hidden_size]
        # num_mask:       [batch_size, num_size + constant_size]
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # hidden: [batch_size, num_size + constant_size, hidden_size]

        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        # energy_in: [batch_size, num_size + constant_size, 2*hidden_size]

        # score: s(y|q; c; P)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
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
        # hidden: context vector q
        # encoder_outputs: final hidden state
        # hidden:           [      1, batch_size, hidden_size]
        # encoder_outputs:  [seq_len, batch_size, hidden_size]
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        # hidden: [seq_len, batch_size, hidden_size]
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        # energy_in: [seq_len,  batch_size, 2*hidden_size]
        # energy_in: [seq_len * batch_size, 2*hidden_size]
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        # score_feature: [seq_len * batch_size, 2*hidden_size]

        # attn_energies: score(q, h_{s}^{p})
        attn_energies = self.score(score_feature)  # (S x B) x 1
        # attn_energies: [seq_len * batch_size, 1]

        attn_energies = attn_energies.squeeze(1)
        # attn_energies: [seq_len * batch_size]

        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        # attn_energies: [batch_size, seq_len]

        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S
        # attn_energies: [batch_size, seq_len]

        # attn_energies: [batch_size, 1, seq_len]
        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_layers       = n_layers
        self.dropout        = dropout

        self.embedding = nn.Embedding(num_embeddings=input_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers,
                               dropout=dropout,
                               bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # embedded: [seq_len, batch_size, embedding_size]
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)

        # pade_outputs: [seq_len, batch_size, n_direction * hidden_size]
        # pade_hidden:  [n_layer * n_direction, batch_size, hidden_size]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        # problem_output: [batch_size, hidden_size]
        # pade_output:    [seq_len, batch_size, hidden_size]

        # final hidden state
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs   = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H

        return pade_outputs, problem_output


# Top-down Goal Decomposition
class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size  # 512
        self.input_size  = input_size   # 2 (constant size)
        self.op_nums     = op_nums      # 5 (operator nums)

        # Define layers
        self.dropout = nn.Dropout(dropout)

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
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)  # 初始时为全0向量
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)  # [1, 512]

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:  # left sub-tree embedding is None, generate left child node
                # 在初始化根节点时，h_l为h_{s}^{p}，即Encoder的hidden state输出
                # 2. Left Sub-Goal Generation
                c = self.dropout(c)                   # h_l
                g = torch.tanh(self.concat_l(c))      # Q_{le}
                t = torch.sigmoid(self.concat_lg(c))  # g_l
                current_node_temp.append(g * t)       # q_l
            else:  # left sub-tree embedding is not None, generate right child node
                # 3. Right Sub-Goal Generation
                ld = self.dropout(l)                                      # ld = sub-tree left tree embedding
                c = self.dropout(c)                                       # h_r
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))      # Q_{re}
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))  # g_r
                current_node_temp.append(g * t)                           # q_r

        current_node = torch.stack(current_node_temp)
        # current node: goal vector q
        # current_node: [batch_size, 1, hidden_size]

        current_embeddings = self.dropout(current_node)
        # current_embeddings: goal vector q
        # current_embeddings: [batch_size, 1, hidden_size]

        # current_embeddings: goal vector q
        # encoder_outputs:    final hidden state h_{sp}

        # 1. Top-Down Goal Decomposition
        # current_embeddings: [      1, batch_size, hidden_size]
        # encoder_outputs:    [seq_len, batch_size, hidden_size]
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        # current_attn:  a_{s}
        # current_attn: [batch_size, 1, seq_len]

        # CONTEXT VECTOR c: summarizes relevant information of the problem at hand
        # current_attn:    [batch_size, 1,       seq_len]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        # current_context: context vector c
        # current_context: [batch_size, 1, hidden_size]

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N

        # embedding_weight: CONSTANT EMBEDDING MATRIX: e(y|P) = M_{con}
        # embedding_weight: [batch_size, constant_size, hidden_size]
        # num_pades:        NUMBER EMBEDDING MATRIX:   e(y|P) = M_{num}
        # num_pades:        [batch_size, num_size, hidden_size]
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N
        # embedding_weight: [batch_size, num_size + constant_size, hidden_size]

        # current_node:    root goal    vector q
        # current_context: root context vector c
        leaf_input = torch.cat((current_node, current_context), 2)
        # leaf_input: [batch_size, 1, 2*hidden_size]

        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)
        # leaf_input: [batch_size, 2*hidden_size]

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)

        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        # embedding_weight_:  number embedding matrix e(y|P)
        # embedding_weight_: [batch_size, num_size + constant_size, hidden_size]

        # leaf_input:        [batch_size, 1, 2*hidden_size]
        # embedding_weight_: [batch_size, num_size + constant_size, hidden_size]
        # mask_nums:         [batch_size, num_size + constant_size]
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)
        # num_score:         [batch_size, num_size + constant_size]

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)
        # op: [batch_size, op_num]

        # return p_leaf, num_score, op, current_embeddings, current_attn

        # num_score:         [batch_size, num_size + constant_size]
        # op:                [batch_size, op_num]
        # current_node:      [batch_size, 1, hidden_size]
        # current_context:   [batch_size, 1, hidden_size]
        # embedding_weight:  [batch_size, num_size + constant_size, hidden_size]
        return num_score, op, current_node, current_context, embedding_weight
        # current_node:      goal    vector q
        # current_context:   context vector c
        # embedding_weight:  current number + constant embedding matrix


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size  # 128
        self.hidden_size    = hidden_size     # 512

        # op_nums: 操作符数量
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

        node_label  = self.em_dropout(node_label_)
        # node_label:  [batch_size, embedding_size]

        node_embedding  = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding  = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)
        # node_embedding:  [batch_size, hidden_size]
        # current_context: [batch_size, hidden_size]

        # 2. Left Sub-Goal Generation
        # node_embedding:  parent goal    vector q
        # current_context: parent context vector c
        # node_label:      parent token embedding e(y^|P)
        l_child   = torch.tanh(   self.generate_l( torch.cat((node_embedding, current_context, node_label), 1)))  # C_l
        # l_child:   [batch_size, hidden_size]
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))  # o_l
        # l_child_g: [batch_size, hidden_size]
        l_child   = l_child * l_child_g  # h_l
        # l_child:   [batch_size, hidden_size]

        # 3. Right Sub-Goal Generation
        # node_embedding:  parent goal    vector q
        # current_context: parent context vector c
        # node_label:      parent token embedding e(y^|P)
        r_child   = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))      # C_r
        # r_child:   [batch_size, hidden_size]
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))  # o_r
        # r_child_g: [batch_size, hidden_size]
        r_child   = r_child * r_child_g  # h_r
        # r_child:   [batch_size, hidden_size]

        return l_child, r_child, node_label_
