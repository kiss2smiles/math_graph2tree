import os
import sys
import copy
import math
import torch.nn as nn
import torch.optim

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
gra_dir = os.path.dirname(par_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)

from math23k.src.model_utils import get_all_number_encoder_outputs
from math23k.src.model_utils import generate_tree_input
from math23k.src.masked_cross_entropy import masked_cross_entropy

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        # embedding: torch.Size([1, 512]) = 1 * hidden_size
        self.embedding = embedding
        self.left_flag = left_flag


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score           = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack      = copy_list(node_stack)
        self.left_childs     = copy_list(left_childs)
        self.out             = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        # embedding: batch_size * hidden_size
        self.embedding = embedding
        self.terminalc = terminal


# input_batch(list)       ** 原始文本中单词在词表中的索引
# input_length(list)      ** 原始文本中单词序列的长度
# target_batch(list)      ** 输出公式中的单词在词表中的索引
# target_length(list)     ** 输出公式中单词序列的长度

# nums_stack_batch(list)  ** 原始文本中的重复单词在number_values list中的索引
# num_size_batch(list)    ** 原始文本中的数字个数
# generate_nums(list)     ** 原始数据集中所包含的常数(1, 3.14)

# num_pos(list)           ** 原始文本中的数字在原始文本中的位置
# batch_graph(numpy)      ** 建立好的Quantity Cell Graph 和 Quantity Comparison Graph
def train_tree(input_batch,       input_length,      target_batch,       target_length,
               nums_stack_batch,  num_size_batch,    generate_nums,
               encoder,           predict,           generate,           merge,
               encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
               output_lang,       num_pos,           batch_graph,        english=False):

    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)  # seq_len = max_len

    # seq_mask: 未被mask的部分填充为0，被mask的部分填充为1
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)
    # seq_mask: [batch_size, seq_len]

    # sequence num mask for attention
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)  # max_num_size = num_size + constant_size
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    # num_mask: [batch_size, num_size + constant_size]

    unk = output_lang.word2index["UNK"]  # 22

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var   = torch.LongTensor(input_batch ).transpose(0, 1)
    target      = torch.LongTensor(target_batch).transpose(0, 1)
    batch_graph = torch.LongTensor(batch_graph)
    # input_var:   [batch_size, seq_len] => [seq_len, batch_size]
    # target:      [batch_size, tgt_len] => [tgt_len, batch_size]
    # batch_graph: [batch_size, 5, seq_len, seq_len]

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    # padding_hidden: [1, hidden_size]

    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var      = input_var.cuda()
        seq_mask       = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask       = num_mask.cuda()
        batch_graph    = batch_graph.cuda()
    # input_var:      [seq_len, batch_size]
    # seq_mask:       [batch_size, seq_len]
    # padding_hidden: [1,      hidden_size]
    # num_mask:       [batch_size, num_size + constant_size]
    # batch_graph:    [batch_size, 5, seq_len, seq_len]

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    # 1. encoder
    # 在RNN Encoder之后接上一个batch_graph，即将图的信息融入网络中
    # input_var(Tensor):    [seq_len, batch_size]
    # input_length(list):   [batch_size]
    # batch_graph(Tensor):  [batch_size, 5, seq_len, seq_len]
    encoder_outputs, problem_output = encoder(input_seqs=input_var,
                                              input_lengths=input_length,
                                              batch_graph=batch_graph)
    # encoder_outputs: [seq_len, batch_size, hidden_size]
    # problem_output:  [         batch_size, hidden_size]

    max_target_length = max(target_length)  # 最大的公式长度
    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)  # 文本中出现的数字个数

    # 取出文本中所有数字对应embedding
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs=encoder_outputs,
                                                              num_pos=num_pos,
                                                              batch_size=batch_size,
                                                              num_size=num_size,
                                                              hidden_size=encoder.hidden_size)
    # all_nums_encoder_outputs: [batch_size, num_size, hidden_size]

    # n_words:   output_lang.n_words   = 23
    # num_start: output_lang.num_start = 5
    num_start = output_lang.num_start

    # node_stacks: TreeNode, 记录节点node中的 goal vector q
    node_stacks = [[TreeNode(embedding=_, left_flag=False)] for _ in problem_output.split(1, dim=0)]  # [1, hidden_size]

    # embedding_stacks: TreeEmbedding, 记录节点node之前的节点的subtree embedding t(list)
    embeddings_stacks = [[] for _ in range(batch_size)]

    # left_childs: 记录节点node中当前节点的subtree embedding t
    left_childs       = [None for _ in range(batch_size)]

    # 先生成根节点，再生成左子树节点，最后生成右子树节点
    for t in range(max_target_length):
        # 2. predict
        # encoder_outputs:          node representation(words)
        # all_nums_encoder_outputs: node representation(numbers)

        # encoder_outputs:          [seq_len,  batch_size, hidden_size]
        # all_nums_encoder_outputs: [batch_size, num_size, hidden_size]
        # padding_hidden:           [1,       hidden_size]
        # seq_mask:                 [batch_size, seq_len]
        # num_mask:                 [batch_size, num_size + constant_size]
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks=node_stacks,
            left_childs=left_childs,
            encoder_outputs=encoder_outputs,
            num_pades=all_nums_encoder_outputs,
            padding_hidden=padding_hidden,
            seq_mask=seq_mask,
            mask_nums=num_mask)
        # num_score:               [batch_size, num_size + constant_size]
        # op:                      [batch_size, operator_size]  # 包含5个操作符
        # current_embeddings:      [batch_size, 1, hidden_size]
        # current_context:         [batch_size, 1, hidden_size]
        # current_nums_embeddings: [batch_size, num_size + constant_size, hidden_size]

        # outputs: target分类器分数, y^
        outputs = torch.cat((op, num_score), dim=1)
        # outputs: [batch_size, operator_size + num_size + constant_size]

        all_node_outputs.append(outputs)

        # num_start: 5  = num index start
        # unk:       22 = UNK word index

        # 预测出每一个target的值(debug)
        target_t, generate_input = generate_tree_input(target=target[t].tolist(),
                                                       decoder_output=outputs,
                                                       nums_stack_batch=nums_stack_batch,
                                                       num_start=num_start,
                                                       unk=unk)
        # target_t:       target token index = The token with the highest probability
        # generate_input: target token index = The token with the highest probability
        # target_t:       [batch_size] = ground_truth
        # generate_input: [batch_size]

        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()

        # 3. generate (decompose goal vector q to q_l and q_r)
        # current_embeddings: [batch_size, 1, hidden_size]
        # generate_input:     [batch_size]
        # current_context:    [batch_size, 1, hidden_size]
        left_child, right_child, node_label = generate(node_embedding=current_embeddings,
                                                       node_label=generate_input,
                                                       current_context=current_context)
        # left_child:  h_l    = [batch_size,    hidden_size]
        # right_child: h_r    = [batch_size,    hidden_size]
        # node_label:  e(y|P) = [batch_size, embedding_size]

        left_childs = []
        # 在每个batch中依次生成left_node, operator_node, right_node
        for idx, l, r, node_stack, i, o in zip(range(batch_size),
                                               left_child.split(1),
                                               right_child.split(1),
                                               node_stacks,
                                               target[t].tolist(),
                                               embeddings_stacks):

            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            # 生成的node为操作符
            if i < num_start:
                # 生成新的右孩子节点的h_r
                # r.embedding: [1, hidden_size]
                node_stack.append(TreeNode(embedding=r, left_flag=False))

                # 生成新的左孩子节点的h_l
                # l.embedding: [1, hidden_size]
                node_stack.append(TreeNode(embedding=l, left_flag=True))  # left_flag标志此时的node为左节点

                # 更新非叶子节点的Tree embedding, 初始时为token embedding t
                o.append(
                    TreeEmbedding(embedding=node_label[idx].unsqueeze(0), terminal=False))  # terminal=False: 非叶子节点
                # sub_tree embedding (operator) = current_num: [1, embedding_size]

            # 生成的node为操作数
            else:
                # update sub tree embedding = t
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                # current_num: [1, hidden_size]

                # 通过左孩子节点和父节点的sub-tree embedding来更新右孩子节点的sub-tree embedding
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()  # sub_stree.terminal = True  (左孩子节点)
                    op        = o.pop()  # op.terminal        = False (父节点(为操作数))

                    # 更新叶子节点的Tree embedding
                    #   如果此时为右孩子节点，则通过左孩子节点的 subtree embedding t_l 和 右孩子节点的subtree embedding t_r
                    #   来更新根节点的subtree embedding t
                    # op.embedding:        [1, embedding_size] = parent node token embedding = e(y^|P)
                    # sub_stree.embedding: [1,    hidden_size] = left_sub_tree_embedding     = t_l
                    # current_num:         [1,    hidden_size] = right_sub_tree_embedding    = t_r

                    current_num = merge(node_embedding=op.embedding,
                                        sub_tree_1=sub_stree.embedding,
                                        sub_tree_2=current_num)
                    # current_num: [1, hidden_size]

                o.append(TreeEmbedding(embedding=current_num, terminal=True))  # terminal=True: 为叶子节点
                # sub_tree embedding (number) = current_num: [1, hidden_size]

            # 更新left_childs: left_childs记录所有的subtree embedding，并且在生成新节点时更新
            if len(o) > 0 and o[-1].terminal:  # 此时为叶子节点
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)  # 此时为非叶子节点

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    # all_node_outputs: [batch_size, tgt_len, operator_size + num_size + constant_size]

    target = target.transpose(0, 1).contiguous()
    # target: [batch_size, tgt_len]

    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # 计算loss时，在公式中的每一个位置训练一个分类器
    loss = masked_cross_entropy(logits=all_node_outputs,
                                target=target,
                                length=target_length)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


# input_batch       ** 原始文本中单词在词表中的索引
# input_length      ** 原始文本中单词序列的长度
# generate_nums     ** 原始数据集中所包含的常数(1, 3.14)

# num_pos_batch     ** 原始文本中的数字在原始文本中的位置
# batch_graph       ** 建立好的Quantity Cell Graph 和 Quantity Comparison Graph
def evaluate_tree(input_batch, input_length,  generate_nums,
                  encoder,     predict,       generate,      merge,
                  output_lang, num_pos,       batch_graph,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask    = torch.ByteTensor(1, input_length).fill_(0)
    # seq_mask: [1, seq_len]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

    input_var   = torch.LongTensor(input_batch).unsqueeze(1)
    # input_var: [seq_len, 1]

    batch_graph = torch.LongTensor(batch_graph)
    # batch_graph: [1, 5, seq_len, seq_len]

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)
    # num_mask: [1, num_size + constant_size]

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    # padding_hidden: [1, hidden_size]

    batch_size = 1
    if USE_CUDA:
        input_var      = input_var.cuda()
        seq_mask       = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask       = num_mask.cuda()
        batch_graph    = batch_graph.cuda()
        # input_var:      [seq_len, 1]
        # seq_mask:       [1, seq_len]
        # padding_hidden: [1, hidden_size]
        # num_mask:       [1, num_size + constant_size]
        # batch_graph:    [1, 5, seq_len, seq_len]
    # Run words through encoder

    # input_var:     [seq_len, 1]
    # input_length = [seq_len]
    # batch_graph:   [1, 5, seq_len, seq_len]
    encoder_outputs, problem_output = encoder(input_seqs=input_var,
                                              input_lengths=[input_length],
                                              batch_graph=batch_graph)
    # encoder_outputs: [seq_len, 1, hidden_size]
    # problem_output:  [         1, hidden_size]

    # Prepare input and output variables

    num_size = len(num_pos)  # num_size
    # encoder_outputs: [seq_len, 1, hidden_size]
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs=encoder_outputs,
                                                              num_pos=[num_pos],
                                                              batch_size=batch_size,
                                                              num_size=num_size,
                                                              hidden_size=encoder.hidden_size)
    # all_nums_encoder_outputs: [1, num_size, hidden_size]

    num_start = output_lang.num_start  # 5

    # node_stacks: TreeNode, 记录节点node中的 goal vector q
    #   初始根节点n0为goal vector q0 = problem_output
    node_stacks = [[TreeNode(embedding=_, left_flag=False)] for _ in problem_output.split(1, dim=0)]

    # embedding_stacks: TreeEmbedding, 记录节点node之前的节点的subtree embedding t(list)
    #   如果为操作符(非叶子节点),此时embedding_stacks添加operator的token embedding e(y|P)，并设置terminal=False
    #   如果为操作数(左孩子节点),此时embedding_stacks添加operator的token embedding e(y|P)，并设置terminal=True
    #   如果为操作数(右孩子节点),此时
    #      初始化右孩子节点的subtree embedding t_r 为token embedding e(y|P)
    #      弹出左孩子节点(terminal=True)的subtree embedding t_l和根节点的subtree embedding (parent t)
    #      循环完成merge操作, 得到右孩子节点的最终subtree embedding t_r，并设置terminal=True
    embeddings_stacks = [[]   for _ in range(batch_size)]

    # left_childs: 记录节点node中的subtree embedding t
    #   如果为操作符(非叶子节点),此时left_childs输出为None
    #   如果为操作数(左孩子节点),此时left_childs输出为左孩子节点的subtree embedding t_l
    #   如果为操作数(右孩子节点),此时left_childs输出为右孩子节点的subtree embedding t_r
    left_childs       = [None for _ in range(batch_size)]

    beams = [TreeBeam(score=0.0,
                      node_stack=node_stacks,
                      embedding_stack=embeddings_stacks,
                      left_childs=left_childs,
                      out=[])]

    for t in range(max_length):
        current_beams = []

        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue

            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            # b.node_stack(len):        [1]
            # left_childs(len):         [1]
            # encoder_outputs:          [seq_len,  1, hidden_size]
            # all_nums_encoder_outputs: [1, num_size, hidden_size]
            # padding_hidden:           [1, hidden_size]
            # seq_mask:                 [1, seq_len]
            # num_mask:                 [1, num_size + constant_size]
            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks=b.node_stack,
                left_childs=left_childs,
                encoder_outputs=encoder_outputs,
                num_pades=all_nums_encoder_outputs,
                padding_hidden=padding_hidden,
                seq_mask=seq_mask,
                mask_nums=num_mask)
            # num_score:               [1, num_size + constant_size]
            # op / op_score:           [1, operator_size]

            # GOAL VECTOR q
            # current_embeddings:      [1, 1, hidden_size]

            # CONTEXT VECTOR c
            # current_context:         [1, 1, hidden_size]

            # CURRENT NUMBER EMBEDDING MATRIX M_{num}
            # current_nums_embeddings: [1, num_size + constant_size, hidden_size]

            # out_score: [1, operator_size + num_size + constant_size]
            out_score = torch.cat((op, num_score), dim=1)
            out_score = nn.functional.log_softmax(out_score, dim=1)

            topv, topi = out_score.topk(beam_size)
            # topv, topi: values, indexes
            # topv = values:  [1, beam_size]
            # topi = indexes: [1, beam_size]

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                # 预测的结果为运算符
                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()

                    # current_embeddings: goal vector q
                    # generate_input:     token idxs => token embedding e(y|P)
                    # current_context:    context vector c

                    # current_embeddings: [1, 1, hidden_size]
                    # generate_input:     [1]
                    # current_context:    [1, 1, hidden_size]
                    left_child, right_child, node_label = generate(node_embedding=current_embeddings,
                                                                   node_label=generate_input,
                                                                   current_context=current_context)
                    # ** left_child:  当前node的left  child的h_l
                    # ** right_child: 当前node的right child的h_r
                    # ** node_label:  当前node的token embedding e(y|P)
                    # left_child  = h_l:    [1, hidden_size]
                    # right_child = h_r:    [1, hidden_size]
                    # node_label  = e(y|P): [1, embedding_size]

                    # 生成新的右孩子节点的h_l, r.embedding: [1, hidden_size]
                    current_node_stack[0].append(
                        TreeNode(embedding=right_child, left_flag=False))

                    # 生成新的左孩子节点的h_r, r.embedding: [1, hidden_size]
                    current_node_stack[0].append(
                        TreeNode(embedding=left_child, left_flag=True))

                    # 更新非叶子节点的SubTree embedding = 当前节点的token embedding e(y|P)
                    current_embeddings_stacks[0].append(
                        TreeEmbedding(embedding=node_label[0].unsqueeze(0), terminal=False))  # terminal=False: 非叶子节点
                    # sub_tree embedding(operator) = node_label: [1, embedding_size]

                # 预测的结果为运算数
                else:
                    # update sub tree embedding = t
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()  # 左孩子节点
                        op        = current_embeddings_stacks[0].pop()  # 右孩子节点

                        # 更新叶子节点的Tree embedding
                        # 如果此时为右孩子节点，则通过左孩子节点和右孩子节点的subtree embedding来更新根节点的subtree embedding

                        # op.embedding:          [1, embedding_size] = parent node token embedding = e(y^|P)
                        # sub_stree.embedding:   [1,    hidden_size] = left_sub_tree_embedding     = t_l
                        # current_num.embedding: [1,    hidden_size] = right_sub_tree_embedding    = t_r
                        current_num = merge(node_embedding=op.embedding,
                                            sub_tree_1=sub_stree.embedding,
                                            sub_tree_2=current_num)
                        # sub_tree embedding(number) = current_num: [1, embedding_size]

                    current_embeddings_stacks[0].append(
                        TreeEmbedding(embedding=current_num, terminal=True))  # terminal=True: 叶子节点

                # 更新left_childs: left_childs记录所有的subtree embedding，并且在生成新节点时更新
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)  # 此时为叶子节点
                else:
                    current_left_childs.append(None)  # 此时为非叶子节点

                # node_stacks:      TreeNode,      记录节点node中的 goal vector q
                # embedding_stacks: TreeEmbedding, 记录节点node之前的节点的subtree embedding t(list)
                # left_childs:      left_child:    记录节点node中的subtree embedding t
                current_beams.append(TreeBeam(score=b.score+float(tv),
                                              node_stack=current_node_stack,
                                              embedding_stack=current_embeddings_stacks,
                                              left_childs=current_left_childs,
                                              out=current_out))  # 当前预测出的token_idx

        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:  # 此时还可以继续生成新的节点
                flag = False
        if flag:
            break

    return beams[0].out
