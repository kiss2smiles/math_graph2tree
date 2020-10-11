import os
import sys
import copy
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
        self.score = score
        self.node_stack      = copy_list(node_stack)
        self.embedding_stack = copy_list(embedding_stack)
        self.left_childs     = copy_list(left_childs)
        self.out             = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        # embedding: batch_size * hidden_size
        self.embedding = embedding
        self.terminal = terminal


# without subtree embedding
def topdown_train_tree(input_batch,       input_length,      target_batch,      target_length,
                       nums_stack_batch,  num_size_batch,    generate_nums,
                       encoder,           predict,           generate,
                       encoder_optimizer, predict_optimizer, generate_optimizer,
                       output_lang,       num_pos,           english=False):

    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()

    if USE_CUDA:
        input_var      = input_var.cuda()
        seq_mask       = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask       = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_seqs=input_var,
                                              input_lengths=input_length,
                                              hidden=None)

    # Prepare input and output variables
    node_stacks = [[TreeNode(embedding=_, left_flag=False)] for _ in problem_output.split(1, dim=0)]
    max_target_length = max(target_length)
    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)

    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs=encoder_outputs,
                                                              num_pos=num_pos,
                                                              batch_size=batch_size,
                                                              num_size=num_size,
                                                              hidden_size=encoder.hidden_size)

    num_start = output_lang.num_start
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks=node_stacks,
            left_childs=left_childs,
            encoder_outputs=encoder_outputs,
            num_pades=all_nums_encoder_outputs,
            padding_hidden=padding_hidden,
            seq_mask=seq_mask,
            mask_nums=num_mask)

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        # ?
        target_t, generate_input = generate_tree_input(target=target[t].tolist(),
                                                       decoder_output=outputs,
                                                       nums_stack_batch=nums_stack_batch,
                                                       num_start=num_start,
                                                       unk=unk)
        target[t] = target_t

        if USE_CUDA:
            generate_input = generate_input.cuda()

        left_child, right_child, node_label = generate(node_embedding=current_embeddings,
                                                       node_label=generate_input,
                                                       current_context=current_context)

        for idx, l, r, node_stack, i in zip(range(batch_size),
                                            left_child.split(1),
                                            right_child.split(1),
                                            node_stacks,
                                            target[t].tolist()):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                continue

            if i < num_start:
                node_stack.append(TreeNode(embedding=r, left_flag=False))
                node_stack.append(TreeNode(embedding=l, left_flag=True))

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    loss = masked_cross_entropy(logits=all_node_outputs,
                                target=target,
                                length=target_length)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def topdown_evaluate_tree(input_batch,   input_length, generate_nums,
                          encoder,       predict,      generate,
                          output_lang,   num_pos,      beam_size=5,
                          english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var      = input_var.cuda()
        seq_mask       = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask       = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_seqs=input_var,
                                              input_lengths=[input_length],
                                              hidden=None)

    # Prepare input and output variables
    node_stacks = [[TreeNode(embedding=_, left_flag=False)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs=encoder_outputs,
                                                              num_pos=[num_pos],
                                                              batch_size=batch_size,
                                                              num_size=num_size,
                                                              hidden_size=encoder.hidden_size)
    num_start = output_lang.num_start

    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs     = [None for _ in range(batch_size)]

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

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks=b.node_stack,
                left_childs=left_childs,
                encoder_outputs=encoder_outputs,
                num_pades=all_nums_encoder_outputs,
                padding_hidden=padding_hidden,
                seq_mask=seq_mask,
                mask_nums=num_mask)

            out_score = torch.cat((op, num_score), dim=1)
            out_score = nn.functional.log_softmax(out_score, dim=1)
            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()

                    left_child, right_child, node_label = generate(node_embedding=current_embeddings,
                                                                   node_label=generate_input,
                                                                   current_context=current_context)

                    current_node_stack[0].append(TreeNode(embedding=right_child, left_flag=False))
                    current_node_stack[0].append(TreeNode(embedding=left_child, left_flag=True))

                current_beams.append(
                    TreeBeam(score=b.score+float(tv),
                             node_stack=current_node_stack,
                             embedding_stack=embeddings_stacks,
                             left_childs=left_childs,
                             out=current_out))

        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
