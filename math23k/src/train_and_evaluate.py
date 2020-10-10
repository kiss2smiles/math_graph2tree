import os
import sys
import math
import time
import torch
import torch.optim
import torch.nn.functional as f

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
gra_dir = os.path.dirname(par_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)

from math23k.src.masked_cross_entropy import *
from math23k.src.pre_data import *
from math23k.src.expressions_transfer import *
from math23k.src.models import *

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max

    # 判断当前target是运算符还是运算数，如果为运算符，则重复step2，否则，跳转执行step3
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]

        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


# 将数字编码添加到文本序列编码的最后
# seq_len =  98
# num_size =  8
# indices =  [
#   18, 33, 46, 86, 0, 0, 0, 0,       = 0+18, 0+33, 0+46, 0+86,
#   103, 107, 114, 118, 141, 0, 0, 0, = 98+5, 98+9, 98+16, 98+20, 98+43
#   210, 216, 230, 234, 237, 0, 0, 0, = 196+14, 196+20, 196+34, 196+38, 196+41,
#   295, 315, 328, 335, 341, 0, 0, 0, = 294+1, 294+21, 294+34, 294+41, 294+47,
#   400, 412, 414, 432, 434, 0, 0, 0, = 392+8, 392+20, 392+22, 392+40, 392+42,
# ...
# ]
# num_pos =  [
#   [18, 33, 46, 86],
#   [5, 9, 16, 20, 43],
#   [14, 20, 34, 38, 41],
#   [1, 21, 34, 41, 47],
#   [8, 20, 22, 40, 42],
# ...
# ]
# masked_index = [
#   [0]*512, [0]*512, [0]*512, [0]*512, [1]*512, [1]*512, [1]*512, [1]*512,
#   [0]*512, [0]*512, [0]*512, [0]*512, [0]*512, [1]*512, [1]*512, [1]*512,
#   [0]*512, [0]*512, [0]*512, [0]*512, [0]*512, [1]*512, [1]*512, [1]*512,
#   [0]*512, [0]*512, [0]*512, [0]*512, [0]*512, [1]*512, [1]*512, [1]*512,
# ]
def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()

    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]

    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]

    # indices:      batch_size * num_size
    # masked_index: batch_size * num_size * hidden_size
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)

    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()

    # encoder_outputs: seq_len * batch_size * hidden_size
    # all_outputs:     batch_size * seq_len * hidden_size
    # all_embedding:   (batch_size * seq_len) * hidden_size
    all_outputs   = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H

    # all_embedding: (batch_size * seq_len) * hidden_size
    # indices:       (batch_size * num_size)
    # all_num:       (batch_size * num_size) * hidden_size
    all_num = all_embedding.index_select(0, indices)

    # all_num:        batch_size * num_size  * hidden_size
    all_num = all_num.view(batch_size, num_size, hidden_size)
    # 填充的位置的embedding初始化为随机的embedding
    return all_num.masked_fill_(masked_index, 0.0)


def train_attn(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
               generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
               use_teacher_forcing=1, beam_size=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2
    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()

        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss


def evaluate_attn(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                  beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0].all_output
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0].all_output


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


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        # embedding: batch_size * hidden_size
        self.embedding = embedding
        self.terminal = terminal


# ** 原始文本的单词在词典中的索引
#   input_batch =  [154, 285, 157, 545, 65,  746, 42,  286, 285,  287,
#                   151, 289, 290, 291, 114, 26,  154, 155, 1561, 1,
#                   59,  103, 157, 155, 42,  287, 423, 289, 26,   157,
#                   155, 289, 1,   59,  103, 156, 135, 150, 1,    61,
#                   13,  154, 155, 58,  59,  114, 1,   61,  26,   157,
#                   155, 58,  59,  114, 1,   61,  13,  605, 286,  285,
#                   287, 151, 995, 150, 34,  61,  52]

# ** 原始文本的单词序列的长度
#   input_length =  67

# ** 输出公式的单词在词典中的索引
#   target_batch =  [2, 2, 0, 2, 7, 8, 10, 0, 8, 11,
#                    9, 0, 0, 0, 0, 0, 0]

# ** 输出公式的单词序列的长度
#   target_length =  11

# ** num_stack
#   num_stack_batch =  []

# ** 文本中的数字个数
#   num_size_batche =  5

# ** 数据集中的常数在output_vocab的索引
#   generate_nums =  [5, 6]

# ** 文本中的数字在原始文本中的位置
#   num_pos =  [19, 32, 38, 46, 54]

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
    # input_var:    [seq_len, batch_size]
    # input_length: [batch_size]
    # batch_graph:  [batch_size, 5, seq_len, seq_len]
    encoder_outputs, problem_output = encoder(input_seqs=input_var,
                                              input_lengths=input_length,
                                              batch_graph=batch_graph)
    # ** encoder_outputs: final_hidden_states h_{s}^{p}
    # ** problem_output:  root node n0 goal vector q0 = h_{n}^{p} + h_{0}^{p}
    # encoder_outputs: [seq_len, batch_size, hidden_size]
    # problem_output:  [batch_size, hidden_size]

    max_target_length = max(target_length)  # 最大的公式长度
    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)  # 文本中出现的数字个数

    # 取出文本中所有数字对应embedding
    # number   embedding 取自encoder_outputs
    # constant embedding 取自nn.Embedding
    # operator embedding 取自nn.Embedding
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
    #   初始根节点n0为goal vector q0 = problem_output
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # [1, hidden_size]

    # embedding_stacks: TreeEmbedding, 记录节点node之前的节点的subtree embedding t(list)
    #   如果为操作符(非叶子节点),此时embedding_stacks添加operator的token embedding e(y|P)，并设置terminal=False
    #   如果为操作数(左孩子节点),此时embedding_stacks添加operator的token embedding e(y|P)，并设置terminal=True
    #   如果为操作数(右孩子节点),此时
    #      初始化右孩子节点的subtree embedding t_r 为token embedding e(y|P)
    #      弹出左孩子节点(terminal=True)的subtree embedding t_l和根节点的subtree embedding (parent t)
    #      循环完成merge操作, 得到右孩子节点的最终subtree embedding t_r，并设置terminal=True
    embeddings_stacks = [[] for _ in range(batch_size)]

    # left_childs: 记录节点node中的subtree embedding t
    #   如果为操作符(非叶子节点),此时left_childs输出为None
    #   如果为操作数(左孩子节点),此时left_childs输出为左孩子节点的subtree embedding t_l
    #   如果为操作数(右孩子节点),此时left_childs输出为右孩子节点的subtree embedding t_r
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

        # GOAL VECTOR q
        # current_embeddings:      [batch_size, 1, hidden_size]

        # CONTEXT VECTOR c
        # current_context:         [batch_size, 1, hidden_size]

        # CURRENT NUMBER EMBEDDING MATRIX M_{num}
        # current_nums_embeddings: [batch_size, num_size + constant_size, hidden_size]

        # outputs: target分类器分数, y^
        outputs = torch.cat((op, num_score), dim=1)
        # outputs: [batch_size, operator_size + num_size + constant_size]

        all_node_outputs.append(outputs)

        # num_start: 5  = num index start
        # unk:       22 = UNK word index

        # 预测出每一个target的值(?)
        target_t, generate_input = generate_tree_input(target=target[t].tolist(),
                                                       decoder_output=outputs,
                                                       nums_stack_batch=nums_stack_batch,
                                                       num_start=num_start,
                                                       unk=unk)
        # target_t:       target token index = The token with the highest probability
        # generate_input: target token index = The token with the highest probability (将数字的embedding初始化为op_emb)
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
        # ** left_child:  当前node的left  child的h_l
        # ** right_child: 当前node的right child的h_r
        # ** node_label:  当前node的token embedding e(y|P)
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
                node_stack.append(TreeNode(r))

                # 生成新的左孩子节点的h_l
                # l.embedding: [1, hidden_size]
                node_stack.append(TreeNode(l, left_flag=True))  # left_flag标志此时的node为左节点

                # 更新非叶子节点的Tree embedding, 初始时为token embedding t
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))  # terminal=False: 非叶子节点
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
                    # op.embedding:        parent node token embedding = e(y^|P)
                    # sub_stree.embedding: left_sub_tree_embedding     = t_l
                    # current_num:         right_sub_tree_embedding    = t_r

                    # op.embedding:          [1, embedding_size]
                    # sub_stree.embedding:   [1,    hidden_size]
                    # current_num.embedding: [1,    hidden_size]
                    current_num = merge(node_embedding=op.embedding,
                                        sub_tree_1=sub_stree.embedding,
                                        sub_tree_2=current_num)
                    # current_num: [1, hidden_size]

                o.append(TreeEmbedding(current_num, terminal=True))  # terminal=True: 为叶子节点
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
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_tree(input_batch, input_length, generate_nums,
                  encoder, predict, generate, merge,
                  output_lang, num_pos, batch_graph, beam_size=5,
                  english=False, max_length=MAX_OUTPUT_LENGTH):

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
    # num_pos        = [[5, 9, 20]]
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs=encoder_outputs,
                                                              num_pos=[num_pos],
                                                              batch_size=batch_size,
                                                              num_size=num_size,
                                                              hidden_size=encoder.hidden_size)
    # all_nums_encoder_outputs: [1, num_size, hidden_size]
    num_start = output_lang.num_start  # 5
    # node_stacks: TreeNode, 记录节点node中的 goal vector q
    #   初始根节点n0为goal vector q0 = problem_output
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

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
            # encoder_outputs:          [seq_len, 1, hidden_size]
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
                        TreeNode(embedding=right_child))

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

                        # op.embedding:          parent node token embedding = e(y^|P)
                        # sub_stree.embedding:   left_sub_tree_embedding     = t_l
                        # current_num.embedding: right_sub_tree_embedding    = t_r

                        # op.embedding:          [1, embedding_size]
                        # sub_stree.embedding:   [1,    hidden_size]
                        # current_num.embedding: [1,    hidden_size]
                        current_num = merge(node_embedding=op.embedding,
                                            sub_tree_1=sub_stree.embedding,
                                            sub_tree_2=current_num)
                        # sub_tree embedding(number) = current_num: [1, embedding_size]

                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))  # terminal=True: 叶子节点

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


def topdown_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch,
                       generate_nums, encoder, predict, generate, encoder_optimizer, predict_optimizer,
                       generate_optimizer, output_lang, num_pos, english=False):
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
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    max_target_length = max(target_length)
    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        for idx, l, r, node_stack, i in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                            node_stacks, target[t].tolist()):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def topdown_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, output_lang, num_pos,
                          beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

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
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
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
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                current_beams.append(
                    TreeBeam(b.score+float(tv), current_node_stack, embeddings_stacks, left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
