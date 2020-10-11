import os
import sys
import random
import torch.optim
import torch.nn.functional as f

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
gra_dir = os.path.dirname(par_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)

from math23k.src.model_utils import generate_decoder_input
from math23k.src.model_utils import generate_rule_mask
from math23k.src.masked_cross_entropy import masked_cross_entropy

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score      = score
        self.input_var  = input_var
        self.hidden     = hidden
        self.all_output = all_output


def train_attn(input_batch,       input_length,     target_batch, target_length,
               num_batch,         nums_stack_batch, copy_nums,    generate_nums,
               encoder,           decoder,
               encoder_optimizer, decoder_optimizer,
               output_lang,       clip=0,
               use_teacher_forcing=1, beam_size=1, english=False):

    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2  # 5
    unk       = output_lang.word2index["UNK"]        # 22

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target    = torch.LongTensor(target_batch).transpose(0, 1)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask  = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_seqs=input_var,
                                              input_lengths=input_length,
                                              hidden=None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length   = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(input_seq=decoder_input,
                                                     last_hidden=decoder_hidden,
                                                     encoder_outputs=encoder_outputs,
                                                     seq_mask=seq_mask)

            all_decoder_outputs[t] = decoder_output

            decoder_input = generate_decoder_input(target=target[t],
                                                   decoder_output=decoder_output,
                                                   nums_stack_batch=nums_stack_batch,
                                                   num_start=num_start,
                                                   unk=unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()

        beam_list.append(Beam(score=score,
                              input_var=decoder_input,
                              hidden=decoder_hidden,
                              all_output=all_decoder_outputs))

        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden  = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length,      batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden  = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input  = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input=decoder_input,
                                               nums_batch=num_batch,
                                               word2index=output_lang.word2index,
                                               batch_size=batch_size,
                                               nums_start=num_start,
                                               copy_nums=copy_nums,
                                               generate_nums=generate_nums,
                                               english=english)

                if USE_CUDA:
                    rule_mask     = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(input_seq=decoder_input,
                                                         last_hidden=decoder_hidden,
                                                         encoder_outputs=encoder_outputs,
                                                         seq_mask=seq_mask)

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

                beam_list.append(Beam(score=topv[:, k],
                                      input_var=temp_input,
                                      hidden=temp_hidden,
                                      all_output=temp_output))

        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(target=target[t],
                                               decoder_output=all_decoder_outputs[t],
                                               nums_stack_batch=nums_stack_batch,
                                               num_start=num_start,
                                               unk=unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        logits=all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target=target.transpose(0, 1).contiguous(),  # -> batch x seq
        length=target_length
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


def evaluate_attn(input_seq,   input_length,
                  num_list,    copy_nums,    generate_nums,
                  encoder,     decoder,
                  output_lang, beam_size=1,  english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask  = torch.ByteTensor(1, input_length).fill_(0)
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
    encoder_outputs, encoder_hidden = encoder(input_seqs=input_var,
                                              input_lengths=[input_length],
                                              hidden=None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    beam_list = list()
    score = 0
    beam_list.append(Beam(score=score,
                          input_var=decoder_input,
                          hidden=decoder_hidden,
                          all_output=[]))

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

            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(input_seq=decoder_input,
                                                     last_hidden=decoder_hidden,
                                                     encoder_outputs=encoder_outputs,
                                                     seq_mask=seq_mask)

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
            temp_list.append(Beam(score=float(topv[k]),
                                  input_var=temp_input,
                                  hidden=temp_hidden,
                                  all_output=temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0].all_output
