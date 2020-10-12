# coding: utf-8
import os
import sys
import json
import time
import torch

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from math23k.src.expressions_transfer import from_infix_to_prefix
from math23k.src.models import EncoderSeq
from math23k.src.models import Prediction
from math23k.src.models import GenerateNode
from math23k.src.models import Merge
from math23k.src.model_utils import compute_prefix_tree_result
from math23k.src.pre_data import load_raw_data
from math23k.src.pre_data import transfer_num
from math23k.src.pre_data import prepare_data
from math23k.src.pre_data import prepare_train_batch
from math23k.src.pre_data import get_single_example_graph
from math23k.src.train_and_evaluate import train_tree
from math23k.src.train_and_evaluate import evaluate_tree
from math23k.src.train_and_evaluate import time_since

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


# batch_size = 64
batch_size = 2
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'


def read_json(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file


def get_train_test_fold(ori_path, prefix, data, pairs, group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test  = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path  = ori_path + mode_test  + prefix
    train = read_json(train_path)
    valid = read_json(valid_path)
    test  = read_json(test_path)
    train_id = [item['id'] for item in train]
    valid_id = [item['id'] for item in valid]
    test_id  = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold  = []

    for item, pair, g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    # pair: (input_seq, output_seq(prefix), numbers, num_pos, group_num)
    return train_fold, test_fold, valid_fold


data       = load_raw_data("data/Math_23K.json")
group_data = read_json("data/Math_23K_processed.json")
# generate_nums: ['1', '3.14'] // 数据集中的常数
# copy_nums: 文本中最多出现的数字个数
pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
    # pair: (input_seq, output_seq(infix), numbers, num_pos)
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path=ori_path,
                                                        prefix=prefix,
                                                        data=data,
                                                        pairs=pairs,
                                                        group=group_data)

best_acc_fold = []
pairs_tested = test_fold
# pairs_trained = valid_fold
pairs_trained = train_fold

# pair: (input_seq(index), len(input_seq), output_seq(index), len(output_seq), numbers, num_pos, num_stack, group_num)
input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained=pairs_trained,
                                                                pairs_tested=pairs_tested,
                                                                trim_min_count=5,
                                                                generate_nums=generate_nums,
                                                                copy_nums=copy_nums,
                                                                tree=True)

# save input vocab
with open("data/input_vocab.json", "w", encoding="utf-8") as writer:
    json.dump(input_lang.word2index, writer, ensure_ascii=False, indent=4)
# save output vocab
with open("data/output_vocab.json", "w", encoding="utf-8") as writer:
    json.dump(output_lang.word2index, writer, ensure_ascii=False, indent=4)

# output_lang.n_words: 23
# copy_nums:           15 (N0-N14)
# len(generate_nums):  2
# op_nums:             5
# embedding_size:      128
# hidden_size:         512

# Initialize models
op_nums = output_lang.n_words - copy_nums - 1 - len(generate_nums)
encoder  = EncoderSeq(  input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers)
predict  = Prediction(  hidden_size=hidden_size, op_nums=op_nums, input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=op_nums, embedding_size=embedding_size)
merge    = Merge(       hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer  = torch.optim.Adam(encoder.parameters(),  lr=learning_rate, weight_decay=weight_decay)
predict_optimizer  = torch.optim.Adam(predict.parameters(),  lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer    = torch.optim.Adam(merge.parameters(),    lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler  = torch.optim.lr_scheduler.StepLR(encoder_optimizer,  step_size=20, gamma=0.5)
predict_scheduler  = torch.optim.lr_scheduler.StepLR(predict_optimizer,  step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler    = torch.optim.lr_scheduler.StepLR(merge_optimizer,    step_size=20, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    loss_total = 0
    input_batches, input_lengths,     output_batches,  output_lengths,\
    nums_batches,  num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, \
    graph_batches = prepare_train_batch(train_pairs, batch_size)

    print("epoch:", epoch + 1)
    start = time.time()

    for idx in range(len(input_lengths)):
        opr_input = ['+', '-', '*', '/', '^']
        num_value_batch = num_value_batches[idx]
        batch_size      = len(num_value_batch)

        loss = train_tree(
            input_batch=input_batches[idx],
            input_length=input_lengths[idx],
            target_batch=output_batches[idx],
            target_length=output_lengths[idx],
            nums_stack_batch=num_stack_batches[idx],
            num_size_batch=num_size_batches[idx],
            generate_nums=generate_num_ids,
            encoder=encoder,
            predict=predict,
            generate=generate,
            merge=merge,
            encoder_optimizer=encoder_optimizer,
            predict_optimizer=predict_optimizer,
            generate_optimizer=generate_optimizer,
            merge_optimizer=merge_optimizer,
            output_lang=output_lang,
            num_pos=num_pos_batches[idx],
            batch_graph=graph_batches[idx])
        loss_total += loss

    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")

    if epoch % 2 == 0 or epoch > n_epochs - 5:
        value_ac    = 0
        equation_ac = 0
        eval_total  = 0
        start = time.time()

        for test_batch in test_pairs:  # batch_size = 1
            batch_graph = get_single_example_graph(input_batch=test_batch[0],
                                                   input_length=test_batch[1],
                                                   group=test_batch[7],
                                                   num_value=test_batch[4],
                                                   num_pos=test_batch[5])

            test_res = evaluate_tree(
                input_batch=test_batch[0],
                input_length=test_batch[1],
                generate_nums=generate_num_ids,
                encoder=encoder,
                predict=predict,
                generate=generate,
                merge=merge,
                output_lang=output_lang,
                num_pos=test_batch[5],
                batch_graph=batch_graph,
                beam_size=beam_size
            )

            val_ac, equ_ac, _, _ = compute_prefix_tree_result(
                test_res=test_res,
                test_tar=test_batch[2],
                output_lang=output_lang,
                num_list=test_batch[4],
                num_stack=test_batch[6]
            )
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1

        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        torch.save(encoder.state_dict(),  "model_traintest/encoder")
        torch.save(predict.state_dict(),  "model_traintest/predict")
        torch.save(generate.state_dict(), "model_traintest/generate")
        torch.save(merge.state_dict(),    "model_traintest/merge")
        if epoch == n_epochs - 1:
            best_acc_fold.append((equation_ac, value_ac, eval_total))

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))
