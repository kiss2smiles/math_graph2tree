# coding: utf-8
import os
import sys

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from math23k.src.train_and_evaluate import *
from math23k.src.models import *
from math23k.src.expressions_transfer import *
from math23k.src.bert_embedding import *


def read_json(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file


batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'


def get_train_test_fold(ori_path, prefix, data, pairs, group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test  = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path  = ori_path + mode_test + prefix
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
        # # 其中所有数字都被替换为NUM标识
        # pair[0] = 分词后的单词序列
        # eg: ['在', '一', '正方形', '花池', '的', 'NUM', '周', '栽', '了', 'NUM', '棵', '柳树', '，', '每', '两棵', '柳树', '之间',
        #      '的', '间隔', '是', 'NUM', '米', '，', '这个', '正方形', '的', '周长', '=', '多少', '米', '？']
        # pair[1] = 分词后的公式的前序表达式(ground_truth)
        # eg: ['*', 'N1', 'N2']
        # pair[2] = 文本中提到的所有数字序列
        # eg: ['4', '44', '20']
        # pair[3] = 数字在文本中的所有位置
        # eg: [5, 9, 20]
        # pair[4] = group_num
        # eg: [12, 13, 14, 24, 25, 26, 27]
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold


def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num


data       = load_raw_data("data/Math_23K.json")
group_data = read_json("data/Math_23K_processed.json")
# generate_nums: ['1', '3.14'] // 数据集中的常数
pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
# p: input_seq, out_seq, nums, num_pos
# p[0] =  [
#   '镇海', '雅乐', '学校', '二年级', '的', '小朋友', '到', '一条', '小路', '的',
#   '一边', '植树', '．', '小朋友', '们', '每隔', 'NUM', '米', '种', '一棵树',
#   '（', '马路', '两头', '都', '种', '了', '树', '）', '，', '最后',
#   '发现', '一共', '种', '了', 'NUM', '棵', '，', '这', '条', '小路',
#   '长', '多少', '米', '．'
# ]
# p[1] =  ['(', 'N1', '-', '1', ')', '*', 'N0']
# p[2] =  ['2', '11']
# p[3] =  [16, 34]
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
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

# output_vocab:
# {
#   '*': 0, '-': 1, '+': 2, '/': 3, '^': 4,
#   '1': 5, '3.14': 6,
#   'N0': 7,  'N1': 8,  'N2':  9,  'N3':  10, 'N4':  11, 'N5':  12, 'N6':  13, 'N7': 14,
#   'N8': 15, 'N9': 16, 'N10': 17, 'N11': 18, 'N12': 19, 'N13': 20, 'N14': 21,
#   'UNK': 22
# }

# output_lang.n_words: 23
# copy_nums: 15
# len(generate_nums): 2
# op_nums: = 23 - 15 - 1 - 2 = 5
# Initialize models
encoder  = EncoderSeq(  input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers)
predict  = Prediction(  hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), embedding_size=embedding_size)
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

    # ** 原始文本的单词在词典中的索引
    #   input_batches[idx] =  [1513, 324, 1000, 1000, 488,  118,  681, 965, 26,  129,
    #                          1513, 6,   965,  126,  253,  564,  85,  676, 6,   1,
    #                          26,   398, 1513, 2247, 1000, 1000, 1,   229, 103, 26,
    #                          1513, 324, 1000, 1000, 965,  229,  126, 6,   91,  71,
    #                          1,    325, 1,    13,   564,  85,   177, 965, 34,  229,
    #                          52]

    # ** 原始文本的单词序列的长度
    #   input_lengths[idx] =  51

    # ** 输出公式的单词在词典中的索引
    #   output_batches[idx] =  [3, 8, 1, 7, 3, 9, 2, 9, 10]

    # ** 输出公式的单词序列的长度
    #   output_lengths[idx] =  9

    # ** 文本中不重复的数字个数 = len(num)
    #   num_batches[idx][0] = 4

    # ** num_stack
    #   num_stack_batches[idx] =  []

    # ** 文本中的数字在原始文本中的位置
    #   num_pos_batches[idx][0] =  [19, 26, 40, 42]

    # ** 文本中的数字个数 = len(num_pos)
    #   num_size_batches[idx][0] =  4

    # ** 文本中的数字的值
    #   num_value_batches[idx][0] = ['60%', '20', '2', '3']

    # ** 常数在output_vocab中的索引位置
    #   generate_num_ids = [5, 6]

    for idx in range(len(input_lengths)):
        opr_input = ['+', '-', '*', '/', '^']
        num_value_batch = num_value_batches[idx]

        # total_len: max_num_size
        total_len = len(generate_nums) + max([len(num_value_batch[idx1]) for idx1 in range(batch_size)])

        # current_opr_embedding2: [batch_size, operator_size, 1, 1024]
        # current_num_embedding2: [batch_size, number_size + constant_size, 10, 1024]
        current_opr_embedding2 = torch.zeros(batch_size, 5,         1,  1024)
        current_num_embedding2 = torch.zeros(batch_size, total_len, 10, 1024)

        # 利用bert embedding 初始化number embedding
        for idx1 in range(batch_size):
            # numbers: ['1', '3.14', '4', '1', '10', '7', '8']
            for idx2, item in enumerate(generate_nums + num_value_batch[idx1]):
                embedding = get_embedding(item, max_seq_len=10)
                item_len  = embedding.size(0)
                current_num_embedding2[idx1, idx2, :item_len] = embedding

        # 利用bert embedding 初始化operator embedding
        for idx1 in range(batch_size):
            # opr_input: ['+', '-', '*', '/', '^']
            for idx2, item in enumerate(opr_input):
                embedding = get_embedding(item)
                current_opr_embedding2[idx1, idx2] = embedding

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

    # batch[0]: 单词在词表中的索引位置
    #   eg: [207, 35, 796, 2, 6, 1, 197, 481, 23, 1, 30, 484, 26, 58, 3269, 484, 1088, 6, 1903, 71, 1, 16, 26, 49, 796, 6, 439, 75, 34, 16, 52]
    # batch[1]: 索引序列的长度
    #   eg: 31
    # batch[2]: 数学表达式的先序遍历
    #   eg: [0, 8, 9]
    # batch[3]: 表达式序列的长度
    #   eg: 3
    # batch[4]: 文本中提到的所有数字序列
    #   eg: ['4', '44', '20']
    # batch[5]: 数字在文本中的所有位置
    #   eg: [5, 9, 20]
    # batch[6]: num_stack
    #   eg: []
    # batch[7]: group_num
    #   eg: [12, 13, 14, 24, 25, 26, 27]
    if epoch % 2 == 0 or epoch > n_epochs - 5:
        value_ac    = 0
        equation_ac = 0
        eval_total  = 0
        start = time.time()

        for test_batch in test_pairs:
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
