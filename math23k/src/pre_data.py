import re
import copy
import json
import random
import numpy as np

PAD_token = 0


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + \
                          ["N" + str(i) for i in range(copy_nums)] + ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = []
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[2:]
                out_data.append(temp)
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[:-2]
                out_data.append(temp)
                continue
    return out_data


def load_roth_data(filename):  # load the json data to dict(dict()) for roth data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = {}
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
    return out_data


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")  # 匹配文本中的所有数字

    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []

        seg       = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]  # remove "x="

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end   = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        pairs.append((input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


def transfer_english_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs.append((input_seq, eq_segs, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


def transfer_roth_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = {}
    generate_nums = {}
    copy_nums = 0
    for key in data:
        d = data[key]
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs[key] = (input_seq, eq_segs, nums, num_pos)

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


# # 其中所有数字都被替换为NUM标识
# pair[0] = input_seq  = 分词后的单词序列
# pair[1] = output_seq = 分词后的公式的前序表达式(ground_truth)
# pair[2] = numbers    = 文本中提到的所有数字序列
# pair[3] = num_pos    = 数字在文本中的所有位置
# pair[4] = group_num  = 与数字相关的所有单词信息
# num_stack: 文本中的重复数字在pair[2]中的位置 || 文本中无法识别的数字

# pair[1] =    ['/', '*', 'N1', 'N2', '5']
# output_cell: [3, 0, 8, 9, 22]
# pair[2] =    ['5', '16.5', '2.1', '5']
# num_stack:   [[0, 3]]
# 表示文本中的重复数字在公式中出现了一次

# pair[1] =    ['/', '-', 'N2', '4', '4']
# output_cell:
# pair[2] =    ['4', '1', '36', '4']
# num_stack:   [[0, 3], [0, 3]]
# 表示文本中的重复数字在公式中出现了两次

# pair[0] =  ['小',   '芳',  '家', 'NUM', '月份', '用水量', '是', 'NUM', '吨',   '，',
#             '每吨', '水',   '的', '价格', '是',   'NUM',   '元', '，',  '小',   '芳',
#             '家',   '一共', '有', 'NUM', '口',   '人',    '，', '平均', '每人', '应交',
#             '多少', '水费', '？']
# input_cell = [68,  178, 179, 1,   86,  2,  71,  1,   180, 26,
#               181, 182, 6,   183, 71,  1,  184, 26,  68,  178,
#               179, 29,  118, 1,   185, 85, 26,  186, 187, 188,
#               34,  189, 52]
# len(input_cell) = 33
# pair[1] =  ['/', '*', 'N1', 'N2', '5'] => ['/', '*', 'N1', 'N2', 'UNK']
# output_cell = [3, 0, 8, 9, 22]
# len(output_cell) = 5
# pair[2] =  ['5', '16.5', '2.1', '5']
# pair[3] =  [3, 7, 15, 23]
# pair[4] =  [3, 4, 5, 4, 5, 7, 17, 18, 19, 22, 23, 24, 25]
# num_stack: [[0, 3]]
# 当出现重复数字时，将output_seq中相应的数字位置设置为UNK

def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])

    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:  # output_seq
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:  # output_vocab
                flag_not = False
                for i, j in enumerate(pair[2]):  # numbers
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell  = indexes_from_sentence(input_lang,  pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        train_pairs.append((input_cell,
                            len(input_cell),
                            output_cell,
                            len(output_cell),
                            pair[2],
                            pair[3],
                            num_stack,
                            pair[4]))

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        test_pairs.append((input_cell,
                           len(input_cell),
                           output_cell,
                           len(output_cell),
                           pair[2],
                           pair[3],
                           num_stack,
                           pair[4]))
    print('Number of testind data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs
# pair: (input_seq(index), len(input_seq), output_seq(index), len(output_seq),
#        number_values, num_pos, num_stack, group_num)


def prepare_de_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        input_lang.add_sen_to_vocab(pair[0])
        output_lang.add_sen_to_vocab(pair[1])

    input_lang.build_input_lang(trim_min_count)

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        # train_pairs.append([input_cell,
        #                     len(input_cell),
        #                     pair[1],
        #                     0,
        #                     pair[2],
        #                     pair[3],
        #                     num_stack,
        #                     pair[4]])
        train_pairs.append([input_cell,
                            len(input_cell),
                            pair[1],
                            0,
                            pair[2],
                            pair[3],
                            num_stack])

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell,
        #                     len(input_cell),
        #                     output_cell,
        #                     len(output_cell),
        #                     pair[2],
        #                     pair[3],
        #                     num_stack,
        #                     pair[4]))

        test_pairs.append((input_cell,
                           len(input_cell),
                           output_cell,
                           len(output_cell),
                           pair[2],
                           pair[3],
                           num_stack))
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq


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


# num net graph
def get_lower_num_graph(max_len,          # max_seq_len
                        sentence_length,  # seq_len
                        num_list,         # num_value
                        id_num_list,      # num_pos(number)
                        contain_zh_flag=True):

    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)
    for i in range(sentence_length):
        diag_ele[i] = 1

    graph = np.diag(diag_ele)  # 对角矩阵
    if not contain_zh_flag:
        return graph

    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) <= float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1

    return graph  # [max_len, max_len]


def get_greater_num_graph(max_len,          # max_seq_len
                          sentence_length,  # seq_len
                          num_list,         # num_value
                          id_num_list,      # num_pos
                          contain_zh_flag=True):

    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)
    for i in range(sentence_length):
        diag_ele[i] = 1

    graph = np.diag(diag_ele)  # 对角矩阵
    if not contain_zh_flag:
        return graph

    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) > float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    return graph  # [max_len, max_len]


# attribute between graph
def get_attribute_between_graph(input_batch,         # input_var
                                max_len,             # max_seq_len
                                id_num_list,         # num_pos(number)
                                sentence_length,     # seq_len
                                quantity_cell_list,  # group_num(attribute)
                                contain_zh_flag=True):

    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1

    graph = np.diag(diag_ele)  # 对角矩阵
    if not contain_zh_flag:
        return graph

    # connection between number and group_num
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1

    # connection between same group_num
    for i in quantity_cell_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len:
                if input_batch[i] == input_batch[j]:
                    graph[i][j] = 1
                    graph[j][i] = 1

    return graph  # [max_len, max_len]


# quantity between graph
def get_quantity_between_graph(max_len,             # max_seq_len
                               id_num_list,         # num_pos(number)
                               sentence_length,     # seq_len
                               quantity_cell_list,  # group_num(attribute)
                               contain_zh_flag=True):

    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1

    graph = np.diag(diag_ele)  # 对角矩阵
    if not contain_zh_flag:
        return graph

    # connection between number and group_num
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1

    # connection between number
    for i in id_num_list:
        for j in id_num_list:
            graph[i][j] = 1
            graph[j][i] = 1

    return graph  # [max_len, max_len]


# quantity cell graph
def get_quantity_cell_graph(max_len,             # max_seq_len
                            id_num_list,         # num_pos(number)
                            sentence_length,     # seq_len
                            quantity_cell_list,  # group_num(attribute)
                            contain_zh_flag=True):

    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1

    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph

    # connection between number and group_num
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1

    return graph  # [max_len, max_len]


# 建立Quantity Cell Graph 和 Quantity Comparison Graph
def get_single_batch_graph(input_batch,
                           input_length,
                           group,
                           num_value,
                           num_pos):

    batch_graph = []
    max_len = max(input_length)  # max_seq_len

    for i in range(len(input_length)):
        input_batch_t      = input_batch[i]   # input_var
        sentence_length    = input_length[i]  # seq_len
        quantity_cell_list = group[i]         # group_num(attribute)
        num_list           = num_value[i]     # num_value
        id_num_list        = num_pos[i]       # num_pos(number)

        # Quantity Cell Graph
        graph_newc    = get_quantity_cell_graph(max_len=max_len,
                                                id_num_list=id_num_list,
                                                sentence_length=sentence_length,
                                                quantity_cell_list=quantity_cell_list)

        # Quantity Comparison Graph
        graph_greater = get_greater_num_graph(max_len=max_len,
                                              sentence_length=sentence_length,
                                              num_list=num_list,
                                              id_num_list=id_num_list)

        # Quantity Comparison Graph
        graph_lower   = get_lower_num_graph(max_len=max_len,
                                            sentence_length=sentence_length,
                                            num_list=num_list,
                                            id_num_list=id_num_list)

        # Quantity Quantity graph(数字之间建立无向边)
        graph_quanbet = get_quantity_between_graph(max_len=max_len,
                                                   id_num_list=id_num_list,
                                                   sentence_length=sentence_length,
                                                   quantity_cell_list=quantity_cell_list)

        # Quantity Cell Graph(数字与Attribute之间建立无向边)
        graph_attbet  = get_attribute_between_graph(input_batch=input_batch_t,
                                                    max_len=max_len,
                                                    id_num_list=id_num_list,
                                                    sentence_length=sentence_length,
                                                    quantity_cell_list=quantity_cell_list)

        graph_total = [
            graph_newc.tolist(),
            graph_greater.tolist(),
            graph_lower.tolist(),
            graph_quanbet.tolist(),
            graph_attbet.tolist()
        ]
        batch_graph.append(graph_total)

    batch_graph = np.array(batch_graph)  # [batch_size, 5, max_seq_len, max_seq_len]
    return batch_graph


def get_single_example_graph(input_batch,
                             input_length,
                             group,
                             num_value,
                             num_pos):
    batch_graph = []
    max_len            = input_length  # max_seq_len
    sentence_length    = input_length  # seq_len
    quantity_cell_list = group         # group_num(attribute)
    num_list           = num_value     # num_value
    id_num_list        = num_pos       # num_pos(number)

    graph_newc    = get_quantity_cell_graph(max_len=max_len,
                                            id_num_list=id_num_list,
                                            sentence_length=sentence_length,
                                            quantity_cell_list=quantity_cell_list)

    graph_quanbet = get_quantity_between_graph(max_len=max_len,
                                               id_num_list=id_num_list,
                                               sentence_length=sentence_length,
                                               quantity_cell_list=quantity_cell_list)

    graph_attbet  = get_attribute_between_graph(input_batch=input_batch,
                                                max_len=max_len,
                                                id_num_list=id_num_list,
                                                sentence_length=sentence_length,
                                                quantity_cell_list=quantity_cell_list)

    graph_greater = get_greater_num_graph(max_len=max_len,
                                          sentence_length=sentence_length,
                                          num_list=num_list,
                                          id_num_list=id_num_list)

    graph_lower   = get_greater_num_graph(max_len=max_len,
                                          sentence_length=sentence_length,
                                          num_list=num_list,
                                          id_num_list=id_num_list)

    graph_total = [
        graph_newc.tolist(),
        graph_greater.tolist(),
        graph_lower.tolist(),
        graph_quanbet.tolist(),
        graph_attbet.tolist()
    ]
    batch_graph.append(graph_total)
    batch_graph = np.array(batch_graph)  # [1, 5, max_seq_len, max_seq_len]
    return batch_graph


# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    # pair: (input_seq(index), len(input_seq), output_seq(index), len(output_seq), numbers, num_pos, num_stack, group_num)
    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _, _ in batch:
            input_length.append(i)   # len(input_seq)
            output_length.append(j)  # len(output_seq)

        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max  = input_length[0]     # max_seq_len
        output_len_max = max(output_length)  # max_tgt_len
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []

        # pair:  i         = input_seq(index),  => input_batches     ** 原始文本中单词在词表中的索引
        #        li        = len(input_seq),    => input_length      ** 原始文本中单词序列的长度
        #                  = len(number_values) => nums_batches      ** 原始文本中的数字个数
        #        j         = output_seq(index), => output_batches    ** 输出公式中的单词在词表中的索引
        #        lj        = len(output_seq),   => output_length     ** 输出公式中单词序列的长度
        #        num_stack = num_stack,         => num_stack_batches ** 原始文本中的重复单词在number_values list中的索引
        #        num_pos   = num_pos,           => num_pos_batches   ** 原始文本中的数字在原始文本中的位置
        #                  len(number_values)   => num_size_batches  ** 原始文本中的数字个数
        #        num       = number_values,     => num_value_batches ** 原始文本中的所有数字
        #        group     = group_num)         => group_batches     ** 原始文本中的所有group_num
        for i, li, j, lj, num, num_pos, num_stack, group in batch:
            num_batch.append(len(num))
            input_batch.append( pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            num_value_batch.append(num)
            group_batch.append(group)

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        group_batches.append(group_batch)
        graph_batch = get_single_batch_graph(input_batch=input_batch,
                                             input_length=input_length,
                                             group=group_batch,
                                             num_value=num_value_batch,
                                             num_pos=num_pos_batch)
        graph_batches.append(graph_batch)

    return input_batches, input_lengths, output_batches, output_lengths, \
           nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches


def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack


def prepare_de_train_batch(pairs_to_batch, batch_size, output_lang, rate, english=False):
    pairs = []
    b_pairs = copy.deepcopy(pairs_to_batch)
    for pair in b_pairs:
        p = copy.deepcopy(pair)
        pair[2] = check_bracket(pair[2], english)

        temp_out = exchange(pair[2], rate)
        temp_out = check_bracket(temp_out, english)

        p[2] = indexes_from_sentence(output_lang, pair[2])
        p[3] = len(p[2])
        pairs.append(p)

        temp_out_a = allocation(pair[2], rate)
        temp_out_a = check_bracket(temp_out_a, english)

        if temp_out_a != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out_a, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out_a)
            p[3] = len(p[2])
            pairs.append(p)

        if temp_out != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out)
            p[3] = len(p[2])
            pairs.append(p)

            if temp_out_a != pair[2]:
                p = copy.deepcopy(pair)
                temp_out_a = allocation(temp_out, rate)
                temp_out_a = check_bracket(temp_out_a, english)
                if temp_out_a != temp_out:
                    p[6] = get_num_stack(temp_out_a, output_lang, p[4])
                    p[2] = indexes_from_sentence(output_lang, temp_out_a)
                    p[3] = len(p[2])
                    pairs.append(p)
    print("this epoch training data is", len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        for i, li, j, lj, num, num_pos, num_stack in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches


# Multiplication exchange rate
def exchange(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    while idx < len(ex):
        s = ex[idx]
        if (s == "*" or s == "+") and random.random() < rate:
            lidx = idx - 1
            ridx = idx + 1
            if s == "+":
                flag = 0
                while not (lidx == -1 or ((ex[lidx] == "+" or ex[lidx] == "-") and flag == 0) or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex) or ((ex[ridx] == "+" or ex[ridx] == "-") and flag == 0) or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            else:
                flag = 0
                while not (lidx == -1
                           or ((ex[lidx] == "+" or ex[lidx] == "-" or ex[lidx] == "*" or ex[lidx] == "/") and flag == 0)
                           or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex)
                           or ((ex[ridx] == "+" or ex[ridx] == "-" or ex[ridx] == "*" or ex[ridx] == "/") and flag == 0)
                           or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            if lidx > 0 and ((s == "+" and ex[lidx - 1] == "-") or (s == "*" and ex[lidx - 1] == "/")):
                lidx -= 1
                ex = ex[:lidx] + ex[idx:ridx + 1] + ex[lidx:idx] + ex[ridx + 1:]
            else:
                ex = ex[:lidx] + ex[idx + 1:ridx + 1] + [s] + ex[lidx:idx] + ex[ridx + 1:]
            idx = ridx
        idx += 1
    return ex


def check_bracket(x, english=False):
    if english:
        for idx, s in enumerate(x):
            if s == '[':
                x[idx] = '('
            elif s == '}':
                x[idx] = ')'
        s = x[0]
        idx = 0
        if s == "(":
            flag = 1
            temp_idx = idx + 1
            while flag > 0 and temp_idx < len(x):
                if x[temp_idx] == ")":
                    flag -= 1
                elif x[temp_idx] == "(":
                    flag += 1
                temp_idx += 1
            if temp_idx == len(x):
                x = x[idx + 1:temp_idx - 1]
            elif x[temp_idx] != "*" and x[temp_idx] != "/":
                x = x[idx + 1:temp_idx - 1] + x[temp_idx:]
        while True:
            y = len(x)
            for idx, s in enumerate(x):
                if s == "+" and idx + 1 < len(x) and x[idx + 1] == "(":
                    flag = 1
                    temp_idx = idx + 2
                    while flag > 0 and temp_idx < len(x):
                        if x[temp_idx] == ")":
                            flag -= 1
                        elif x[temp_idx] == "(":
                            flag += 1
                        temp_idx += 1
                    if temp_idx == len(x):
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1]
                        break
                    elif x[temp_idx] != "*" and x[temp_idx] != "/":
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1] + x[temp_idx:]
                        break
            if y == len(x):
                break
        return x

    lx = len(x)
    for idx, s in enumerate(x):
        if s == "[":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == "]":
                    flag_b += 1
                elif x[temp_idx] == "[":
                    flag_b -= 1
                if x[temp_idx] == "(" or x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == "]" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "("
                x[temp_idx] = ")"
                continue
        if s == "(":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == ")":
                    flag_b += 1
                elif x[temp_idx] == "(":
                    flag_b -= 1
                if x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == ")" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "["
                x[temp_idx] = "]"
    return x


# Multiplication allocation rate
def allocation(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    lex = len(ex)
    while idx < len(ex):
        if (ex[idx] == "/" or ex[idx] == "*") and (ex[idx - 1] == "]" or ex[idx - 1] == ")"):
            ridx = idx + 1
            r_allo = []
            r_last = []
            flag = 0
            flag_mmd = False
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag += 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        r_last = ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                    elif ex[ridx] == "*" or ex[ridx] == "/":
                        flag_mmd = True
                        r_last = [")"] + ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                elif flag == -1:
                    r_last = ex[ridx:]
                    r_allo = ex[idx + 1: ridx]
                    break
                ridx += 1
            if len(r_allo) == 0:
                r_allo = ex[idx + 1:]
            flag = 0
            lidx = idx - 1
            flag_al = False
            flag_md = False
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag -= 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[lidx] == "+" or ex[lidx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                lidx -= 1
            if lidx != 0 and ex[lidx - 1] == "/":
                flag_al = False
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = lidx + 1
                temp_res = ex[:lidx]
                if flag_mmd:
                    temp_res += ["("]
                if lidx - 1 > 0:
                    if ex[lidx - 1] == "-" or ex[lidx - 1] == "*" or ex[lidx - 1] == "/":
                        flag_md = True
                        temp_res += ["("]
                flag = 0
                lidx += 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 0:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    temp_idx += 1
                temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo
                if flag_md:
                    temp_res += [")"]
                temp_res += r_last
                return temp_res
        if ex[idx] == "*" and (ex[idx + 1] == "[" or ex[idx + 1] == "("):
            lidx = idx - 1
            l_allo = []
            temp_res = []
            flag = 0
            flag_md = False  # flag for x or /
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag += 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[lidx] == "+":
                        temp_res = ex[:lidx + 1]
                        l_allo = ex[lidx + 1: idx]
                        break
                    elif ex[lidx] == "-":
                        flag_md = True  # flag for -
                        temp_res = ex[:lidx] + ["("]
                        l_allo = ex[lidx + 1: idx]
                        break
                elif flag == 1:
                    temp_res = ex[:lidx + 1]
                    l_allo = ex[lidx + 1: idx]
                    break
                lidx -= 1
            if len(l_allo) == 0:
                l_allo = ex[:idx]
            flag = 0
            ridx = idx + 1
            flag_al = False
            all_res = []
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag -= 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                ridx += 1
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = idx + 1
                flag = 0
                lidx = temp_idx + 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 1:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            all_res += l_allo + [ex[idx]] + ex[lidx: temp_idx] + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    if flag == 0:
                        break
                    temp_idx += 1
                if flag_md:
                    temp_res += all_res + [")"]
                elif ex[temp_idx + 1] == "*" or ex[temp_idx + 1] == "/":
                    temp_res += ["("] + all_res + [")"]
                temp_res += ex[temp_idx + 1:]
                return temp_res
        idx += 1
    return ex
