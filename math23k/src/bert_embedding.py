import os
import sys
import torch
cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
gra_dir = os.path.dirname(par_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)
from math23k.pretrained_bert.configuration_bert import BertConfig
from math23k.pretrained_bert.modeling_bert_layer1_emb import BertForMaskedLM
from math23k.pretrained_bert.tokenization_bert import BertTokenizer

bert_path   = os.path.join(par_dir, "pretrained_model/op1")
config_path = os.path.join(bert_path, "config.json")
bert_config = BertConfig.from_pretrained(config_path)
model       = BertForMaskedLM(bert_config).cuda()

state_dict_path = os.path.join(bert_path, "pytorch_model.bin")
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict['model'])
model.eval()

vocab_path = os.path.join(bert_path, 'vocab_digit.txt')
tokenizer = BertTokenizer.from_pretrained(vocab_path)


def get_embedding(seq, max_seq_len=10):
    tokens = tokenizer.tokenize(seq)
    tokens = tokens[:max_seq_len]

    input_len = len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    attention_mask = [1] * input_len
    token_type_ids = [0] * input_len

    input_ids = torch.LongTensor([input_ids]).cuda()
    attention_mask = torch.LongTensor([attention_mask]).cuda()
    token_type_ids = torch.LongTensor([token_type_ids]).cuda()
    with torch.no_grad():
        emb_output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    emb_output = emb_output[0]
    return emb_output


seq = "113"
emb = get_embedding(seq)
print("emb size = ", emb.size())
