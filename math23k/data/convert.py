import json

train_path = "train23k_processed.json"
valid_path = "valid23k_processed.json"
test_path  = "test23k_processed.json"
whole_path = "whole_processed.json"

_train_path = "train23k_processed_utf8.json"
_valid_path = "valid23k_processed_utf8.json"
_test_path  = "test23k_processed_utf8.json"
_whole_path = "whole_processed_utf8.json"
with open(file=whole_path, mode="r", encoding="utf-8") as reader:
    input_data = json.load(reader)

with open(file=_whole_path, mode="w", encoding="utf-8") as writer:
    json.dump(input_data, writer, ensure_ascii=False, indent=4)
