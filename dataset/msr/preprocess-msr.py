# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json

js_all = json.load(open('msr.json'))

train_index = set()
valid_index = set()
test_index = set()

with open('train_msr.txt') as f:
    for line in f:
        line = line.strip()
        train_index.add(int(line))

with open('valid_msr.txt') as f:
    for line in f:
        line = line.strip()
        valid_index.add(int(line))

with open('test_msr.txt') as f:
    for line in f:
        line = line.strip()
        test_index.add(int(line))

with open('train_msr.jsonl', "w+") as f:
    for idx, js in enumerate(js_all):
        if idx in train_index:
            js['idx'] = idx
            f.write(json.dumps(js) + '\n')

with open('valid_msr.jsonl', "w+") as f:
    for idx, js in enumerate(js_all):
        if idx in valid_index:
            js['idx'] = idx
            f.write(json.dumps(js) + '\n')

with open('test_msr.jsonl', "w+") as f:
    for idx, js in enumerate(js_all):
        if idx in test_index:
            js['idx'] = idx
            f.write(json.dumps(js) + '\n')
