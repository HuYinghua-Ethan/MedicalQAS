import os
import jieba
import json
import pickle
import torch


train_path = "./data/train.txt"
test_path = "./data/test.txt"
dev_path = "./data/dev.txt"
vocab_path = './cache/vocab.json'
label_map_path = './cache/schema.json'
stopwords_path = './cache/hit_stopwords.txt'

stopwords = open(stopwords_path, encoding='utf-8').read().split('\n')

lines = []

with open(train_path, 'r', encoding='utf-8') as file:
    lines.extend(file.readlines())
with open(test_path, 'r', encoding='utf-8') as file:
    lines.extend(file.readlines())
with open(dev_path, 'r', encoding='utf-8') as file:
    lines.extend(file.readlines())


# PAD:在一个batch中不同长度的序列用该字符补齐，padding
# UNK:当验证集活测试机出现词表以外的词时，用该字符代替。unknown
vocab = {'PAD': 0, 'UNK': 1}
label_map = {}

# 将字符存入词表
for i in range(len(lines)):
    content = lines[i].split(' ')
    try:
        word = content[0]
        if word in stopwords:
            continue
        if word not in vocab and word != "\n":
            vocab[word] = len(vocab)
        
        label = content[1]
        if label.startswith(('B_','I_')):
            label = label.split('\n')
            try:
                if label_map[label[0]]:
                    continue
            except:
                label_map[label[0]] = len(label_map)
    except:
        continue
    
label_map['O'] = len(label_map)


# with open(label_map_path, 'w', encoding='utf-8') as f:
#     f.write(json.dumps(label_map, ensure_ascii=False, indent=2))


with open(vocab_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(vocab, ensure_ascii=False, indent=2))

