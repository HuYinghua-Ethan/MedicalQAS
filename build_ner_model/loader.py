# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer 
from collections import defaultdict


"""
数据加载
"""

class MyDataset:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = self.load_vocab(config["vocab_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)  # 类别数 23
        # print(self.config["class_num"])
        self.stop_words = self.load_stop_words(config["stop_words_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.load()
        print('-' * 10 + '数据加载完成' + '-' * 10)

    def load(self):
        self.data = []
        with open(self.path, encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                segment = segment.split("\n")
                for line in segment: # 一行一行遍历
                    # print(line)
                    # input()
                    if line.strip() == "":
                        continue
                    if line.strip() in self.stop_words:
                        continue
                    else:
                        char, label = line.split(' ')  # 以空格进行切分
                        # print(char)
                        # print(label)
                        # input()
                        sentence.append(char)
                        labels.append(int(self.schema[label]))
                self.sentences.append("".join(sentence)) 
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)  # token pad 的 label 为 -1
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
                # print(len(self.data[0][0]))  # 200
                # print(len(self.data[0][1]))  # 200
                # input()
        return 

        # with open(self.path, encoding="utf-8") as f:
        #     lines = f.readlines()
        # for line in lines:
        #     content = line.split(' ')
        #     if content[0] == '\n':
        #         continue
        #     if content[0] in self.stop_words:
        #         continue
        #     else:
        #         char = content[0]
        #         label = content[1]


    def load_vocab(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            return json.load(f)

    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)
    
    def load_stop_words(self, stop_words_path):
        with open(stop_words_path, encoding="utf-8") as f:
            stop_words = f.read().split('\n')
            # print(self.stop_words[1])
        print('-' * 10 + '停用词加载完成' + '-' * 10)
        return stop_words
    
    def encode_sentence(self, text, padding=True):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["UNK"]))  
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



def load_data(data_path, config, shuffle=True):
    dataset = MyDataset(data_path, config)
    dl = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    ds = MyDataset(Config["train_data_path"], Config)
