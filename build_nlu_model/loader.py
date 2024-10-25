# -*- coding: utf-8 -*-

import json
import re
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class MyDataset:
    def __init__(self, data_path, config):
        self.config = config
        self.data_path = data_path
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()

    def load(self):
        self.data = []
        # 加载停用词
        self.load_stop_words(self.config["stop_words_data_path"])
        self.load_schema(self.config["schema_path"])  # label: index
        self.index_to_label = dict((y,x) for x, y in self.schema.items())
        # print(self.index_to_label)
        self.config["class_num"] = len(self.schema)
        # print(self.config["class_num"])
        # input()

        # 加载数据
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]  # discard the first line "text,label" 
            for line in lines:
                data = line.strip().split(",")
                text = data[0].strip()  # 去除空格
                text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+?", "", text)  # 去掉一些标点
                label = data[-1].strip()  # 去除空格
                label = int(self.schema[label])
                # print(text, label)
                # input()
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
                # print(self.data)
                # input()
        return
        
    # 加载停用词
    def load_stop_words(self, stop_words_data_path):
        with open(stop_words_data_path, encoding="utf-8") as f:
            self.stop_words = f.read().split('\n')
            # print(self.stop_words[1])
        print('-' * 10 + '停用词加载完成' + '-' * 10)

    # 加载schema
    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf-8") as f:
            self.schema = json.load(f)
        # print(self.schema)
        print('-' * 10 + 'schema加载完成' + '-' * 10)

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
    

