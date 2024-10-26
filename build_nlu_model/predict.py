# -*- coding: utf-8 -*-

import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import NLU_Model
from transformers import BertTokenizer

class Predictor:
    def __init__(self, config, model_path):
        self.config = config
        self.attribute_schema = json.load(open(config["schema_path"], encoding="utf8"))
        self.index_to_label = dict((y, x) for x, y in self.attribute_schema.items())
        self.config["class_num"] = len(self.attribute_schema)
        self.model = NLU_Model(config)
        model_type = config["model_type"]
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print('-' * 10 + '模型加载完成' + '-' * 10)

    def predict(self, text):
        input_id = []
        if self.config["model_type"] == "bert":
            input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True)
        with torch.no_grad():
            attribute_pred = self.model(torch.LongTensor([input_id]))
            attribute_label = torch.argmax(attribute_pred)
        pred_attribute = self.index_to_label[int(attribute_label)]
        return pred_attribute


if __name__ == "__main__":
    pd = Predictor(Config, "nlu_model/nlu_model.pth")
    
    while True:
        text = input("请输入问题: ")
        intent = pd.predict(text)
        print("问题类型: ", intent)
        
