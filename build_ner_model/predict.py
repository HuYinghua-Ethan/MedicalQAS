# -*- coding: utf-8 -*-

import torch
import re
import json
import numpy as np
from collections import defaultdict
# from config import Config
from build_ner_model.config import Config
# from model import NER_Model
from build_ner_model.model import NER_Model
from transformers import BertTokenizer

class NER_Predictor:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["class_num"] = len(self.schema)  # 类别数 23
        self.config["vocab_size"] = len(self.vocab)
        self.model = NER_Model(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print('-' * 10 + '模型加载完毕' + '-' * 10)

    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)
        
    def load_vocab(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            return json.load(f)
        
    def find_start_to_end_ranges(self, label_index_strat, label_index_end, labels):
        entities = []
        in_four_sequence = False
        start_one_index = -1

        for i, label in enumerate(labels):
            if label == label_index_strat and start_one_index == -1:
                start_one_index = i  # 记录1的起始位置
            elif label == label_index_end:
                if start_one_index != -1:
                    in_four_sequence = True  # 开始一个4的序列
            else:
                if in_four_sequence:
                    # 记录范围：从1开始到最后一个4的结束位置
                    entities.append((start_one_index, i))
                    in_four_sequence = False
                    start_one_index = -1  # 重置

        # 处理序列结束时仍然在4的情况
        if in_four_sequence and start_one_index != -1:
            entities.append((start_one_index, len(labels)))
        return entities
    
    def decode(self, labels, sentence):
        labels = [x for x in labels[:len(sentence)]]
        print(labels)
        results = defaultdict(list)

        disease_ranges = self.find_start_to_end_ranges(self.schema["B_disease"], self.schema["I_disease"], labels)
        print(disease_ranges)
        for range in disease_ranges:
            s, e = range
            results["disease"].append(sentence[s:e])
        crowd_ranges = self.find_start_to_end_ranges(self.schema["B_crowd"], self.schema["I_crowd"], labels)
        for range in crowd_ranges:
            s, e = range
            results["crowd"].append(sentence[s:e])
        symptom_ranges = self.find_start_to_end_ranges(self.schema["B_symptom"], self.schema["I_symptom"], labels)
        for range in symptom_ranges:
            s, e = range
            results["symptom"].append(sentence[s:e])
        body_ranges = self.find_start_to_end_ranges(self.schema["B_body"], self.schema["I_body"], labels)
        for range in body_ranges:
            s, e = range
            results["body"].append(sentence[s:e])
        treatment_ranges = self.find_start_to_end_ranges(self.schema["B_treatment"], self.schema["I_treatment"], labels)
        for range in treatment_ranges:
            s, e = range
            results["treatment"].append(sentence[s:e])
        time_ranges = self.find_start_to_end_ranges(self.schema["B_time"], self.schema["I_time"], labels)
        for range in time_ranges:
            s, e = range
            results["time"].append(sentence[s:e])
        drug_ranges = self.find_start_to_end_ranges(self.schema["B_drug"], self.schema["I_drug"], labels)
        for range in drug_ranges:
            s, e = range
            results["drug"].append(sentence[s:e])
        feature_ranges = self.find_start_to_end_ranges(self.schema["B_feature"], self.schema["I_feature"], labels)
        for range in feature_ranges:
            s, e = range
            results["feature"].append(sentence[s:e])
        physiology_ranges = self.find_start_to_end_ranges(self.schema["B_physiology"], self.schema["I_physiology"], labels)
        for range in physiology_ranges:
            s, e = range
            results["physiology"].append(sentence[s:e])
        test_ranges = self.find_start_to_end_ranges(self.schema["B_test"], self.schema["I_test"], labels)
        for range in test_ranges:
            s, e = range
            results["test"].append(sentence[s:e])
        department_ranges = self.find_start_to_end_ranges(self.schema["B_department"], self.schema["I_department"], labels)
        for range in department_ranges:
            s, e = range
            results["department"].append(sentence[s:e])

        return results



    def predict(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["UNK"]))  
        with torch.no_grad():
            pred_results = self.model(torch.LongTensor([input_id]))
            if not self.config["use_crf"]:
                pred_results = torch.argmax(pred_results[0], dim=-1)
            # print(pred_results) # [[0, 1, 22, 22, 16, 17, 22, 22, 22, 22, 22]]
            # input()
            pred_labels = pred_results[0]
        entities = self.decode(pred_labels, text)
        print(entities)
        


if __name__ == "__main__":
    pd = NER_Predictor(Config, "ner_model/ner_model.pth")
    
    while True:
        text = input("请输入问题: ")
        entities = pd.predict(text)
        print("实体有: ", entities)

