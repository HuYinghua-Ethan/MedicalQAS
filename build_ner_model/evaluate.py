# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data



class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.schema = self.valid_data.dataset.schema

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"disease": defaultdict(int),
                           "crowd": defaultdict(int),
                           "symptom": defaultdict(int),
                           "body": defaultdict(int),
                           "treatment": defaultdict(int),
                           "time": defaultdict(int),
                           "drug": defaultdict(int),
                           "feature": defaultdict(int),
                           "physiology": defaultdict(int),
                           "test": defaultdict(int),
                           "department": defaultdict(int)}
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_id) # 经过 linear 层后输出的结果
                self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

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

    def decode(self, sentence, labels):
        labels = [x for x in labels[:len(sentence)]]
        results = defaultdict(list)

        disease_ranges = self.find_start_to_end_ranges(self.schema["B_disease"], self.schema["I_disease"], labels)
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
        

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) ==  len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)  # 预测结果的最大值
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            print("-------------------------------")
            print("真实的实体 :")
            print(true_entities)
            print("-------------------------------")
            print("预测的实体 :")
            print(pred_entities)
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本总的实体数
            for key in ["disease", "crowd", "symptom", "body", "treatment", "time", "drug", "feature", "physiology", "test", "department"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])  # 预测出的实体在真实的实体中才能说明预测的实体是识别正确
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return


    def show_stats(self):
        F1_scores = []
        for key in ["disease", "crowd", "symptom", "body", "treatment", "time", "drug", "feature", "physiology", "test", "department"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本总的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"]) # 1e-5 是一个很小的数，用于防止分母为零的情况。
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: % f" % np.mean(F1_scores))  # Macre-F1 就是所有类别的 F1 的平均值。
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["disease", "crowd", "symptom", "body", "treatment", "time", "drug", "feature", "physiology", "test", "department"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["disease", "crowd", "symptom", "body", "treatment", "time", "drug", "feature", "physiology", "test", "department"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["disease", "crowd", "symptom", "body", "treatment", "time", "drug", "feature", "physiology", "test", "department"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

