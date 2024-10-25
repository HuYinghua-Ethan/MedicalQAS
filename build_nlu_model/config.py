# -*- coding: utf-8 -*-


"""
配置参数信息
"""

Config = {
    "model_path": "nlu_model",
    "stop_words_data_path": "./cache/hit_stopwords.txt",
    "schema_path": "./data/schema.json",
    "train_data_path": "./data/self_train.csv",
    "valid_data_path": "./data/self_test.csv",
    "model_type":"bert",
    "max_length": 50,
    "hidden_size":256,
    "kernel_size": 3,
    "num_layers": 2,
    "epochs": 20,
    "batch_size": 64,   # 或者128
    "pooling_style": "max", # 或者 avg
    "dropout": 0.1,   # 0.5
    "optimizer": "adam",
    "learning_rate": 1e-4, # 或者 1e-3, 1e-5
    "pretrain_model_path": r"D:\2024 AILearning\BaDou_Course\Practical_Project\bert-base-chinese",
    "seed": 987
}












