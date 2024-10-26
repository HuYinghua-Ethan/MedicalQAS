# -*- coding: utf-8 -*-

"""
配置参数信息
"""


Config = {
    "model_path": "ner_model",
    "stop_words_path": "./cache/hit_stopwords.txt",
    "schema_path": "./cache/schema.json",
    "vocab_path": "./cache/vocab.json",
    "train_data_path": "./data/train.txt",
    "valid_data_path": "./data/test.txt",
    "dev_data_path": "./data/dev.txt",
    "model_type":"bert",
    "max_length": 200,
    "hidden_size":256,
    "kernel_size": 3,
    "num_layers": 2,
    "use_crf": False,   # 使用CRF
    "epochs": 50,
    "batch_size": 32,   # 或者128
    "pooling_style": "max", # 或者 avg
    "dropout": 0.1,   # 0.5
    "optimizer": "adam",
    "learning_rate": 1e-3, # 或者 1e-3, 1e-5
    "pretrain_model_path": r"D:\2024 AILearning\BaDou_Course\Practical_Project\bert-base-chinese",
    "seed": 987
}


