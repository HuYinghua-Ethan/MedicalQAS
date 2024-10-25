# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel


class NLU_Model(nn.Module):
    def __init__(self, config):
        super(NLU_Model, self).__init__()
        model_type = config["model_type"]
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]
        dropout = config["dropout"]
        self.use_bert = False
        if model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size  # BERT的hidden_size = 768

        self.classify = nn.Linear(hidden_size, class_num)  # 映射到 class_num 维度
        self.pooling_style = config["pooling_style"]
        self.dropout = nn.Dropout(dropout)  # dropout
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:
            x = self.encoder(x)

        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]  # shape: (batch_size, seq_len, hidden_size)

        # 池化操作
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze() # (batch_size, sen_len, input_dim) -> (batch_size, hidden_size)
        # 也可以直接使用序列最后一个位置的向量 (batch_size, hidden_size)
        # x = x[:, -1, :]
        # x = self.dropout(x) # dropout  等一下取消注释观察效果会不会好点
        predict = self.classify(x)  # (batch_size, hidden_size) -> (batch_size, class_num)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict
        
# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
