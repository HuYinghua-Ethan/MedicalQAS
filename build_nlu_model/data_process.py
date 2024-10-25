

""""
建立 schema.json 文件，将标签转化为对应的index
"""
from collections import defaultdict
import json



labels = set()

with open("./data/self_train.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]  # discard the first line "text,label" 
    for line in lines:
        data = line.strip().split(",")
        label = data[-1].strip()  # 去除空格
        labels.add(label)
        
label_to_index = {}
for index, label in enumerate(labels):
    label_to_index[label] = index
    
# 将 label_to_index 保存到schema.json文件中
with open("./data/schema.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(label_to_index, ensure_ascii=False, indent=2))

