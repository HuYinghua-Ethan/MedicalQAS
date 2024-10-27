
import json
import difflib

from py2neo import Graph
from slot_config import *
from build_nlu_model.predict import NLU_Predictor
from build_ner_model.predict import NER_Predictor


graph = Graph("neo4j://localhost:7687", user="neo4j", password="neo4j.")
unrecognized_replay = unrecognized['replay_answer'] # 非常抱歉，我还不知道如何回答您，我正在努力学习中~


# 找相似的实体
def entity_similarity(entity_type, text):
    cql = "MATCH (n:{entity_type}) RETURN n.name".format(entity_type=entity_type + 's')
    entities = graph.run(cql).data()
    """
    entities = graph.run(cql).data() 返回的是一个字典列表，
    包含了从图数据库中查询到的实体数据。
    具体来说，每个字典代表一个实体，字典的键通常是列名，值是对应的实体属性
    [
        {'name': 'Entity1'},
        {'name': 'Entity2'},
    ...
    ]
    """
    max_sim_entity = ''
    max_lev_distance = 0
    for i, item in enumerate(entities):
        # 计算Levenshtein距离 - 编辑距离
        """
        Levenshtein距离（或称编辑距离）是一个衡量两个字符串之间差异的指标。
        具体来说，它表示将一个字符串转换成另一个字符串所需的最少单字符编辑操作次数，
        这些操作包括：
        插入：在字符串中插入一个字符。
        删除：从字符串中删除一个字符。
        替换：将字符串中的一个字符替换为另一个字符。
        Levenshtein距离的值越小，表示两个字符串越相似，反之则表示它们越不同。
        """
        val = list(item.values())[0]
        # difflib.SequenceMatcher(None, text, val).ratio() 的作用是计算两个字符串 text 和 val 之间的相似度比例
        # 相似度越高，输出值越大
        lev_distance = difflib.SequenceMatcher(None, text, val).ratio()
        if lev_distance > 0.6 and lev_distance < 1:
            if lev_distance > max_lev_distance:
                max_lev_distance = lev_distance
                max_sim_entity = val
    return max_sim_entity
        


"""
实体识别的模型需要再改进一下
或者可以考虑用正则表达式来做实体识别
"""
def robot_answer(text, old_entity=None, old_intent=None):
    if text == "是的":
        user_intent = old_intent
        entity_info = old_entity
    elif text == "不是":
        return unrecognized_replay
    else:  # 识别用户问题的意图和实体
        # 意图识别
        nlu_pd = NLU_Predictor(r"D:\2024 AILearning\BaDou_Course\Practical_Project\Medical_QAS_Base_On_KG\build_nlu_model\nlu_model\nlu_model.pth")
        user_intent = nlu_pd.predict(text)
        user_intent = user_intent.strip(' ')  # 去除空格
        # print(user_intent)  # 治疗方法
        # input()
        # 实体识别 
        # entity_info = NER_Predictor.predict(text) # 输出 [[entity, entity_type]]
        entity_info = [["口腔炎", "disease"]] # 手动测试
    try:
        slot_info = slot_dict[user_intent]
    except KeyError:  # 没有识别到意图
        return unrecognized_replay
    
    cql_template = slot_info.get('cql_template')  # 根据意图获取对应的cql语句
    deny_response = slot_info.get('deny_response')
    ask_template = slot_info.get('ask_template')    
    reply_template = slot_info.get('reply_template')

    if not entity_info:  # 实体不存在，但是有意图，可能是二次提问的情况
        if user_intent:
            entity_info = old_entity  # 将上次提问的实体赋予本次问话中
    # print(user_intent, entity_info)
    if entity_info: # [[entity, entity_type]]
        entity_type = entity_info[0][1]
        if entity_type != 'disease':  # 目前只能根据疾病来回答问题
            return unrecognized_replay
        entity = entity_info[0][0]
        if isinstance(cql_template, list):  # 多查询语句
            res = []
            for q in cql_template:
                cql = q.format(Disease=entity)
                data = graph.run(cql).data()
                res.extend([list(item.values())[0] for item in data if list(item.values())[0] != None])
        else:  # 单查询语句    
            cql = cql_template.format(Disease=entity)
            # print(cql)
            data = graph.run(cql).data()
            res = [list(item.values())[0] for item in data if list(item.values())[0] != None]
        
        if not res:  # 如果没有检测到答案
            # 检测是否存在该实体
            cql = "MATCH(p:diseases) WHERE p.name='{Disease}' RETURN p.name".format(Disease=entity)
            data = graph.run(cql).data()

            if not data:  # 实体不存在
                # 文本相似度匹配
                sim_entity = entity_similarity(entity_type, entity)
                reply = ask_template.format(Disease=sim_entity)
                entity_info[0][0] = sim_entity
                return [reply, entity_info, user_intent]
            # 如果没有找到相似的实体
            reply = deny_response.format(Disease=entity)
        else:
            answer = "、".join([str(i) for i in res])  # 把答案串起来
            reply_template = reply_template.format(Disease=entity)
            reply = reply_template + answer
        return [reply, entity_info, user_intent]
    else:
        return unrecognized_replay 

        


if __name__ == "__main__":
    old_entity = ''
    old_intent = ''
    # while True:
    #     input_text = input("请输入您的问题：")
    #     if old_entity:
    #         reply = robot_answer(input_text, old_entity, old_intent)
    #     else:
    #         reply = robot_answer(input_text)
    input_text = "口腔炎有哪些治疗手段？"
    if old_entity:
        reply = robot_answer(input_text, old_entity, old_intent)
    else:
        reply = robot_answer(input_text)

    if isinstance(reply, list):
        answer = reply[0]
        old_entity = reply[1]
        old_intent = reply[2]
    else:
        answer = reply
        old_entity = ''
        old_intent = ''

    print(answer)
    



