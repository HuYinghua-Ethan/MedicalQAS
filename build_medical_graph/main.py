# -*- coding: utf-8 -*-

import json
import codecs
import threading

from tqdm import tqdm # 进度条 
from py2neo import Graph



class BuildMedicalGraph:
    def __init__(self):
        self.graph = Graph("neo4j://localhost:7687", user="neo4j", password="neo4j.")
        self.data_path = './data/medical.json'  # 数据集路径
        # 定义实体节点
        self.drugs = [] # 药物
        self.recipes = [] # 食谱
        self.foods = [] # 食物
        self.checks = [] # 检查
        self.departments = [] # 科室
        self.producers = [] # 药企
        self.diseases = [] # 疾病
        self.symptoms = [] # 症状
        self.diseases_infos = [] # 疾病所有信息
        # 构建实体间的关系
        self.rels_department = []
        self.rels_noteat = []
        self.rels_doeat = []
        self.rels_recommandeat = []
        self.rels_commonddrug = []
        self.rels_recommanddrug = []
        self.rels_check = []
        self.rels_drug_producer = []
        self.rels_symptom = []
        self.rels_acompany = []
        self.rels_category = []

    # 提取三元组
    def extract_triples(self):
        with open(self.data_path, 'r', encoding="utf-8") as f:
            for line in f:
                data_json = json.loads(line)
                disease_dict = {}
                # 记录疾病的名字
                disease = data_json['name']
                disease_dict['name'] = disease
                self.diseases.append(disease)
                # print(disease_dict)
                # input()
                disease_dict['desc'] = ''  # 疾病描述
                disease_dict['prevent'] = '' # 预防
                disease_dict['cause'] = '' # 病因
                disease_dict['easy_get'] = '' # 易感人群
                disease_dict['cure_department'] = '' # 治疗科室
                disease_dict['cure_way'] = '' # 治疗方式
                disease_dict['cure_lasttime'] = '' # 治愈周期
                disease_dict['symptom'] = '' # 症状
                disease_dict['cured_prob'] = ''# 治愈概率
                if 'symptom' in data_json:
                    self.symptoms += data_json['symptom']
                    for symptom in data_json['symptom']:  # data_json['symptom'] 是列表，里面存储一些疾病的症状
                        self.rels_symptom.append([disease,'has_symptom',symptom]) 
                        # print(self.rels_symptom) # [['肺泡蛋白质沉积症', 'has_symptom', '紫绀']] ['肺泡蛋白质沉积症', 'has_symptom', '胸痛']]
                        # input()
                if 'acompany' in data_json:
                    for acompany in data_json['acompany']: # data_json['acompany'] 是列表，里面存储相关病症
                        self.rels_acompany.append([disease,'acompany',acompany])
                        self.diseases.append(acompany)
                        # print(self.rels_acompany) # [['肺泡蛋白质沉积症', 'acompany', '多重肺部感染'], ['百日咳', 'acompany', '肺不张'], ['苯中毒', 'acompany', '贫血']]
                        # input()

                if 'desc' in data_json:
                    disease_dict['desc'] = data_json['desc']

                if 'prevent' in data_json:
                    disease_dict['prevent'] = data_json['prevent']

                if 'cause' in data_json:
                    disease_dict['cause'] = data_json['cause']

                if 'get_prob' in data_json:
                    disease_dict['get_prob'] = data_json['get_prob']

                if 'easy_get' in data_json:
                    disease_dict['easy_get'] = data_json['easy_get']
                
                if 'cure_department' in data_json:
                    cure_department = data_json['cure_department']
                    if len(cure_department) == 1:
                        self.rels_category.append([disease,'cure_department',cure_department[0]])
                    if len(cure_department) == 2:
                        big = cure_department[0]
                        small = cure_department[1]
                        self.rels_department.append([small,'belongs_to',big])
                        self.rels_category.append([disease,'cure_department','small'])

                    disease_dict['cure_department'] = cure_department
                    self.departments += cure_department

                if 'cure_way' in data_json:
                    disease_dict['cure_way'] = data_json['cure_way']

                if 'cure_lasttime' in data_json:
                    disease_dict['cure_lasttime'] = data_json['cure_lasttime']

                if 'cured_prob' in data_json:
                    disease_dict['cured_prob'] = data_json['cured_prob']

                if 'common_drug' in data_json:
                    common_drug = data_json['common_drug']
                    for drug in common_drug:
                        self.rels_commonddrug.append([disease, 'has_common_drug', drug])
                    self.drugs += common_drug

                if 'recommand_drug' in data_json:
                    recommand_drug = data_json['recommand_drug']
                    self.drugs += recommand_drug
                    for drug in recommand_drug:
                        self.rels_recommanddrug.append([disease, 'recommand_drug', drug])

                if 'not_eat' in data_json:
                    not_eat = data_json['not_eat']
                    for _not in not_eat:
                        self.rels_noteat.append([disease, 'not_eat', _not])

                    self.foods += not_eat
                    do_eat = data_json['do_eat']
                    for _do in do_eat:
                        self.rels_doeat.append([disease, 'do_eat', _do])

                    self.foods += do_eat

                if 'recommand_eat' in data_json:
                    recommand_eat = data_json['recommand_eat']
                    for _recommand in recommand_eat:
                        self.rels_recommandeat.append([disease, 'recommand_recipes', _recommand])
                    self.recipes += recommand_eat

                if 'check' in data_json:
                    check = data_json['check']
                    for _check in check:
                        self.rels_check.append([disease, 'need_check', _check])
                    self.checks += check

                if 'drug_detail' in data_json:
                    for det in data_json['drug_detail']:
                        det_spilt = det.split('(')  # det: 北京同仁堂百咳静糖浆(百咳静糖浆)  det_spilt: [北京同仁堂百咳静糖浆, 百咳静糖浆)]
                        if len(det_spilt) == 2:
                            p, d = det_spilt  # p: 北京同仁堂百咳静糖浆  d: 百咳静糖浆)
                            d = d.rstrip(')') # 百咳静糖浆) -> 百咳静糖浆
                            if p.find(d) > 0:
                                p = p.rstrip(d)  # 从字符串 p 的右侧去除指定字符 d
                            self.producers.append(p)
                            self.drugs.append(d)
                            self.rels_drug_producer.append([p, 'production', d])
                        else:
                            d = det_spilt[0]
                            self.drugs.append(d)

                self.diseases_infos.append(disease_dict)
                # print(self.diseases_infos)
                # input()
                """
                [{'name': '肺泡蛋白质沉积症', 
                  'desc': '肺泡蛋白质沉积症(简称PAP)，又称Rosen-Castle-man-Liebow综合征，是一种罕见疾病。该病以肺泡和细支气管腔内充满PAS染色阳性，来自肺的富磷脂蛋白质物质为其特征，好发于青中年，男性发病约3倍于女性。', 
                  'prevent': '1、避免感染分支杆菌病，卡氏肺囊肿肺炎，巨细胞病毒等。\n2、注意锻炼身体，提高免疫力。', 
                  'cause': '病因未明，推测与几方面因素有关：如大量粉尘吸入（铝，二氧 化硅等），机体免疫功能下降（尤其婴幼儿），遗传因素，酗酒，微生物感染等，而对于感染，有时很难确认是原发致病因素还是继发于肺泡蛋白沉着症，例如巨细胞病毒，卡氏肺孢子虫，组织胞浆菌感染等均发现有肺泡内高蛋白沉着。\n虽然启动因素尚不明确，但基本上同意发病过程为脂质代谢障碍所致，即由于机体内，外因素作用引起肺泡表面活性物质的代谢异常，到目前为止，研究较多的有肺泡巨噬细胞活力，动物实验证明巨噬细胞吞噬粉尘后其活力明显下降，而病员灌洗液中的巨噬细胞内颗粒可使正常细胞活力下降，经支气管肺泡灌洗治疗后，其肺泡巨噬细胞活力可上升，而研究未发现Ⅱ型细胞生成蛋白增加，全身脂代谢也无异常，因此目前一般认为本病与 清除能力下降有关。', 
                  'easy_get': '', 
                  'cure_department': ['内科', '呼吸内科'], 
                  'cure_way': ['支气管肺泡灌洗'], 
                  'cure_lasttime': '约3个月', 
                  'symptom': '', 
                  'cured_prob': '约40%', 
                  'get_prob': '0.00002%'}]
                """
        with open("./data/disease_infos.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.diseases_infos, ensure_ascii=False, indent=2))
        # print('-' * 10 + '三元组提取完成' + '-' * 10)
                
    def write_nodes(self,entity,entity_type):
        """
        write_nodes 函数用于创建实体节点
        MERGE：这个关键字用于查找现有节点，如果找不到，则创建一个新节点。
        """
        # print("写入 {0} 实体".format(entity_type))
        for node in set(entity):
            cql  = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(label=entity_type, entity_name=node.replace("'", ""))
            # print(cql)  # MERGE(n:drugs{name:'穿心莲内酯片'})
            # input()
            try:
                self.graph.run(cql)
                # input()
            except Exception as e:
                print(e)
                print(cql)
        # print('-' * 10 + '节点创建完成' + '-' * 10)
                
    def write_edges(self, triples, head_type, tail_type):
        """
        write_edges函数用于创建实体的之间的关系
        triples: 三元组
        head_type: 头节点
        tail_type: 尾节点
        """
        # print("写入 {0} 关系".format(triples))
        # input()
        for head, relation, tail in triples:
            cql = """MATCH(p:{head_type}),(q:{tail_type})
                    WHERE p.name='{head}' AND q.name='{tail}'
                    MERGE (p)-[r:{relation}]->(q)""".format(
                head_type=head_type, tail_type=tail_type, head=head.replace("'", ""),
                tail=tail.replace("'", ""), relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)
        # print('-' * 10 + '节点关系创建完成' + '-' * 10)
          
    # 创建实体
    def create_entities(self):
        self.write_nodes(self.drugs,'drugs')
        self.write_nodes(self.recipes, 'recipes')
        self.write_nodes(self.foods, 'foods')
        self.write_nodes(self.checks, 'checks')
        self.write_nodes(self.departments, 'departments')
        self.write_nodes(self.producers, 'producers')
        self.write_nodes(self.diseases, 'diseases')
        self.write_nodes(self.symptoms, 'symptoms')

    # 创建关系
    def create_relations(self):
        self.write_edges(self.rels_department,'departments','departments')
        self.write_edges(self.rels_noteat, 'diseases', 'foods')
        self.write_edges(self.rels_doeat, 'diseases', 'foods')
        self.write_edges(self.rels_recommandeat, 'diseases', 'recipes')
        self.write_edges(self.rels_commonddrug, 'diseases', 'drugs')
        self.write_edges(self.rels_recommanddrug, 'diseases', 'drugs')
        self.write_edges(self.rels_check, 'diseases', '检查')
        self.write_edges(self.rels_drug_producer, 'producers', 'drugs')
        self.write_edges(self.rels_symptom, 'diseases', 'symptoms')
        self.write_edges(self.rels_acompany, 'diseases', 'diseases')
        self.write_edges(self.rels_category, 'diseases', 'departments')

    def set_attributes(self, entity_infos, etype):
        """
        创建属性
        """
        # print("写入 {0} 实体的属性".format(etype))
        for e_dict in entity_infos[892:]:
            name = e_dict['name']
            # print(name)
            del e_dict['name']   # 把 name 键值删掉
            for k, v in e_dict.items():
                if k in ['cure_department', 'cure_way']:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}={v}""".format(label=etype, name=name.replace("'", ""), k=k, v=v)
                else:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}='{v}'""".format(label=etype, name=name.replace("'", ""), k=k,
                                                  v=v.replace("'", "").replace("\n", ""))
                try:
                    self.graph.run(cql)
                except Exception as e:
                    print(e)
                    print(cql)
        # print('-' * 10 + '实体属性创建完成' + '-' * 10)
        
    def set_diseases_attributes(self):
        self.set_attributes(self.diseases_infos, "diseases")


    def export_data(self, data, path):
        if isinstance(data[0], str):
            data = sorted([d.strip("...") for d in set(data)])
        with codecs.open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))

    # 将数据保存到json文件
    def export_entitys_relations(self):
        self.export_data(self.drugs, './data/json/drugs.json')
        self.export_data(self.recipes, './data/json/recipes.json')
        self.export_data(self.foods, './data/json/foods.json')
        self.export_data(self.checks, './data/json/checks.json')
        self.export_data(self.departments, './data/json/departments.json')
        self.export_data(self.producers, './data/json/producers.json')
        self.export_data(self.diseases, './data/json/diseases.json')
        self.export_data(self.symptoms, './data/json/symptoms.json')

        self.export_data(self.rels_department, './data/json/rels_department.json')
        self.export_data(self.rels_noteat, './data/json/rels_noteat.json')
        self.export_data(self.rels_doeat, './data/json/rels_doeat.json')
        self.export_data(self.rels_recommandeat, './data/json/rels_recommandeat.json')
        self.export_data(self.rels_commonddrug, './data/json/rels_commonddrug.json')
        self.export_data(self.rels_recommanddrug, './data/json/rels_recommanddrug.json')
        self.export_data(self.rels_check, './data/json/rels_check.json')
        self.export_data(self.rels_drug_producer, './data/json/rels_drug_producer.json')
        self.export_data(self.rels_symptom, './data/json/rels_symptom.json')
        self.export_data(self.rels_acompany, './data/json/rels_acompany.json')
        self.export_data(self.rels_category, './data/json/rels_category.json')
        # print('-' * 10 + '数据导出完成' + '-' * 10)


if __name__ == "__main__":
    bmg = BuildMedicalGraph()
    bmg.extract_triples()  # 提取三元组
    print('-' * 10 + '三元组提取完成' + '-' * 10)
    bmg.create_entities()  # db创建实体节点
    print('-' * 10 + '节点创建完成' + '-' * 10)
    bmg.create_relations() # db创建实体关系
    print('-' * 10 + '节点关系创建完成' + '-' * 10)
    bmg.set_diseases_attributes() # db创建实体属性
    print('-' * 10 + '实体属性创建完成' + '-' * 10)
    bmg.export_entitys_relations() # 导出实体和关系
    print('-' * 10 + '数据导出完成' + '-' * 10)


