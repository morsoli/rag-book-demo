import json
import re
from typing import List, Dict
import spacy
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# ------------------ 查询预处理 ------------------
def query_preprocessing(query: str) -> List[str]:
    """
    查询预处理：分词、去除停用词、词形还原
    """
    # 加载中文spaCy模型（需要提前安装：pip install spacy && python -m spacy download zh_core_web_sm）
    try:
        nlp = spacy.load("zh_core_web_sm")
    except:
        print("请先安装spaCy中文模型：pip install spacy && python -m spacy download zh_core_web_sm")
        return [query]

    # 简单的停用词列表
    stop_words = {"的", "是", "在", "了", "和", "或", "一个"}
    
    doc = nlp(query)
    tokens = [token.text for token in doc if token.text not in stop_words and not token.is_punct]
    return tokens

# ------------------ 查询扩展：同义词扩展 ------------------
# 模拟的同义词字典
synonyms = {
    "鸡肉": ["鸡胸肉", "鸡腿肉", "鸡翅"],
    "牛肉": ["牛排", "牛腩", "牛肋条"],
    "面条": ["意大利面", "拉面", "挂面"]
}

def synonym_expansion(query: str) -> List[str]:
    """
    同义词扩展：添加查询词的同义词
    """
    expanded = [query]
    if query in synonyms:
        expanded.extend(synonyms[query])
    return expanded

# ------------------ 查询扩展：上下位词扩展 ------------------
class Node:
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

def build_category_tree():
    """
    构建简单的分类树（以食物为例）
    """
    food = Node("食物")
    
    meat = Node("肉类")
    noodle = Node("面食")
    food.add_child(meat)
    food.add_child(noodle)
    
    chicken = Node("鸡肉")
    beef = Node("牛肉")
    meat.add_child(chicken)
    meat.add_child(beef)
    
    pasta = Node("面条")
    noodle.add_child(pasta)
    
    return food

def get_hypernyms(node: Node) -> List[str]:
    """获取上位词"""
    hypernyms = []
    current = node.parent
    while current:
        hypernyms.append(current.name)
        current = current.parent
    return hypernyms

def get_hyponyms(node: Node) -> List[str]:
    """获取下位词"""
    return [child.name for child in node.children]

def find_node(root: Node, name: str) -> Node:
    """在树中查找节点"""
    if root.name == name:
        return root
    for child in root.children:
        found = find_node(child, name)
        if found:
            return found
    return None

def hyponym_hypernym_expansion(query: str, root: Node) -> List[str]:
    """
    上下位词扩展：添加上位词和下位词
    """
    node = find_node(root, query)
    if not node:
        return [query]
    
    expanded = [query]
    expanded.extend(get_hypernyms(node))
    expanded.extend(get_hyponyms(node))
    return expanded

# ------------------ 查询扩展：基于上下文的扩展 ------------------
def load_embedding_model():
    """
    加载预训练句子嵌入模型
    """
    print("正在加载句子嵌入模型，这可能需要一些时间...")
    try:
        model_id = "iic/nlp_gte_sentence-embedding_chinese-base"
        pipeline_se = pipeline(Tasks.sentence_embedding, model=model_id, sequence_length=512)
        print("句子嵌入模型加载完成！")
        return pipeline_se
    except:
        print("请安装modelscope并确保网络连接：pip install modelscope")
        return None

def get_sentence_embedding(sentence: str, pipeline_se) -> np.ndarray:
    """
    获取句子的嵌入向量
    """
    result = pipeline_se(input={"source_sentence": [sentence]})
    return result['text_embedding'][0]

def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量之间的余弦相似度
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def context_based_expansion(query: str, pipeline_se, candidates: List[str], top_k: int = 3) -> List[str]:
    """
    基于上下文的查询扩展：使用词嵌入模型找到语义相近的词
    """
    query_vector = get_sentence_embedding(query, pipeline_se)
    similarities = []
    
    for candidate in candidates:
        candidate_vector = get_sentence_embedding(candidate, pipeline_se)
        similarity = calculate_similarity(query_vector, candidate_vector)
        similarities.append((candidate, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in similarities[:top_k]]

# ------------------ 查询理解：实体识别 ------------------
def extract_entities(query: str) -> Dict:
    """
    实体识别：提取查询中的人名、地名和组织机构
    """
    try:
        nlp = spacy.load("zh_core_web_sm")
    except:
        print("请先安装spaCy中文模型：pip install spacy && python -m spacy download zh_core_web_sm")
        return {"人名": [], "地名": [], "组织机构": []}
    
    doc = nlp(query)
    entities = {"人名": [], "地名": [], "组织机构": []}
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["人名"].append(ent.text)
        elif ent.label_ == "GPE":
            entities["地名"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["组织机构"].append(ent.text)
    
    return entities

# ------------------ 查询理解：意图分类 ------------------
def classify_intent(query: str) -> str:
    """
    意图分类：基于简单规则判断查询意图
    """
    query = query.lower()
    if any(keyword in query for keyword in ["是什么", "介绍", "告诉我"]):
        return "information"
    elif any(keyword in query for keyword in ["设置", "执行", "打开"]):
        return "action"
    elif any(keyword in query for keyword in ["比较", "与", "vs"]):
        return "comparison"
    return "information"  # 默认意图

# ------------------ 查询重写 ------------------
def rewrite_query(query: str) -> str:
    """
    查询重写：基于规则将查询转换为结构化格式
    """
    query = query.lower()
    if "谁是" in query:
        return query.replace("谁是", "person:")
    elif "在哪里" in query:
        return query.replace("在哪里", "location:")
    elif "年" in query and re.search(r'\d{4}', query):
        year = re.search(r'\d{4}', query).group()
        return f"year:{year} {query}"
    return query

# ------------------ 综合测试函数 ------------------
def test_search_system():
    """
    测试查询转换系统
    """
    # 初始化分类树和嵌入模型
    food_tree = build_category_tree()
    pipeline_se = load_embedding_model()
    
    # 候选词列表（用于上下文扩展）
    candidates = ["人工智能", "机器学习", "深度学习", "科技发展", "疫苗研发", "气候变化", "电动汽车", "远程办公"]
    
    print("欢迎使用综合查询转换系统！")
    while True:
        query = input("\n请输入您的查询（输入'退出'结束程序）: ")
        if query == '退出':
            break
        
        print(f"\n原始查询: {query}")
        
        # 1. 查询预处理
        tokens = query_preprocessing(query)
        print(f"预处理结果: {tokens}")
        
        # 2. 查询扩展
        # 同义词扩展
        expanded_synonyms = synonym_expansion(query)
        print(f"同义词扩展: {expanded_synonyms}")
        
        # 上下位词扩展
        expanded_hyponyms = hyponym_hypernym_expansion(query, food_tree)
        print(f"上下位词扩展: {expanded_hyponyms}")
        
        # 基于上下文的扩展
        if pipeline_se:
            expanded_context = context_based_expansion(query, pipeline_se, candidates)
            print(f"基于上下文的扩展: {expanded_context}")
        
        # 3. 查询理解
        entities = extract_entities(query)
        print(f"实体识别: {json.dumps(entities, ensure_ascii=False, indent=2)}")
        
        intent = classify_intent(query)
        print(f"意图分类: {intent}")
        
        rewritten_query = rewrite_query(query)
        print(f"重写查询: {rewritten_query}")

# 运行测试
if __name__ == "__main__":
    test_search_system()