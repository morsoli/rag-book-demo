# 导入必要的库
import math
import datetime
import torch
from typing import List, Dict, Set, Tuple
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import textstat
from datetime import datetime, timedelta
import jieba

# 示例文档数据
documents = [
    {
        'id': 1,
        'content': '人工智能近年来在医疗、教育和金融领域取得了显著进展。',
        'timestamp': datetime.now() - timedelta(days=1),
        'citation_count': 5000,
        'relevance_score': 0.9
    },
    {
        'id': 2,
        'content': '工业革命标志着历史的转折点，带来了技术进步和生产力提升。',
        'timestamp': datetime.now() - timedelta(days=365),
        'citation_count': 1000,
        'relevance_score': 0.7
    },
    {
        'id': 3,
        'content': '文化包括社会行为和规范，通过语言、艺术等形式表现。',
        'timestamp': datetime.now() - timedelta(days=7),
        'citation_count': 3000,
        'relevance_score': 0.8
    }
]

# === 精确匹配检索 ===
# 1. 倒排索引
class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}

    def add_document(self, doc_id: int, content: str):
        """添加文档到倒排索引"""
        words = list(jieba.cut(content))  # 使用 jieba 分词支持中文
        for word in words:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)

    def search(self, query: str) -> Set[int]:
        """根据查询检索文档ID"""
        query_words = list(jieba.cut(query))
        if not query_words:
            return set()
        result = self.index.get(query_words[0], set())
        for word in query_words[1:]:
            result = result.intersection(self.index.get(word, set()))
        return result

# 2. B树
class BTreeNode:
    def __init__(self, leaf: bool = False):
        self.leaf = leaf
        self.keys = []
        self.child = []

class BTree:
    def __init__(self, t: int):
        self.root = BTreeNode(True)
        self.t = t

    def insert(self, k: int):
        """插入键"""
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.child.insert(0, root)
            self._split_child(temp, 0)
            self._insert_non_full(temp, k)
        else:
            self._insert_non_full(root, k)

    def _insert_non_full(self, x: BTreeNode, k: int):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.child[i].keys) == (2 * self.t) - 1:
                self._split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self._insert_non_full(x.child[i], k)

    def _split_child(self, x: BTreeNode, i: int):
        t = self.t
        y = x.child[i]
        z = BTreeNode(y.leaf)
        x.child.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        z.keys = y.keys[t: (2 * t) - 1]
        y.keys = y.keys[0: t - 1]
        if not y.leaf:
            z.child = y.child[t: 2 * t]
            y.child = y.child[0: t - 1]

    def search(self, k: int, x: BTreeNode = None) -> Tuple[BTreeNode, int]:
        """搜索键"""
        if x is None:
            x = self.root
        i = 0
        while i < len(x.keys) and k > x.keys[i]:
            i += 1
        if i < len(x.keys) and k == x.keys[i]:
            return (x, i)
        elif x.leaf:
            return None
        else:
            return self.search(k, x.child[i])

# === 相似度检索 ===
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 * magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """计算欧氏距离"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

# === 语义检索 ===
def get_bert_embedding(text: str) -> np.ndarray:
    """获取 BERT 嵌入"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 使用中文模型
    model = BertModel.from_pretrained('bert-base-chinese')
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# === 混合检索策略 ===
class HybridSearchEngine:
    def __init__(self):
        self.inverted_index = InvertedIndex()
        self.document_vectors: Dict[int, np.ndarray] = {}
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

    def add_document(self, doc_id: int, content: str):
        """添加文档到倒排索引和向量存储"""
        self.inverted_index.add_document(doc_id, content)
        self.document_vectors[doc_id] = get_bert_embedding(content)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """混合检索：倒排索引筛选 + 语义相似度排序"""
        candidate_docs = self.inverted_index.search(query)
        query_vector = get_bert_embedding(query)
        similarities = []
        for doc_id in candidate_docs:
            similarity = cosine_similarity(query_vector[0], self.document_vectors[doc_id][0])
            similarities.append((doc_id, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# === 检索结果排序与过滤 ===
# 1. 相关性得分排序
def relevance_score_sort(query: str, docs: List[Dict], similarity_func) -> List[Tuple[Dict, float]]:
    """基于相关性得分排序"""
    query_vector = get_bert_embedding(query)[0]
    scored_docs = []
    for doc in docs:
        doc_vector = get_bert_embedding(doc['content'])[0]
        score = similarity_func(query_vector, doc_vector)
        scored_docs.append((doc, score))
    return sorted(scored_docs, key=lambda x: x[1], reverse=True)

# 2. 时间因素排序
def time_based_sort(docs: List[Dict], timestamp_key: str, recency_weight: float = 0.5) -> List[Dict]:
    """基于时间因素排序"""
    now = datetime.now()
    def score(doc):
        age = now - doc[timestamp_key]
        age_score = 1 / (age.total_seconds() + 1)  # 避免除以零
        return doc['relevance_score'] * (1 - recency_weight) + age_score * recency_weight
    return sorted(docs, key=score, reverse=True)

# 3. 权威性排序
def authority_based_sort(docs: List[Dict], authority_score_func, authority_weight: float = 0.3) -> List[Dict]:
    """基于权威性排序"""
    def score(doc):
        return doc['relevance_score'] * (1 - authority_weight) + authority_score_func(doc) * authority_weight
    return sorted(docs, key=score, reverse=True)

def citation_count_score(doc: Dict) -> float:
    """计算引用计数得分"""
    return min(doc['citation_count'] / 10000, 1)  # 归一化

# 4. 多样性排序
def diversity_based_sort(docs: List[Dict], top_k: int = 5, diversity_threshold: float = 0.7) -> List[Dict]:
    """基于多样性排序"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc['content'] for doc in docs])
    sorted_docs = sorted(docs, key=lambda x: x['relevance_score'], reverse=True)
    diverse_docs = [sorted_docs[0]]
    for doc in sorted_docs[1:]:
        if len(diverse_docs) >= top_k:
            break
        doc_vector = tfidf_matrix[docs.index(doc)]
        max_similarity = max(cosine_similarity(doc_vector, tfidf_matrix[docs.index(d)])[0][0] for d in diverse_docs)
        if max_similarity < diversity_threshold:
            diverse_docs.append(doc)
    return diverse_docs

# 5. 内容过滤
def content_filter(docs: List[Dict], keywords: List[str], min_keyword_count: int = 1) -> List[Dict]:
    """基于关键词过滤"""
    def contains_keywords(doc):
        return sum(1 for keyword in keywords if keyword.lower() in doc['content'].lower()) >= min_keyword_count
    return list(filter(contains_keywords, docs))

# 6. 质量过滤
def quality_filter(docs: List[Dict], min_length: int = 50, min_readability_score: float = 50) -> List[Dict]:
    """基于质量过滤"""
    def meets_quality_standards(doc):
        content = doc['content']
        return len(content) >= min_length and textstat.flesch_reading_ease(content) >= min_readability_score
    return list(filter(meets_quality_standards, docs))

# 7. 时间范围过滤
def time_range_filter(docs: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
    """基于时间范围过滤"""
    def within_time_range(doc):
        return start_date <= doc['timestamp'] <= end_date
    return list(filter(within_time_range, docs))

# 8. 综合排序与过滤
def comprehensive_rank_and_filter(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """综合排序与过滤策略"""
    # 内容过滤
    filtered_docs = content_filter(docs, query.split())
    # 质量过滤
    filtered_docs = quality_filter(filtered_docs)
    # 相关性得分排序
    scored_docs = relevance_score_sort(query, filtered_docs, cosine_similarity)
    scored_docs = [doc for doc, _ in scored_docs]  # 提取文档
    # 时间因素排序
    time_sorted_docs = time_based_sort(scored_docs, 'timestamp')
    # 多样性排序
    diverse_docs = diversity_based_sort(time_sorted_docs, top_k)
    return diverse_docs[:top_k]

# 主函数：演示检索策略
if __name__ == "__main__":
    # === 精确匹配检索 ===
    print("=== 倒排索引检索 ===")
    inverted_index = InvertedIndex()
    for doc in documents:
        inverted_index.add_document(doc['id'], doc['content'])
    query = "人工智能 医疗"
    results = inverted_index.search(query)
    print(f"查询 '{query}' 结果: {results}")

    print("\n=== B树检索 ===")
    b_tree = BTree(3)
    for doc_id in [doc['id'] for doc in documents]:
        b_tree.insert(doc_id)
    print(f"搜索文档ID 1: {b_tree.search(1)}")
    print(f"搜索文档ID 4: {b_tree.search(4)}")

    # === 相似度检索 ===
    print("\n=== 余弦相似度与欧氏距离 ===")
    doc_vectors = [[1, 1, 1, 0, 0], [0, 1, 1, 1, 1]]
    query_vector = [1, 0, 1, 0, 1]
    print(f"余弦相似度 (Doc1, Query): {cosine_similarity(doc_vectors[0], query_vector)}")
    print(f"余弦相似度 (Doc2, Query): {cosine_similarity(doc_vectors[1], query_vector)}")
    print(f"欧氏距离 (Doc1, Query): {euclidean_distance(doc_vectors[0], query_vector)}")
    print(f"欧氏距离 (Doc2, Query): {euclidean_distance(doc_vectors[1], query_vector)}")

    # === 语义检索 ===
    print("\n=== BERT 语义检索 ===")
    query = "人工智能在医疗领域的应用"
    similarities = []
    query_embedding = get_bert_embedding(query)[0]
    for doc in documents:
        doc_embedding = get_bert_embedding(doc['content'])[0]
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc['id'], similarity))
    for doc_id, sim in similarities:
        print(f"文档 {doc_id} 相似度: {sim:.4f}")

    # === 混合检索 ===
    print("\n=== 混合检索 ===")
    hybrid_engine = HybridSearchEngine()
    for doc in documents:
        hybrid_engine.add_document(doc['id'], doc['content'])
    results = hybrid_engine.search(query, top_k=2)
    for doc_id, similarity in results:
        print(f"文档 {doc_id}: 相似度 {similarity:.4f}")

    # === 检索结果排序与过滤 ===
    print("\n=== 综合排序与过滤 ===")
    results = comprehensive_rank_and_filter(query, documents)
    for doc in results:
        print(f"文档 {doc['id']}: {doc['content']}")
