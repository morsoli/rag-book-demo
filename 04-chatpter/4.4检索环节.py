# 导入必要的库
import datetime
import sys
import numpy as np
from typing import Dict, List, Set, Tuple
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import jieba
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re
from PIL import Image
import io
import requests
from sentence_transformers import SentenceTransformer
import torch

# 示例文档数据
documents = [
    {
        'id': '1',
        'content': '人工智能近年来在医疗、教育和金融领域取得了显著进展。',
        'last_modified': datetime.datetime.now() - datetime.timedelta(days=1)
    },
    {
        'id': '2',
        'content': '工业革命标志着历史的转折点，带来了技术进步和生产力提升。',
        'last_modified': datetime.datetime.now() - datetime.timedelta(days=365)
    },
    {
        'id': '3',
        'content': '文化包括社会行为和规范，通过语言、艺术等形式表现。',
        'last_modified': datetime.datetime.now() - datetime.timedelta(days=7)
    }
]

# === 4.1 索引构建与优化 ===
# 1. 增量更新
class Document:
    def __init__(self, id: str, content: str, last_modified: datetime.datetime):
        self.id = id
        self.content = content
        self.last_modified = last_modified

class Index:
    def __init__(self):
        self.documents: Dict[str, Tuple[str, List[str]]] = {}
        self.last_update = datetime.datetime.min

    def add_document(self, doc: Document):
        """添加或更新文档到索引"""
        tokens = list(jieba.cut(doc.content))  # 使用 jieba 分词
        self.documents[doc.id] = (doc.content, tokens)

    def remove_document(self, doc_id: str):
        """从索引中删除文档"""
        if doc_id in self.documents:
            del self.documents[doc_id]

    def search(self, query: str) -> List[str]:
        """根据查询检索文档ID"""
        query_tokens = list(jieba.cut(query))
        return [doc_id for doc_id, (_, tokens) in self.documents.items() if any(token in tokens for token in query_tokens)]

class DocumentStore:
    def __init__(self):
        self.documents: Dict[str, Document] = {}

    def add_or_update_document(self, doc: Document):
        """添加或更新文档到存储"""
        self.documents[doc.id] = doc

    def get_modified_documents(self, since: datetime.datetime) -> List[Document]:
        """获取自指定时间以来修改的文档"""
        return [doc for doc in self.documents.values() if doc.last_modified > since]

    def get_all_document_ids(self) -> Set[str]:
        """获取所有文档ID"""
        return set(self.documents.keys())

    def get_all_documents(self) -> List[Document]:
        """获取所有文档"""
        return list(self.documents.values())

def incremental_update(index: Index, doc_store: DocumentStore):
    """执行增量更新"""
    modified_docs = doc_store.get_modified_documents(index.last_update)
    for doc in modified_docs:
        index.add_document(doc)
    
    all_doc_ids = doc_store.get_all_document_ids()
    indexed_doc_ids = set(index.documents.keys())
    deleted_doc_ids = indexed_doc_ids - all_doc_ids
    for doc_id in deleted_doc_ids:
        index.remove_document(doc_id)
    
    index.last_update = datetime.datetime.now()
    print(f"更新 {len(modified_docs)} 个文档，移除 {len(deleted_doc_ids)} 个文档。")

# 2. 全量重建
def full_reindex(doc_store: DocumentStore) -> Index:
    """执行全量重建"""
    new_index = Index()
    all_docs = doc_store.get_all_documents()
    
    for doc in all_docs:
        new_index.add_document(doc)
    
    print(f"重建索引，包含 {len(all_docs)} 个文档。")
    return new_index

# 3. 索引压缩技术
# 字典压缩（前缀压缩）
def prefix_compress(words: List[str]) -> List[Tuple[int, str]]:
    """使用前缀压缩减少词典大小"""
    compressed = []
    prev_word = ""
    for word in words:
        common_prefix = 0
        while (common_prefix < len(prev_word) and common_prefix < len(word) and 
               prev_word[common_prefix] == word[common_prefix]):
            common_prefix += 1
        compressed.append((common_prefix, word[common_prefix:]))
        prev_word = word
    return compressed

# 倒排列表压缩（Variable Byte Encoding）
def variable_byte_encode(number: int) -> List[int]:
    """对数字进行变长字节编码"""
    bytes_list = []
    while True:
        bytes_list.insert(0, number % 128)
        if number < 128:
            break
        number //= 128
    bytes_list[-1] += 128  # 标记最后一个字节
    return bytes_list

def variable_byte_decode(bytes_list: List[int]) -> int:
    """解码变长字节编码"""
    number = 0
    for byte in bytes_list:
        if byte >= 128:
            number = number * 128 + (byte - 128)
        else:
            number = number * 128 + byte
    return number

# 向量量化（简化的Product Quantization）
def simplified_pq(vectors: np.ndarray, n_subvectors: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """简化的向量量化"""
    n_vectors, dim = vectors.shape
    subvector_dim = dim // n_subvectors
    reshaped = vectors.reshape(n_vectors, n_subvectors, subvector_dim)
    centroids = np.mean(reshaped, axis=0)
    quantized = np.argmin(np.abs(reshaped[:, :, None] - centroids[None, :, :, None]), axis=3)
    return centroids, quantized

# 4. 多模态索引构建
class MultimodalIndex:
    def __init__(self):
        self.text_index: Dict[str, Tuple[str, np.ndarray]] = {}  # 文本索引
        self.image_index: Dict[str, np.ndarray] = {}  # 图像索引（模拟）
        self.text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_text_document(self, doc_id: str, text: str):
        """添加文本文档到索引"""
        vector = self.text_model.encode(text)
        self.text_index[doc_id] = (text, vector)

    def add_image(self, doc_id: str, image_url: str):
        """添加图像到索引（模拟特征提取）"""
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            # 模拟图像特征提取（实际应使用ResNet或ViT等模型）
            image_vector = np.random.rand(512)  # 假设512维向量
            self.image_index[doc_id] = image_vector
        except Exception as e:
            print(f"图像加载失败: {str(e)}")

    def search_text(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索文本索引"""
        query_vector = self.text_model.encode(query)
        similarities = []
        for doc_id, (text, vector) in self.text_index.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append({'id': doc_id, 'similarity': similarity, 'text': text})
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def search_image(self, image_url: str, top_k: int = 3) -> List[Dict]:
        """检索图像索引（模拟）"""
        try:
            response = requests.get(image_url)
            Image.open(io.BytesIO(response.content))
            query_vector = np.random.rand(512)  # 模拟图像特征
            similarities = []
            for doc_id, vector in self.image_index.items():
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                similarities.append({'id': doc_id, 'similarity': similarity})
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            return f"图像检索失败: {str(e)}"

# === 4.3 查询转化 ===
# 1. 查询预处理
def preprocess_query(query: str) -> List[str]:
    """查询预处理：分词、去除停用词"""
    stop_words = {'的', '是', '在', '了', '和'}
    tokens = list(jieba.cut(query))
    return [token for token in tokens if token not in stop_words]

# 2. 同义词扩展
synonyms = {
    '人工智能': ['AI', '机器学习', '深度学习'],
    '医疗': ['医学', '健康', '医院'],
    '文化': ['艺术', '传统', '习俗']
}

def synonym_expand(query: str) -> List[str]:
    """同义词扩展"""
    tokens = preprocess_query(query)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if token in synonyms:
            expanded.extend(synonyms[token])
    return list(set(expanded))

# 3. 上下位词扩展
class OntologyNode:
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []

    def add_child(self, child: 'OntologyNode'):
        child.parent = self
        self.children.append(child)

def build_ontology():
    """构建简单的本体树"""
    root = OntologyNode('知识')
    tech = OntologyNode('科技')
    culture = OntologyNode('文化')
    root.add_child(tech)
    root.add_child(culture)
    ai = OntologyNode('人工智能')
    ml = OntologyNode('机器学习')
    tech.add_child(ai)
    ai.add_child(ml)
    art = OntologyNode('艺术')
    tradition = OntologyNode('传统')
    culture.add_child(art)
    culture.add_child(tradition)
    return root

def get_hypernyms(node: OntologyNode) -> List[str]:
    """获取上位词"""
    hypernyms = []
    current = node.parent
    while current:
        hypernyms.append(current.name)
        current = current.parent
    return hypernyms

def get_hyponyms(node: OntologyNode) -> List[str]:
    """获取下位词"""
    return [child.name for child in node.children]

def find_node(root: OntologyNode, name: str) -> OntologyNode:
    """查找节点"""
    if root.name == name:
        return root
    for child in root.children:
        found = find_node(child, name)
        if found:
            return found
    return None

def ontology_expand(query: str, root: OntologyNode) -> List[str]:
    """上下位词扩展"""
    tokens = preprocess_query(query)
    expanded = []
    for token in tokens:
        expanded.append(token)
        node = find_node(root, token)
        if node:
            expanded.extend(get_hypernyms(node))
            expanded.extend(get_hyponyms(node))
    return list(set(expanded))

# 4. 基于上下文的查询扩展
print("加载句子嵌入模型...")
model_id = "iic/nlp_gte_sentence-embedding_chinese-base"
pipeline_se = pipeline(Tasks.sentence_embedding, model=model_id, sequence_length=512)
print("模型加载完成！")

def get_sentence_embedding(sentence: str) -> np.ndarray:
    """获取句子嵌入"""
    result = pipeline_se(input={"source_sentence": [sentence]})
    return result['text_embedding'][0]

def context_based_expansion(query: str, candidates: List[str], top_k: int = 3) -> List[str]:
    """基于上下文的查询扩展"""
    query_vector = get_sentence_embedding(query)
    similarities = []
    for candidate in candidates:
        candidate_vector = get_sentence_embedding(candidate)
        similarity = np.dot(query_vector, candidate_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(candidate_vector))
        similarities.append((candidate, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in similarities[:top_k]]

# 5. 实体识别
def extract_entities(query: str) -> List[Tuple[str, str]]:
    """使用 spaCy 提取实体"""
    try:
        nlp = spacy.load("zh_core_web_sm")  # 使用中文模型
        doc = nlp(query)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except:
        print("请先安装 spaCy 中文模型：python -m spacy download zh_core_web_sm")
        return []

# 6. 意图分类
def train_intent_classifier(X: List[str], y: List[str]) -> any:
    """训练意图分类器"""
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X, y)
    return model

# 7. 查询重写
def rewrite_query(query: str) -> str:
    """简单规则的查询重写"""
    query = query.lower()
    if '谁是' in query:
        return query.replace('谁是', '人名:')
    elif '哪里是' in query:
        return query.replace('哪里是', '地点:')
    elif '人工智能' in query:
        return f"主题:人工智能 {query}"
    return query

# 主函数：演示索引构建与查询转化
if __name__ == "__main__":
    # === 索引构建与优化 ===
    print("=== 增量更新 ===")
    doc_store = DocumentStore()
    index = Index()
    for doc in documents:
        doc_store.add_or_update_document(Document(**doc))
    incremental_update(index, doc_store)

    # 模拟文档更新
    import time
    time.sleep(1)
    doc_updated = Document('2', '工业革命带来了技术进步和生产力提升。', datetime.datetime.now())
    doc_new = Document('4', '云计算提升了系统可扩展性。', datetime.datetime.now())
    doc_store.add_or_update_document(doc_updated)
    doc_store.add_or_update_document(doc_new)
    incremental_update(index, doc_store)

    print("\n搜索 '人工智能':")
    results = index.search("人工智能")
    for doc_id in results:
        print(f"找到文档 {doc_id}: {index.documents[doc_id][0]}")

    print("\n=== 全量重建 ===")
    index = full_reindex(doc_store)
    print("\n搜索 '云计算':")
    results = index.search("云计算")
    for doc_id in results:
        print(f"找到文档 {doc_id}: {doc_store.documents[doc_id].content}")

    print("\n=== 索引压缩 ===")
    # 字典压缩
    words = ['人工智能', '人工神经网络', '人脸识别', '文化', '文化遗产']
    compressed_words = prefix_compress(words)
    print("字典压缩:")
    print("原始词典:", words)
    print("压缩后:", compressed_words)
    print(f"压缩前大小: {sys.getsizeof(words)} bytes")
    print(f"压缩后大小: {sys.getsizeof(compressed_words)} bytes")

    # 倒排列表压缩
    doc_ids = [1, 128, 16384]
    compressed_ids = [variable_byte_encode(id) for id in doc_ids]
    print("\n倒排列表压缩:")
    print("原始文档ID:", doc_ids)
    print("压缩后:", compressed_ids)
    decoded_ids = [variable_byte_decode(comp_id) for comp_id in compressed_ids]
    print("解码后:", decoded_ids)

    # 向量量化
    vectors = np.random.rand(5, 8)
    centroids, quantized = simplified_pq(vectors)
    print("\n向量量化:")
    print("原始向量:\n", vectors)
    print("量化后:\n", quantized)
    print(f"压缩前大小: {vectors.nbytes} bytes")
    print(f"压缩后大小: {centroids.nbytes + quantized.nbytes} bytes")

    print("\n=== 多模态索引 ===")
    multimodal_index = MultimodalIndex()
    for doc in documents:
        multimodal_index.add_text_document(doc['id'], doc['content'])
    # 模拟添加图像（需替换为实际URL）
    multimodal_index.add_image("img1", "https://example.com/image1.jpg")
    print("\n文本检索 '人工智能':")
    text_results = multimodal_index.search_text("人工智能")
    for result in text_results:
        print(f"ID: {result['id']}, Similarity: {result['similarity']:.4f}, Text: {result['text']}")

    # === 查询转化 ===
    print("\n=== 查询预处理 ===")
    query = "人工智能在医疗领域的应用"
    tokens = preprocess_query(query)
    print(f"原始查询: {query}")
    print(f"预处理后: {tokens}")

    print("\n=== 同义词扩展 ===")
    expanded = synonym_expand(query)
    print(f"扩展后: {expanded}")

    print("\n=== 上下位词扩展 ===")
    ontology_root = build_ontology()
    expanded = ontology_expand(query, ontology_root)
    print(f"扩展后: {expanded}")

    print("\n=== 基于上下文的查询扩展 ===")
    candidates = ['机器学习', '深度学习', '医疗技术', '文化艺术']
    expanded = context_based_expansion(query, candidates, top_k=2)
    print(f"扩展后: {expanded}")

    print("\n=== 实体识别 ===")
    entities = extract_entities(query)
    print(f"实体: {entities}")

    print("\n=== 意图分类 ===")
    X = ['人工智能的应用是什么？', '设置早上7点的闹钟', '比较iPhone和三星']
    y = ['information', 'action', 'comparison']
    classifier = train_intent_classifier(X, y)
    predicted_intent = classifier.predict([query])[0]
    print(f"预测意图: {predicted_intent}")

    print("\n=== 查询重写 ===")
    rewritten = rewrite_query(query)
    print(f"原始查询: {query}")
    print(f"重写后: {rewritten}")
