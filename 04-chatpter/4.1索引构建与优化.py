# 导入必要的库
import datetime
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Set
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import sys
import jieba
from PIL import Image
import io
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文档数据
documents = [
    "科技：近年来，人工智能在各个领域取得了显著的进展。特别是在医疗、教育和金融领域，AI技术的应用极大地提高了效率和效果。",
    "历史：工业革命标志着历史的转折点，改变了人们的生活和工作方式。这一时期出现了大量的新发明和技术进步。",
    "文化：文化包括人类社会中的社会行为和规范。这些行为和规范通过语言、艺术、习俗等形式表现出来，反映了社会的价值观和信仰。"
]

# === 索引更新策略 ===
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

# === 索引压缩技术 ===
# 1. 字典压缩（前缀压缩）
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

# 2. 倒排列表压缩（Variable Byte Encoding）
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

# 3. 向量量化（简化的Product Quantization）
def simplified_pq(vectors: np.ndarray, n_subvectors: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """简化的向量量化"""
    n_vectors, dim = vectors.shape
    subvector_dim = dim // n_subvectors
    reshaped = vectors.reshape(n_vectors, n_subvectors, subvector_dim)
    centroids = np.mean(reshaped, axis=0)
    quantized = np.argmin(np.abs(reshaped[:, :, None] - centroids[None, :, :, None]), axis=3)
    return centroids, quantized

# === 多模态索引构建 ===
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
            similarity = 1 - cosine(query_vector, vector)
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
                similarity = 1 - cosine(query_vector, vector)
                similarities.append({'id': doc_id, 'similarity': similarity})
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            return f"图像检索失败: {str(e)}"

# 主函数：演示索引更新、压缩和多模态索引
if __name__ == "__main__":
    # === 增量更新演示 ===
    print("=== 增量更新 ===")
    doc_store = DocumentStore()
    index = Index()

    # 添加初始文档
    doc1 = Document("1", documents[0], datetime.datetime.now())
    doc2 = Document("2", documents[1], datetime.datetime.now())
    doc_store.add_or_update_document(doc1)
    doc_store.add_or_update_document(doc2)
    incremental_update(index, doc_store)

    # 模拟文档更新
    import time
    time.sleep(1)
    doc2_updated = Document("2", "历史：工业革命带来了技术进步和生产力提升。", datetime.datetime.now())
    doc3 = Document("3", documents[2], datetime.datetime.now())
    doc_store.add_or_update_document(doc2_updated)
    doc_store.add_or_update_document(doc3)
    incremental_update(index, doc_store)

    # 搜索
    print("\n搜索 '人工智能':")
    results = index.search("人工智能")
    for doc_id in results:
        print(f"找到文档 {doc_id}: {index.documents[doc_id][0]}")

    # === 全量重建演示 ===
    print("\n=== 全量重建 ===")
    index = full_reindex(doc_store)
    print("\n搜索 '文化':")
    results = index.search("文化")
    for doc_id in results:
        print(f"找到文档 {doc_id}: {doc_store.documents[doc_id].content}")

    # === 索引压缩演示 ===
    print("\n=== 索引压缩 ===")
    # 字典压缩
    words = ["人工智能", "人工神经网络", "人脸识别", "文化", "文化遗产"]
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

    # === 多模态索引演示 ===
    print("\n=== 多模态索引 ===")
    multimodal_index = MultimodalIndex()
    for i, doc in enumerate(documents, 1):
        multimodal_index.add_text_document(str(i), doc)
    
    # 模拟添加图像（需替换为实际图像URL）
    multimodal_index.add_image("img1", "https://example.com/image1.jpg")
    multimodal_index.add_image("img2", "https://example.com/image2.jpg")

    # 文本检索
    print("\n文本检索 '人工智能':")
    text_results = multimodal_index.search_text("人工智能")
    for result in text_results:
        print(f"ID: {result['id']}, Similarity: {result['similarity']:.4f}, Text: {result['text']}")

    # 图像检索（模拟）
    print("\n图像检索:")
    image_results = multimodal_index.search_image("https://example.com/image1.jpg")
    if isinstance(image_results, list):
        for result in image_results:
            print(f"ID: {result['id']}, Similarity: {result['similarity']:.4f}")
