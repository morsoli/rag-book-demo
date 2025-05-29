# 导入必要的库
import hashlib
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# 示例文章
article = """
引言：Lorem ipsum dolor sit amet, consectetur adipiscing elit。
科技：近年来，人工智能在各个领域取得了显著的进展。
历史：工业革命标志着历史的转折点，改变了人们的生活和工作方式。
文化：文化包括人类社会中的社会行为和规范。
"""

# 示例文档集（用于文档摘要索引）
documents = [
    "科技：近年来，人工智能在各个领域取得了显著的进展。特别是在医疗、教育和金融领域，AI技术的应用极大地提高了效率和效果。",
    "历史：工业革命标志着历史的转折点，改变了人们的生活和工作方式。这一时期出现了大量的新发明和技术进步。",
    "文化：文化包括人类社会中的社会行为和规范。这些行为和规范通过语言、艺术、习俗等形式表现出来，反映了社会的价值观和信仰。"
]

# 1. 列表索引
def list_indexing(text: str, length: int = 100) -> List[Dict]:
    """创建列表索引，将文本按固定长度切分为文本块"""
    chunks = [text[i:i+length] for i in range(0, len(text), length)]
    indexed_chunks = []
    for chunk in chunks:
        identifier = hashlib.md5(chunk.encode()).hexdigest()
        indexed_chunks.append({'id': identifier, 'text': chunk})
    return indexed_chunks

def search_list_index(query: str, indexed_chunks: List[Dict]) -> List[Dict]:
    """根据关键词在列表索引中检索相关文本块"""
    results = []
    for chunk in indexed_chunks:
        if query.lower() in chunk['text'].lower():
            results.append(chunk)
    return results

# 2. 关键词表索引
def extract_keywords(chunks: List[Dict]) -> Dict[str, List[str]]:
    """使用TF-IDF提取关键词并构建倒排索引"""
    texts = [chunk['text'] for chunk in chunks]
    vectorizer = TfidfVectorizer(max_features=10)
    vectorizer.fit(texts)
    keywords = vectorizer.get_feature_names_out()
    
    keyword_index = {}
    for i, chunk in enumerate(chunks):
        for keyword in keywords:
            if keyword in chunk['text']:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(chunk['id'])
    return keyword_index

def search_keyword_index(query: str, keyword_index: Dict[str, List[str]], indexed_chunks: List[Dict]) -> List[Dict]:
    """根据关键词在倒排索引中检索相关文本块"""
    results = []
    for keyword in query.split():
        if keyword in keyword_index:
            for id in keyword_index[keyword]:
                for chunk in indexed_chunks:
                    if chunk['id'] == id and chunk not in results:
                        results.append(chunk)
    return results

# 3. 向量索引
def text_to_vectors(chunks: List[str]) -> List[Dict]:
    """将文本块转换为向量并创建向量索引"""
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    vectors = []
    for chunk in chunks:
        identifier = hashlib.md5(chunk.encode()).hexdigest()
        vector = model.encode(chunk)
        vectors.append({'id': identifier, 'vector': vector, 'text': chunk})
    return vectors

def search_vector_index(query: str, indexed_vectors: List[Dict], top_k: int = 3) -> List[Dict]:
    """根据查询向量在向量索引中检索最相似的文本块"""
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_vector = model.encode(query)
    similarities = []
    for vector in indexed_vectors:
        similarity = 1 - cosine(query_vector, vector['vector'])
        similarities.append({'id': vector['id'], 'similarity': similarity, 'text': vector['text']})
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

# 4. 树索引
class TreeNode:
    def __init__(self, identifier: str, text: str):
        self.id = identifier
        self.text = text
        self.children = []

def create_tree_structure(text: str) -> TreeNode:
    """创建树形索引结构"""
    root = TreeNode("root", "文档根节点")
    sections = text.strip().split('\n')
    for i, section in enumerate(sections, 1):
        node = TreeNode(str(i), section.strip())
        node.children.append(TreeNode(f"{i}.1", section.strip()))
        root.children.append(node)
    return root

def search_tree(query: str, node: TreeNode) -> List[TreeNode]:
    """在树索引中检索相关节点"""
    results = []
    if query.lower() in node.text.lower():
        results.append(node)
    for child in node.children:
        results.extend(search_tree(query, child))
    return results

# 5. 文档摘要索引
def generate_summary(text: str) -> str:
    """生成文档摘要（简单截断，实际应用可使用LLM）"""
    return text[:50] + "..."

def create_summary_index(documents: List[str]) -> List[Dict]:
    """创建文档摘要索引"""
    indexed_documents = []
    for doc in documents:
        summary = generate_summary(doc)
        identifier = hashlib.md5(summary.encode()).hexdigest()
        indexed_documents.append({'id': identifier, 'summary': summary, 'document': doc})
    return indexed_documents

def search_summary_index(query: str, indexed_documents: List[Dict]) -> List[Dict]:
    """在文档摘要索引中检索相关文档"""
    results = []
    for doc in indexed_documents:
        if query.lower() in doc['summary'].lower():
            results.append(doc)
    return results

# 主函数：演示各种索引方法
if __name__ == "__main__":
    # 1. 列表索引
    print("=== 列表索引 ===")
    list_chunks = list_indexing(article)
    for chunk in list_chunks:
        print(f"ID: {chunk['id']}, Text: {chunk['text']}")
    query = "人工智能"
    list_results = search_list_index(query, list_chunks)
    print("\n检索结果:")
    for result in list_results:
        print(f"ID: {result['id']}, Text: {result['text']}")

    # 2. 关键词表索引
    print("\n=== 关键词表索引 ===")
    keyword_index = extract_keywords(list_chunks)
    for keyword, ids in keyword_index.items():
        print(f"Keyword: {keyword}, IDs: {ids}")
    keyword_results = search_keyword_index(query, keyword_index, list_chunks)
    print("\n检索结果:")
    for result in keyword_results:
        print(f"ID: {result['id']}, Text: {result['text']}")

    # 3. 向量索引
    print("\n=== 向量索引 ===")
    chunks = [chunk['text'] for chunk in list_chunks]
    vector_index = text_to_vectors(chunks)
    for vector in vector_index:
        print(f"ID: {vector['id']}, Vector: {vector['vector'][:5]}..., Text: {vector['text']}")
    vector_results = search_vector_index(query, vector_index)
    print("\n检索结果:")
    for result in vector_results:
        print(f"ID: {result['id']}, Similarity: {result['similarity']:.4f}, Text: {result['text']}")

    # 4. 树索引
    print("\n=== 树索引 ===")
    tree_root = create_tree_structure(article)
    def print_tree(node, level=0):
        print("  " * level + f"ID: {node.id}, Text: {node.text}")
        for child in node.children:
            print_tree(child, level + 1)
    print_tree(tree_root)
    tree_results = search_tree(query, tree_root)
    print("\n检索结果:")
    for result in tree_results:
        print(f"ID: {result.id}, Text: {result.text}")

    # 5. 文档摘要索引
    print("\n=== 文档摘要索引 ===")
    summary_index = create_summary_index(documents)
    for doc in summary_index:
        print(f"ID: {doc['id']}, Summary: {doc['summary']}, Document: {doc['document']}")
    summary_results = search_summary_index(query, summary_index)
    print("\n检索结果:")
    for result in summary_results:
        print(f"ID: {result['id']}, Summary: {result['summary']}, Document: {result['document']}")