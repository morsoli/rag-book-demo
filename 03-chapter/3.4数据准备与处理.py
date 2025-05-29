# 导入必要的库
import re
import hashlib
import jieba
import numpy as np
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import requests
from bs4 import BeautifulSoup

# 示例文本
long_text = """
从前，有一个小村庄，村里住着一位善良的老奶奶。她每天早晨都会起床，去村外的森林里采蘑菇。森林里有各种各样的蘑菇，有的可以吃，有的不能吃，老奶奶总是非常小心地挑选。

一天早晨，老奶奶像往常一样走进森林。她走啊走，发现了一棵巨大的橡树，树下长满了各种美丽的蘑菇。老奶奶高兴极了，开始采蘑菇。突然，她听到一阵奇怪的声音，好像是某种动物在哭泣。

老奶奶顺着声音找过去，发现一只小狐狸被夹在一个陷阱里，正在挣扎。老奶奶心生怜悯，决定救助这只小狐狸。她小心翼翼地打开陷阱，让小狐狸自由了。小狐狸感激地看着老奶奶，然后迅速跑进森林深处。

从那天起，每当老奶奶去森林采蘑菇的时候，总能发现一些特别大的蘑菇，好像是小狐狸特意留下来给她的礼物。老奶奶心里很高兴，她知道这是小狐狸在报恩。

就这样，老奶奶和小狐狸成了森林里一段奇妙的友谊。村子里的人们也纷纷传颂这个感人的故事，大家都说老奶奶的善良和小狐狸的感恩让整个村庄充满了温暖和爱心。
"""

# 示例Markdown文本
markdown_text = """
# 标题1

## 子标题1

你好，我是张三

你好，我是李四

### 子标题2

你好，我是王五

## 子标题3

你好，我是赵六
"""

# 示例HTML文本
html_text = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>主标题</h1>
        <p>关于主标题的介绍文本。</p>
        <div>
            <h2>二级标题1</h2>
            <p>关于二级标题1的介绍文本。</p>
            <h3>三级标题1-1</h3>
            <p>关于二级标题1下第一个子主题的文本。</p>
            <h3>三级标题1-2</h3>
            <p>关于二级标题1下第二个子主题的文本。</p>
        </div>
        <div>
            <h2>二级标题2</h2>
            <p>关于二级标题2的文本。</p>
        </div>
        <br>
        <p>关于主标题的总结文本。</p>
    </div>
</body>
</html>
"""

# 示例文档集（用于文档摘要索引）
documents = [
    "科技：近年来，人工智能在各个领域取得了显著的进展。特别是在医疗、教育和金融领域，AI技术的应用极大地提高了效率和效果。",
    "历史：工业革命标志着历史的转折点，改变了人们的生活和工作方式。这一时期出现了大量的新发明和技术进步。",
    "文化：文化包括人类社会中的社会行为和规范。这些行为和规范通过语言、艺术、习俗等形式表现出来，反映了社会的价值观和信仰。"
]

# === 数据清洗 ===
# 1.1 读取文本文件
def read_txt_file(file_path: str) -> str:
    """读取txt文件并返回内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"读取文件失败: {str(e)}"

# 1.2 读取PDF文件
def read_pdf_file(file_path: str) -> List:
    """使用langchain的PyPDFLoader读取PDF文件"""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        return pages
    except Exception as e:
        return f"读取PDF文件失败: {str(e)}"

# 1.3 网页爬取
def scrape_webpage(url: str) -> str:
    """使用requests和BeautifulSoup爬取网页内容"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)
    except Exception as e:
        return f"网页爬取失败: {str(e)}"

# 1.4 使用langchain的AsyncHtmlLoader爬取网页
async def scrape_webpage_async(urls: List[str]) -> List:
    """使用langchain的AsyncHtmlLoader异步爬取网页"""
    try:
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()
        return docs
    except Exception as e:
        return f"异步网页爬取失败: {str(e)}"

# 2.1 去除特殊字符和标点符号
def remove_special_characters(text: str) -> str:
    """去除文本中的特殊字符和标点符号，保留中文、字母、数字和空格"""
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return cleaned_text

# 2.2 去除HTML标签
def clean_html_content(docs: List) -> List:
    """使用langchain的Html2TextTransformer将HTML内容转换为纯文本"""
    try:
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        return docs_transformed
    except Exception as e:
        return f"HTML内容清洗失败: {str(e)}"

# 2.3 转换为小写（适用于英文文本）
def convert_to_lowercase(text: str) -> str:
    """将英文文本转换为小写"""
    return text.lower()

# 2.4 去除停用词
stopwords_chinese = ['的', '是', '在', '这', '个', '了', '和', '与', '及']
def remove_stopwords_chinese(text: str) -> List[str]:
    """去除中文停用词"""
    words = list(jieba.cut(text))
    return [word for word in words if word not in stopwords_chinese]

# 3.1 文本分词 - jieba
def jieba_segmentation(text: str, mode: str = 'precise') -> List[str]:
    """使用jieba进行中文分词"""
    if mode == 'full':
        return list(jieba.cut(text, cut_all=True))
    elif mode == 'search':
        return list(jieba.cut_for_search(text))
    else:
        return list(jieba.cut(text, cut_all=False))

# 3.2 文本分词 - TF-IDF
def tfidf_keywords(corpus: List[str], max_features: int = 10) -> List[str]:
    """使用TF-IDF提取关键词"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(corpus)
    return list(vectorizer.get_feature_names_out())

# === 文本分割 ===
# 1. 固定大小分块
def fixed_size_chunking(text: str, chunk_size: int = 200) -> List:
    """使用固定大小分块方法分割文本"""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size)
    return text_splitter.create_documents([text])

# 2. 递归分块
def recursive_chunking(text: str, chunk_size: int = 100, chunk_overlap: int = 20) -> List:
    """使用递归分块方法分割文本"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.create_documents([text])

# 3. Markdown分块
def markdown_chunking(markdown_text: str) -> List:
    """使用MarkdownHeaderTextSplitter分割Markdown格式文本"""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return markdown_splitter.split_text(markdown_text)

# 3. HTML分块
def html_chunking(html_text: str) -> List:
    """使用HTMLHeaderTextSplitter分割HTML格式文本"""
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return html_splitter.split_text(html_text)

# 4. 语义分块
def semantic_chunking(text: str, api_key: str) -> List:
    """使用语义分块方法分割文本"""
    os.environ["OPENAI_API_KEY"] = api_key
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    return text_splitter.create_documents([text])

# === 索引构建 ===
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
    sections = text.strip().split('\n\n')
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

# 主函数：演示数据清洗、文本分割和索引构建
if __name__ == "__main__":
    # === 数据清洗演示 ===
    print("=== 数据清洗 ===")
    # 去除特殊字符
    cleaned_text = remove_special_characters(long_text[:100])
    print("去除特殊字符:", cleaned_text)

    # 转换为小写（英文示例）
    english_text = "Hello, World! This is a TEST."
    lowercase_text = convert_to_lowercase(english_text)
    print("转换为小写:", lowercase_text)

    # 去除停用词
    filtered_text = remove_stopwords_chinese(long_text[:100])
    print("去除停用词:", filtered_text)

    # jieba分词
    seg_list = jieba_segmentation(long_text[:100], mode='precise')
    print("jieba分词（精确模式）:", "/".join(seg_list))

    # TF-IDF关键词提取
    tfidf_words = tfidf_keywords([long_text[:100], long_text[100:200]])
    print("TF-IDF关键词:", tfidf_words)

    # === 文本分割演示 ===
    print("\n=== 文本分割 ===")
    # 固定大小分块
    print("固定大小分块:")
    fixed_chunks = fixed_size_chunking(long_text)
    for i, chunk in enumerate(fixed_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n")

    # 递归分块
    print("递归分块:")
    recursive_chunks = recursive_chunking(long_text)
    for i, chunk in enumerate(recursive_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n")

    # Markdown分块
    print("Markdown分块:")
    markdown_chunks = markdown_chunking(markdown_text)
    for i, chunk in enumerate(markdown_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n元数据: {chunk.metadata}\n")

    # HTML分块
    print("HTML分块:")
    html_chunks = html_chunking(html_text)
    for i, chunk in enumerate(html_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n元数据: {chunk.metadata}\n")

    # === 索引构建演示 ===
    print("\n=== 索引构建 ===")
    # 列表索引
    print("列表索引:")
    list_chunks = list_indexing(long_text)
    for chunk in list_chunks:
        print(f"ID: {chunk['id']}, Text: {chunk['text']}")
    query = "人工智能"
    list_results = search_list_index(query, list_chunks)
    print("\n检索结果:")
    for result in list_results:
        print(f"ID: {result['id']}, Text: {result['text']}")

    # 关键词表索引
    print("\n关键词表索引:")
    keyword_index = extract_keywords(list_chunks)
    for keyword, ids in keyword_index.items():
        print(f"Keyword: {keyword}, IDs: {ids}")
    keyword_results = search_keyword_index(query, keyword_index, list_chunks)
    print("\n检索结果:")
    for result in keyword_results:
        print(f"ID: {result['id']}, Text: {result['text']}")

    # 向量索引
    print("\n向量索引:")
    chunks = [chunk['text'] for chunk in list_chunks]
    vector_index = text_to_vectors(chunks)
    for vector in vector_index:
        print(f"ID: {vector['id']}, Vector: {vector['vector'][:5]}..., Text: {vector['text']}")
    vector_results = search_vector_index(query, vector_index)
    print("\n检索结果:")
    for result in vector_results:
        print(f"ID: {result['id']}, Similarity: {result['similarity']:.4f}, Text: {result['text']}")

    # 树索引
    print("\n树索引:")
    tree_root = create_tree_structure(long_text)
    def print_tree(node, level=0):
        print("  " * level + f"ID: {node.id}, Text: {node.text}")
        for child in node.children:
            print_tree(child, level + 1)
    print_tree(tree_root)
    tree_results = search_tree(query, tree_root)
    print("\n检索结果:")
    for result in tree_results:
        print(f"ID: {result.id}, Text: {result.text}")

    # 文档摘要索引
    print("\n文档摘要索引:")
    summary_index = create_summary_index(documents)
    for doc in summary_index:
        print(f"ID: {doc['id']}, Summary: {doc['summary']}, Document: {doc['document']}")
    summary_results = search_summary_index(query, summary_index)
    print("\n检索结果:")
    for result in summary_results:
        print(f"ID: {result['id']}, Summary: {result['summary']}, Document: {result['document']}")