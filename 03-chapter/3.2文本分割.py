# 导入必要的库
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import os

# 示例文本
long_text = """
从前，有一个小村庄，村里住着一位善良的老奶奶。她每天早晨都会起床，去村外的森林里采蘑菇。森林 ♪

森林里有各种各样的蘑菇，有的可以吃，有的不能吃，老奶奶总是非常小心地挑选。

一天早晨，老奶奶像往常一样走进森林。她走啊走，发现了一棵巨大的橡树，树下长满了各种美丽的蘑菇。老奶奶高兴极了，开始采蘑菇。突然，她听到一阵奇怪的声音，好像是某种动物在哭泣。

老奶奶顺着声音找过去，发现一只小狐狸被夹在一个陷阱里，正在挣扎。老奶奶心生怜悯，决定救助这只小狐狸。她小心翼翼地打开陷阱，让小狐狸自由了。小狐狸感激地看着老奶奶，然后迅速跑进森林深处。

从那天起，每当老奶奶去森林采蘑菇的时候，总能发现一些特别大的蘑菇，好像是小狐狸特意留下来给她的礼物。老奶奶心里很高兴，她知道这是小狐狸在报恩。

就这样，老奶奶和小狐狸成了森林里一段奇妙的友谊。村子里的人们也纷纷传颂这个感人的故事，大家都说老奶奶的善良和小狐狸的感恩让整个村庄充满了温暖和爱心。
"""

# 1. 固定大小分块
def fixed_size_chunking(text, chunk_size=200):
    """使用固定大小分块方法分割文本"""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size)
    texts = text_splitter.create_documents([text])
    return texts

# 2. 递归分块
def recursive_chunking(text, chunk_size=100, chunk_overlap=20):
    """使用递归分块方法分割文本"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([text])
    return texts

# 3. 基于文档逻辑的分块 - Markdown
def markdown_chunking(markdown_text):
    """使用MarkdownHeaderTextSplitter分割Markdown格式文本"""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return markdown_splitter.split_text(markdown_text)

# 3. 基于文档逻辑的分块 - HTML
def html_chunking(html_text):
    """使用HTMLHeaderTextSplitter分割HTML格式文本"""
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return html_splitter.split_text(html_text)

# 4. 语义分块
def semantic_chunking(text, api_key):
    """使用语义分块方法分割文本"""
    os.environ["OPENAI_API_KEY"] = api_key  # 设置OpenAI API密钥
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    texts = text_splitter.create_documents([text])
    return texts

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

# 主函数：演示各种分块方法
if __name__ == "__main__":
    # 1. 固定大小分块
    print("=== 固定大小分块 ===")
    fixed_chunks = fixed_size_chunking(long_text)
    for i, chunk in enumerate(fixed_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n")

    # 2. 递归分块
    print("=== 递归分块 ===")
    recursive_chunks = recursive_chunking(long_text)
    for i, chunk in enumerate(recursive_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n")

    # 3. Markdown分块
    print("=== Markdown分块 ===")
    markdown_chunks = markdown_chunking(markdown_text)
    for i, chunk in enumerate(markdown_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n元数据: {chunk.metadata}\n")

    # 4. HTML分块
    print("=== HTML分块 ===")
    html_chunks = html_chunking(html_text)
    for i, chunk in enumerate(html_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n元数据: {chunk.metadata}\n")

    # 5. 语义分块（需要OpenAI API密钥）
    # 由于需要API密钥，这里仅展示调用方式，实际运行需替换为有效密钥
    # api_key = "<你的API_key>"
    # semantic_chunks = semantic_chunking(long_text, api_key)
    # for i, chunk in enumerate(semantic_chunks):
    #     print(f"Chunk {i+1}:\n{chunk.page_content}\n")