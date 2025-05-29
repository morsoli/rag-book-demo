# 导入必要的库
import re
from langchain_community.document_loaders import PyPDFLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import requests
from bs4 import BeautifulSoup

# 1. 数据收集
# 1.1 读取文本文件
def read_txt_file(file_path):
    """读取txt文件并返回内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"读取文件失败: {str(e)}"

# 1.2 读取PDF文件
def read_pdf_file(file_path):
    """使用langchain的PyPDFLoader读取PDF文件"""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        return pages
    except Exception as e:
        return f"读取PDF文件失败: {str(e)}"

# 1.3 网页爬取
def scrape_webpage(url):
    """使用requests和BeautifulSoup爬取网页内容"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        soup = BeautifulSoup(response.text, 'html.parser')
        # 提取所有<p>标签的文本内容
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)
    except Exception as e:
        return f"网页爬取失败: {str(e)}"

# 1.4 使用langchain的AsyncHtmlLoader爬取网页
async def scrape_webpage_async(urls):
    """使用langchain的AsyncHtmlLoader异步爬取网页"""
    try:
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()
        return docs
    except Exception as e:
        return f"异步网页爬取失败: {str(e)}"

# 2. 文本处理
# 2.1 去除特殊字符和标点符号
def remove_special_characters(text):
    """去除文本中的特殊字符和标点符号，保留中文、字母、数字和空格"""
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return cleaned_text

# 2.2 去除HTML标签
def clean_html_content(docs):
    """使用langchain的Html2TextTransformer将HTML内容转换为纯文本"""
    try:
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        return docs_transformed
    except Exception as e:
        return f"HTML内容清洗失败: {str(e)}"

# 2.3 转换为小写（适用于英文文本）
def convert_to_lowercase(text):
    """将英文文本转换为小写"""
    return text.lower()

# 示例使用
if __name__ == "__main__":
    # 示例1: 读取文本文件
    txt_file_path = "example.txt"  # 替换为实际文件路径
    txt_content = read_txt_file(txt_file_path)
    print("文本文件内容:", txt_content)

    # 示例2: 读取PDF文件
    pdf_file_path = "example_data/layout-parser-paper.pdf"  # 替换为实际PDF文件路径
    pdf_pages = read_pdf_file(pdf_file_path)
    if isinstance(pdf_pages, list):
        print("PDF第一页内容:", pdf_pages[0].page_content[:200])  # 打印前200字符

    # 示例3: 网页爬取
    url = "https://example.com"  # 替换为实际URL
    web_content = scrape_webpage(url)
    print("网页内容:", web_content[:200])  # 打印前200字符

    # 示例4: 异步网页爬取
    urls = ["https://example.com", "https://example.org"]  # 替换为实际URL列表
    # 注意：异步函数需要在异步环境中运行，这里仅展示调用方式
    # docs = await scrape_webpage_async(urls)

    # 示例5: 去除特殊字符
    sample_text = "这是一个示例文本，包含！@#￥%……&*（）—+{}【】、|：“；‘’《》，。？特殊字符和标点符号。"
    cleaned_text = remove_special_characters(sample_text)
    print("清洗后的文本:", cleaned_text)

    # 示例6: 转换英文文本为小写
    sample_english_text = "Hello, World! This is a SAMPLE Text."
    lowercase_text = convert_to_lowercase(sample_english_text)
    print("小写英文文本:", lowercase_text)