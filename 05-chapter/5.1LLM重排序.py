import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from typing import List, Dict

# ------------------ 基于嵌入的重排序 ------------------
def load_embedding_model():
    """
    加载预训练句子嵌入模型
    """
    print("正在加载句子嵌入模型，这可能需要一些时间...")
    try:
        model_id = "iic/nlp_gte_sentence-embedding_chinese-base"
        pipeline_se = pipeline(Tasks.sentence_embedding, model_id=model_id, sequence_length=512)
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

def embedding_based_rerank(query: str, documents: List[Dict], pipeline_se) -> List[Dict]:
    """
    基于嵌入的重排序：根据查询与文档的语义相似度重新排序
    """
    inputs = {
        "source_sentence": [query],
        "sentences_to_compare": [doc["title"] + " " + doc["summary"] for doc in documents]
    }
    
    result = pipeline_se(input=inputs)
    scored_docs = list(zip(documents, result['scores']))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs]

# ------------------ 基于LLM的重排序 ------------------
def llm_based_rerank(query: str, documents: List[Dict]) -> List[int]:
    """
    基于LLM的重排序：模拟大模型根据提示词进行重排序
    注：这里仅模拟Prompt输出，实际应用需调用大模型API
    """
    # 构造Prompt
    context = "\n".join([f"[{i+1}] {doc['title']}：{doc['summary']}" for i, doc in enumerate(documents)])
    prompt = f"""
我将提供多篇文章，每篇文章前都有一个[num]的索引，其中num是数字。请根据与用户查询的相关性对这些文章进行重新排序。

文章来源:
{context}

用户查询: {query}

请执行以下任务:
1. 分析每篇文章与用户查询的相关性
2. 根据相关性从高到低对文章进行排序
3. 以列表形式输出排序结果，每行一个文章索引
4. 只输出相关的文章索引，不相关的无需列出
5. 不要输出任何额外的解释或评论

输出示例:
[2]
[5]
[1]

如果所有文章都不相关，请输出"没有相关文章"。
"""
    # 模拟LLM输出（实际需替换为大模型API调用）
    # 这里根据语义相关性手动模拟排序结果
    scored_docs = []
    for i, doc in enumerate(documents):
        relevance = 0
        if "人工智能" in doc["summary"] and "医疗" in doc["summary"]:
            relevance = 0.9 if "诊断" in doc["summary"] else 0.8
        elif "人工智能" in doc["summary"]:
            relevance = 0.7
        scored_docs.append((i + 1, relevance))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    relevant_indices = [index for index, score in scored_docs if score > 0]
    
    return relevant_indices if relevant_indices else ["没有相关文章"]

# ------------------ 测试函数 ------------------
def test_reranking_system():
    """
    测试重排序系统：比较基于嵌入和基于LLM的重排序结果
    """
    # 示例文档集合
    documents = [
        {"title": "人工智能在医疗诊断中的应用", "summary": "机器学习算法分析患者症状和医疗历史，辅助医生进行更准确的诊断，特别是在皮肤癌检测中表现优异。"},
        {"title": "可再生能源的进展", "summary": "太阳能和风能发电成本下降，清洁能源更经济，减少对化石燃料的依赖。"},
        {"title": "人工智能在医疗影像分析中的作用", "summary": "深度学习模型快速处理X光片和MRI图像，提高诊断速度和准确性，减轻医疗资源压力。"},
        {"title": "全球气候变化的影响", "summary": "北极冰盖融化、海平面上升和极端天气频发，科学家呼吁减少温室气体排放。"},
        {"title": "人工智能技术综述", "summary": "AI技术在各行业应用广泛，包括医疗领域，用于分析数据和辅助诊断。"}
    ]
    
    # 加载嵌入模型
    pipeline_se = load_embedding_model()
    
    print("欢迎使用重排序系统！")
    while True:
        query = input("\n请输入您的查询（输入'退出'结束程序）: ")
        if query == '退出':
            break
        
        print(f"\n用户查询: {query}")
        
        # 基于嵌入的重排序
        if pipeline_se:
            print("\n基于嵌入的重排序结果:")
            reranked_docs = embedding_based_rerank(query, documents, pipeline_se)
            for i, doc in enumerate(reranked_docs, 1):
                print(f"{i}. 标题: {doc['title']}")
                print(f"   摘要: {doc['summary']}")
                print()
        
        # 基于LLM的重排序
        print("\n基于LLM的重排序结果（模拟）：")
        reranked_indices = llm_based_rerank(query, documents)
        if reranked_indices == ["没有相关文章"]:
            print("没有相关文章")
        else:
            for index in reranked_indices:
                doc = documents[index - 1]
                print(f"[{index}] 标题: {doc['title']}")
                print(f"    摘要: {doc['summary']}")
                print()

# 运行测试
if __name__ == "__main__":
    test_reranking_system()