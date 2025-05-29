 
import numpy as np
from scipy.spatial.distance import cosine
 
# 提取嵌入向量
embedding_0 = response.data[0].embedding
embedding_1 = response.data[1].embedding
embedding_2 = response.data[2].embedding

# 计算余弦相似度
similarity_0_1 = 1 - cosine(embedding_0, embedding_1)  # 银行与金钱
similarity_0_2 = 1 - cosine(embedding_0, embedding_2)  # 银行与苹果

print(f"'银行'和'金钱'的相似度: {similarity_0_1}")
print(f"'银行'和'苹果'的相似度: {similarity_0_2}")


similarities = [
    ("'银行' and '金钱'", similarity_0_1),
    ("'银行' and '苹果'", similarity_0_2)
]

sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

print("相似度由高到底排序:")
for pair, similarity in sorted_similarities:
    print(f"{pair}: {similarity}")
