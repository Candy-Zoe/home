# Gensim Word2Vec词向量学习
# 主要内容：Word2Vec模型训练、词向量使用、相似度计算

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

print("=== 加载示例数据 ===")
print(f"示例文本: {common_texts[:3]}")

print("\n=== 训练Word2Vec模型 ===")
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
print(f"词汇表大小: {len(model.wv.index_to_key)}")
print(f"词汇表: {model.wv.index_to_key}")

print("\n=== 获取词向量 ===")
vector = model.wv['computer']
print(f"computer的词向量形状: {vector.shape}")
print(f"computer的词向量前5个值: {vector[:5]}")

print("\n=== 计算词相似度 ===")
similarity = model.wv.similarity('computer', 'laptop')
print(f"computer和laptop的相似度: {similarity:.4f}")

print("\n=== 找相似词 ===")
similar_words = model.wv.most_similar('computer', topn=3)
print("与computer相似的词:")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

print("\n=== 词向量运算 ===")
result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(f"woman + king - man = {result[0][0]} (相似度: {result[0][1]:.4f})")

print("\n=== 保存和加载模型 ===")
model.save('word2vec.model')
loaded_model = Word2Vec.load('word2vec.model')
print("模型已保存并重新加载")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('word2vec.model'):
    import shutil
    shutil.rmtree('word2vec.model')
    print("已删除测试文件")