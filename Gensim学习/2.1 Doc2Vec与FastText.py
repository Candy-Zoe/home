# Gensim Doc2Vec与FastText学习
# 主要内容：Doc2Vec文档向量、FastText词向量、模型训练与应用

from gensim.models import Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import common_texts

print("=== 创建文档数据 ===")
documents = [
    TaggedDocument(words=["i", "love", "natural", "language", "processing"], tags=["doc_0"]),
    TaggedDocument(words=["machine", "learning", "is", "fun"], tags=["doc_1"]),
    TaggedDocument(words=["deep", "learning", "models", "are", "powerful"], tags=["doc_2"]),
    TaggedDocument(words=["python", "is", "great", "for", "data", "science"], tags=["doc_3"]),
    TaggedDocument(words=["artificial", "intelligence", "is", "the", "future"], tags=["doc_4"])
]

print("\n=== 训练Doc2Vec模型 ===")
model = Doc2Vec(vector_size=50, min_count=1, epochs=40)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
print("Doc2Vec模型训练完成")

print("\n=== 获取文档向量 ===")
doc_vec = model.dv["doc_0"]
print(f"文档向量形状: {doc_vec.shape}")
print(f"文档向量前5个值: {doc_vec[:5]}")

print("\n=== 文档相似度 ===")
similar_docs = model.dv.most_similar("doc_0", topn=3)
print("与doc_0最相似的文档:")
for doc_id, score in similar_docs:
    print(f"  {doc_id}: {score:.4f}")

print("\n=== 推断新文档向量 ===")
new_doc = ["natural", "language", "understanding"]
inferred_vec = model.infer_vector(new_doc)
print(f"推断向量前5个值: {inferred_vec[:5]}")

print("\n=== 训练FastText模型 ===")
sentences = [
    ["i", "love", "natural", "language", "processing"],
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "models", "are", "powerful"],
    ["python", "is", "great", "for", "data", "science"]
]

fasttext_model = FastText(vector_size=50, window=3, min_count=1)
fasttext_model.build_vocab(sentences=sentences)
fasttext_model.train(sentences=sentences, total_examples=len(sentences), epochs=10)
print("FastText模型训练完成")

print("\n=== FastText词向量 ===")
word_vec = fasttext_model.wv["learning"]
print(f"词向量形状: {word_vec.shape}")
print(f"词向量前5个值: {word_vec[:5]}")

print("\n=== 词相似度 ===")
similar_words = fasttext_model.wv.most_similar("learning", topn=3)
print("与learning最相似的词:")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

print("\n=== 类比推理 ===")
try:
    analogy = fasttext_model.wv.most_similar(positive=["machine", "learning"], negative=["deep"])
    print("类比推理结果:")
    for word, score in analogy[:3]:
        print(f"  {word}: {score:.4f}")
except:
    print("类比推理不可用")

print("\n=== 保存和加载模型 ===")
model.save("doc2vec.model")
fasttext_model.save("fasttext.model")
print("模型已保存")

loaded_model = Doc2Vec.load("doc2vec.model")
print("模型已加载")

print("\n=== 清理测试文件 ===")
import os
for f in ["doc2vec.model", "doc2vec.model.trainables.syn1neg.npy", "doc2vec.model.wv.vectors.npy", "fasttext.model", "fasttext.model.trainables.syn1neg.npy", "fasttext.model.wv.vectors.npy"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除 {f}")