# Gensim主题模型进阶学习
# 主要内容：LDA调优、LSI模型、HDP模型、相似度查询

from gensim import corpora, models, similarities
from gensim.models import LdaModel, LsiModel, HdpModel, TfidfModel
from gensim.models.coherencemodel import CoherenceModel
import numpy as np

print("=== 创建文档集 ===")
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary ordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of well quasi ordering",
    "Graph minors A survey"
]

texts = [[word for word in document.lower().split()] for document in documents]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print(f"文档数: {len(documents)}")
print(f"词汇表大小: {len(dictionary)}")

print("\n=== TF-IDF模型 ===")
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

print("TF-IDF向量 (第一个文档):")
for doc in corpus_tfidf:
    print(f"  {doc[:5]}")
    break

print("\n=== LDA主题模型 ===")
lda = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

print("LDA主题:")
for idx, topic in lda.print_topics():
    print(f"  主题{idx}: {topic}")

print("\n=== LDA主题一致性 ===")
coherence_model = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()
print(f"主题一致性分数: {coherence_score:.4f}")

print("\n=== LSI模型 ===")
lsi = LsiModel(corpus, num_topics=3, id2word=dictionary)

print("LSI主题:")
for idx, topic in lsi.print_topics():
    print(f"  主题{idx}: {topic}")

print("\n=== HDP模型 ===")
hdp = HdpModel(corpus, id2word=dictionary)

print("HDP主题数:", hdp.m_T)
print("HDP主题:")
for idx, topic in hdp.print_topics()[:3]:
    print(f"  主题{idx}: {topic[:100]}...")

print("\n=== 文档相似度查询 ===")
index = similarities.MatrixSimilarity(lda[corpus])

query = "human computer interaction"
query_bow = dictionary.doc2bow(query.lower().split())
query_lda = lda[query_bow]

sims = index[query_lda]
sims = sorted(enumerate(sims), key=lambda item: -item[1])

print(f"查询: '{query}'")
print("相似文档:")
for doc_id, similarity in sims[:3]:
    print(f"  文档{doc_id}: {similarity:.4f} - '{documents[doc_id][:50]}...'")

print("\n=== 主题分布可视化 ===")
doc_topics = lda.get_document_topics(corpus[0])
print(f"第一个文档的主题分布: {doc_topics}")

print("\n=== 新文档主题推断 ===")
new_doc = "computer system interface"
new_bow = dictionary.doc2bow(new_doc.lower().split())
new_topics = lda[new_bow]
print(f"新文档 '{new_doc}' 的主题分布: {new_topics}")

print("\n=== 主题词提取 ===")
for topic_id in range(lda.num_topics):
    top_words = lda.show_topic(topic_id, topn=5)
    words = [word for word, prob in top_words]
    print(f"主题{topic_id}关键词: {words}")

print("\n=== 模型保存和加载 ===")
lda.save("lda_model.model")
loaded_lda = LdaModel.load("lda_model.model")
print("LDA模型已保存和加载")

print("\n=== 清理 ===")
import os
if os.path.exists("lda_model.model"):
    os.remove("lda_model.model")
    print("模型文件已删除")