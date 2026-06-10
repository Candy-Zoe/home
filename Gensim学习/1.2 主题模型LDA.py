# Gensim主题模型LDA学习
# 主要内容：LDA模型训练、主题推断、可视化

import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.test.utils import common_texts
import matplotlib.pyplot as plt

print("=== 准备数据 ===")
texts = common_texts
print(f"文本数量: {len(texts)}")
print(f"示例文本: {texts[0]}")

print("\n=== 创建词典 ===")
dictionary = corpora.Dictionary(texts)
print(f"词典大小: {len(dictionary)}")
print(f"词典示例: {list(dictionary.items())[:5]}")

print("\n=== 创建语料库 ===")
corpus = [dictionary.doc2bow(text) for text in texts]
print(f"语料库大小: {len(corpus)}")
print(f"第一个文档的词袋表示: {corpus[0]}")

print("\n=== 训练LDA模型 ===")
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, random_state=42)
print("LDA模型训练完成")

print("\n=== 查看主题 ===")
topics = lda_model.print_topics(num_words=5)
print("各主题的关键词:")
for topic in topics:
    print(f"主题 {topic[0]}: {topic[1]}")

print("\n=== 推断文档主题 ===")
doc = ["human", "interface", "computer"]
bow = dictionary.doc2bow(doc)
topic_distribution = lda_model.get_document_topics(bow)
print(f"文档 '{doc}' 的主题分布:")
for topic_id, prob in topic_distribution:
    print(f"  主题 {topic_id}: {prob:.4f}")

print("\n=== 主题一致性 ===")
coherence_model = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence = coherence_model.get_coherence()
print(f"主题一致性得分: {coherence:.4f}")

print("\n=== 可视化主题 ===")
try:
    import pyLDAvis.gensim_models
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.show(vis)
except ImportError:
    print("pyLDAvis未安装，跳过可视化")