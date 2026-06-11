# Gensim主题建模与词向量学习
# 主要内容：Word2Vec、Doc2Vec、TF-IDF、LDA主题建模、文本相似度

# 导入必要的库
import gensim
from gensim import corpora, models, similarities
from gensim.models import Word2Vec, Doc2Vec, KeyedVectors, LdaModel, TfidfModel
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import jieba

# Gensim基础
print("=== Gensim基础 ===")
print(f"Gensim版本: {gensim.__version__}")

# Word2Vec词向量
print("\n=== Word2Vec词向量 ===")

# 准备语料库
corpus = [
    ['机器', '学习', '是', '人工智能', '的', '分支'],
    ['深度', '学习', '是', '机器', '学习', '的', '子集'],
    ['自然', '语言', '处理', '处理', '文本', '数据'],
    ['计算机', '视觉', '识别', '图像', '和', '视频'],
    ['强化', '学习', '通过', '奖励', '训练', '智能体'],
    ['神经', '网络', '模拟', '人脑', '的', '工作', '方式'],
    ['数据', '挖掘', '从', '数据', '中', '提取', '知识'],
    ['推荐', '系统', '为', '用户', '推荐', '商品']
]

# 训练Word2Vec模型
print("训练Word2Vec模型...")
w2v_model = Word2Vec(
    sentences=corpus,
    vector_size=50,  # 词向量维度
    window=3,         # 上下文窗口大小
    min_count=1,      # 最小词频
    workers=4,        # 训练线程数
    epochs=100,       # 训练轮数
    seed=42
)

print(f"词汇表大小: {len(w2v_model.wv.key_to_index)}")
print(f"词向量维度: {w2v_model.vector_size}")

# 查看词向量
word = '学习'
if word in w2v_model.wv:
    vector = w2v_model.wv[word]
    print(f"\n'{word}'的词向量（前10维）:")
    print(vector[:10].round(4))
    print(f"向量形状: {vector.shape}")

# 相似词查询
print("\n=== 相似词查询 ===")
test_words = ['学习', '数据', '智能']
for word in test_words:
    if word in w2v_model.wv:
        similar = w2v_model.wv.most_similar(word, topn=3)
        print(f"\n与'{word}'最相似的词:")
        for sim_word, score in similar:
            print(f"  {sim_word}: {score:.4f}")

# 词向量运算（Word2Vec经典示例）
print("\n=== 词向量运算 ===")
try:
    # 学习 - 机器 + 深度 ≈ ?
    result = w2v_model.wv.most_similar(
        positive=['学习', '深度'],
        negative=['机器'],
        topn=3
    )
    print("学习 - 机器 + 深度 ≈ ?")
    for word, score in result:
        print(f"  {word}: {score:.4f}")
except Exception as e:
    print(f"词向量运算失败: {e}")

# 词向量可视化
print("\n=== 词向量可视化 ===")

# 获取所有词向量
words = list(w2v_model.wv.key_to_index.keys())
word_vectors = np.array([w2v_model.wv[word] for word in words])

# 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words)-1))
vectors_2d = tsne.fit_transform(word_vectors)

# 可视化
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.6)

# 添加词标签
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                fontsize=12, alpha=0.8)

plt.xlabel('t-SNE 维度1')
plt.ylabel('t-SNE 维度2')
plt.title('Word2Vec 词向量可视化')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# TF-IDF模型
print("\n=== TF-IDF模型 ===")

# 准备文档集合
documents = [
    "机器学习是人工智能的重要分支",
    "深度学习是机器学习的子集",
    "自然语言处理帮助计算机理解文本",
    "计算机视觉处理图像和视频",
    "强化学习通过奖励训练智能体",
    "神经网络模拟人脑的工作方式",
    "数据挖掘从数据中提取知识",
    "推荐系统为用户推荐商品"
]

# 分词
tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
print("分词结果:")
for i, tokens in enumerate(tokenized_docs, 1):
    print(f"  文档 {i}: {tokens}")

# 构建词典
dictionary = corpora.Dictionary(tokenized_docs)
print(f"\n词典大小: {len(dictionary)}")
print(f"词典内容: {list(dictionary.items())[:5]}")

# 转换为词袋表示
bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
print(f"\n词袋表示（第一个文档）:")
for word_id, count in bow_corpus[0][:5]:
    word = dictionary[word_id]
    print(f"  {word}: {count}")

# 训练TF-IDF模型
tfidf_model = TfidfModel(bow_corpus)
tfidf_corpus = tfidf_model[bow_corpus]

print(f"\nTF-IDF表示（第一个文档）:")
for word_id, tfidf_score in list(tfidf_corpus[0])[:5]:
    word = dictionary[word_id]
    print(f"  {word}: {tfidf_score:.4f}")

# 文档相似度
print("\n=== 文档相似度 ===")

# 构建相似度索引
index = similarities.SparseMatrixSimilarity(tfidf_corpus, num_features=len(dictionary))

# 查询文档
query = "我想学习机器学习和深度学习"
query_tokens = list(jieba.cut(query))
query_bow = dictionary.doc2bow(query_tokens)
query_tfidf = tfidf_model[query_bow]

# 计算相似度
similarities_scores = index[query_tfidf]
print(f"查询: {query}")
print("\n文档相似度排名:")
sorted_idx = np.argsort(similarities_scores)[::-1]
for rank, idx in enumerate(sorted_idx, 1):
    score = similarities_scores[idx]
    print(f"  {rank}. 相似度: {score:.4f} - {documents[idx]}")

# 可视化相似度
plt.figure(figsize=(10, 6))
plt.bar(range(len(similarities_scores)), similarities_scores[sorted_idx])
plt.xticks(range(len(similarities_scores)), 
           [f'文档{i+1}' for i in sorted_idx], rotation=45)
plt.ylabel('相似度')
plt.title('文档相似度排名')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# LDA主题建模
print("\n=== LDA主题建模 ===")

# 准备更多文档
large_documents = [
    "机器学习算法包括监督学习无监督学习和强化学习",
    "深度学习使用神经网络进行特征学习",
    "自然语言处理是人工智能的重要应用领域",
    "计算机视觉技术广泛应用于图像识别",
    "数据挖掘和数据分析帮助企业决策",
    "推荐系统根据用户偏好推荐内容",
    "强化学习通过与环境交互学习策略",
    "神经网络包括CNN RNN和Transformer",
    "图像分类目标检测是视觉任务",
    "文本分类情感分析是NLP任务",
    "协同过滤是推荐系统的经典方法",
    "卷积神经网络擅长处理图像数据",
    "循环神经网络适合处理序列数据",
    "生成对抗网络用于生成新数据",
    "迁移学习利用预训练模型解决新任务"
]

# 分词
large_tokenized = [list(jieba.cut(doc)) for doc in large_documents]

# 构建词典
large_dict = corpora.Dictionary(large_tokenized)
print(f"词典大小: {len(large_dict)}")

# 词袋表示
large_bow = [large_dict.doc2bow(doc) for doc in large_tokenized]

# 训练LDA模型
num_topics = 3
lda_model = LdaModel(
    corpus=large_bow,
    id2word=large_dict,
    num_topics=num_topics,
    random_state=42,
    passes=20,
    alpha='auto',
    eta='auto'
)

# 打印主题
print(f"\nLDA主题模型（{num_topics}个主题）:")
for topic_id in range(num_topics):
    print(f"\n主题 {topic_id + 1}:")
    top_words = lda_model.show_topic(topic_id, topn=5)
    for word, prob in top_words:
        print(f"  {word}: {prob:.4f}")

# 文档-主题分布
print("\n=== 文档-主题分布 ===")
for i, doc_bow in enumerate(large_bow[:5], 1):
    topic_dist = lda_model.get_document_topics(doc_bow)
    print(f"\n文档 {i}: {large_documents[i-1]}")
    print(f"  主题分布:")
    for topic_id, prob in topic_dist:
        print(f"    主题{topic_id+1}: {prob:.4f}")

# 主题可视化
print("\n=== 主题词分布可视化 ===")
fig, axes = plt.subplots(1, num_topics, figsize=(15, 4))

for topic_id in range(num_topics):
    top_words = lda_model.show_topic(topic_id, topn=8)
    words = [w[0] for w in top_words]
    probs = [w[1] for w in top_words]
    
    axes[topic_id].barh(range(len(words)), probs, color='steelblue')
    axes[topic_id].set_yticks(range(len(words)))
    axes[topic_id].set_yticklabels(words)
    axes[topic_id].set_xlabel('概率')
    axes[topic_id].set_title(f'主题 {topic_id+1}')
    axes[topic_id].invert_yaxis()

plt.tight_layout()
plt.show()

# Doc2Vec文档向量
print("\n=== Doc2Vec文档向量 ===")

# 准备文档
tagged_docs = [TaggedDocument(words=tokens, tags=[f'doc_{i}']) 
               for i, tokens in enumerate(large_tokenized)]

# 训练Doc2Vec
print("训练Doc2Vec模型...")
d2v_model = Doc2Vec(
    documents=tagged_docs,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4,
    epochs=100,
    seed=42
)

# 获取文档向量
doc_vectors = np.array([d2v_model.dv[f'doc_{i}'] for i in range(len(large_documents))])
print(f"文档向量矩阵形状: {doc_vectors.shape}")

# 文档相似度
print("\nDoc2Vec文档相似度:")
similarity_matrix = np.zeros((len(doc_vectors), len(doc_vectors)))
for i in range(len(doc_vectors)):
    for j in range(len(doc_vectors)):
        similarity_matrix[i][j] = np.dot(doc_vectors[i], doc_vectors[j]) / (
            np.linalg.norm(doc_vectors[i]) * np.linalg.norm(doc_vectors[j])
        )

# 显示前5个文档的相似度
print("\n前5个文档的相似度矩阵:")
for i in range(5):
    for j in range(5):
        print(f"{similarity_matrix[i][j]:.3f}", end="  ")
    print()

# 推断新文档的向量
print("\n推断新文档向量:")
new_doc = "我想学习机器学习和深度学习算法"
new_tokens = list(jieba.cut(new_doc))
new_vector = d2v_model.infer_vector(new_tokens, epochs=50)
print(f"新文档: {new_doc}")
print(f"分词: {new_tokens}")
print(f"向量形状: {new_vector.shape}")

# 找到最相似的文档
similarities_to_new = []
for i, doc_vec in enumerate(doc_vectors):
    sim = np.dot(new_vector, doc_vec) / (
        np.linalg.norm(new_vector) * np.linalg.norm(doc_vec)
    )
    similarities_to_new.append((i, sim))

similarities_to_new.sort(key=lambda x: x[1], reverse=True)
print("\n最相似的3个文档:")
for idx, sim in similarities_to_new[:3]:
    print(f"  相似度: {sim:.4f} - {large_documents[idx]}")

# 加载预训练词向量
print("\n=== 加载预训练词向量 ===")
print("提示: 可以使用Google News预训练词向量")
print("下载命令: wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz")
print("加载示例: KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)")

# 实际加载示例（如果有文件）
import os
if os.path.exists('GoogleNews-vectors-negative300.bin'):
    print("加载预训练词向量...")
    pretrained = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print(f"预训练词向量数量: {len(pretrained.key_to_index)}")
    print(f"向量维度: {pretrained.vector_size}")
else:
    print("未找到预训练词向量文件，跳过此示例")

# 文本摘要
print("\n=== 文本摘要 (TextRank) ===")
print("提示: gensim也支持TextRank算法进行文本摘要")
print("使用: from gensim.summarization import summarize")

# 关键词提取
print("\n=== 关键词提取 ===")
print("使用TF-IDF提取关键词:")

# 合并所有文档作为参考语料
all_text = ' '.join(large_documents)
all_tokens = list(jieba.cut(all_text))

# 计算TF-IDF
from collections import Counter
word_freq = Counter(all_tokens)

# 显示词频最高的词
print("\n词频最高的10个词:")
for word, freq in word_freq.most_common(10):
    print(f"  {word}: {freq}")

# 性能优化
print("\n=== 性能优化 ===")
print("1. 使用批量训练 (batch_words)")
print("2. 调整window和min_count参数")
print("3. 使用多线程训练 (workers)")
print("4. 增加训练轮数 (epochs)")
print("5. 使用负采样 (negative)")
print("6. 使用层次softmax (hs)")

# 总结
print("\n=== Gensim学习总结 ===")
print("1. Word2Vec词向量训练和使用")
print("2. 词向量相似度查询和向量运算")
print("3. TF-IDF模型和文档相似度")
print("4. LDA主题建模")
print("5. Doc2Vec文档向量")
print("6. 文本相似度计算")
print("7. 关键词提取")
print("8. 文本摘要")
print("9. 预训练词向量加载")
print("10. 性能优化技巧")

print("\nGensim主题建模与词向量学习完成！")
