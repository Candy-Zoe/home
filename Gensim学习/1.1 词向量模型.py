# Gensim词向量模型学习
# 主要内容：Word2Vec训练、词向量操作、文本相似度计算

# 导入Gensim库
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import numpy as np

# 准备训练数据
print("=== 准备训练数据 ===")

# 示例语料库
sentences = [
    # 自然语言处理相关
    ['自然语言处理', '是', '人工智能', '的', '重要', '应用', '领域'],
    ['机器学习', '和', '深度学习', '是', '实现', '自然语言处理', '的', '关键', '技术'],
    ['词向量', '是', '自然语言处理', '的', '重要', '表示', '方法'],
    ['Word2Vec', '是', '常用', '的', '词向量', '模型'],

    # 计算机视觉相关
    ['计算机视觉', '是', '人工智能', '的', '另一个', '重要', '领域'],
    ['卷积神经网络', 'CNN', '在', '图像识别', '中', '表现', '出色'],
    ['目标检测', '是', '计算机视觉', '的', '核心', '任务'],
    ['图像分割', '和', '目标检测', '是', '计算机视觉', '的', '重要', '应用'],

    # 通用人工智能
    ['人工智能', '正在', '改变', '我们', '的', '生活'],
    ['深度学习', '在', '各个', '领域', '都', '有', '广泛', '应用'],
    ['强化学习', '是', '机器学习', '的', '一个', '重要', '分支'],
    ['迁移学习', '可以', '利用', '已有', '知识', '解决', '新', '问题'],

    # 数据科学
    ['数据科学', '结合', '统计学', '和', '计算机科学'],
    ['大数据', '技术', '为', '人工智能', '提供', '了', '数据', '支持'],
    ['数据可视化', '是', '理解', '数据', '的', '重要', '手段']
]

print(f"语料库句子数量: {len(sentences)}")
print(f"总词汇量: {len(set(word for sent in sentences for word in sent))}")

# 训练Word2Vec模型
print("\n=== 训练Word2Vec模型 ===")

# 创建Word2Vec模型
# 参数说明:
# sentences: 训练语料
# vector_size: 词向量维度
# window: 上下文窗口大小
# min_count: 最小词频阈值
# workers: 并行线程数
# epochs: 训练轮数
# sg: 训练算法 (0=CBOW, 1=Skip-gram)

model = Word2Vec(
    sentences=sentences,
    vector_size=50,      # 词向量维度
    window=3,            # 上下文窗口大小
    min_count=1,         # 最小词频
    workers=4,           # 并行线程数
    epochs=100,          # 训练轮数
    sg=1                 # 使用Skip-gram算法
)

print("Word2Vec模型训练完成")
print(f"词向量维度: {model.wv.vector_size}")
print(f"词汇表大小: {len(model.wv)}")

# 保存和加载模型
print("\n=== 保存和加载模型 ===")

# 保存模型
model.save('word2vec_model.model')
print("模型已保存为 word2vec_model.model")

# 保存词向量（为单独文件）
model.wv.save('word_vectors.kv')
print("词向量已保存为 word_vectors.kv")

# 加载模型
loaded_model = Word2Vec.load('word2vec_model.model')
print("模型已加载")

# 加载词向量
loaded_vectors = KeyedVectors.load('word_vectors.kv')
print("词向量已加载")

# 词向量操作
print("\n=== 词向量操作 ===")

# 获取单个词的向量
word = '人工智能'
if word in model.wv:
    vector = model.wv[word]
    print(f"'{word}'的词向量形状: {vector.shape}")
    print(f"词向量前5个值: {vector[:5]}")
else:
    print(f"'{word}'不在词汇表中")

# 查找相似词
print("\n查找相似词:")

# 查找与"人工智能"相似的词
word_to_find = '人工智能'
similar_words = model.wv.most_similar(word_to_find, topn=5)
print(f"与'{word_to_find}'最相似的词:")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# 查找多个词的平均相似词
positive_words = ['机器学习', '深度学习']
negative_words = ['计算机视觉']

similar = model.wv.most_similar(positive=positive_words, negative=negative_words, topn=3)
print(f"\n与{positive_words}相似但与{negative_words}不相似的词:")
for word, score in similar:
    print(f"  {word}: {score:.4f}")

# 词向量相似度
print("\n=== 计算词向量相似度 ===")

pairs = [
    ('机器学习', '深度学习'),
    ('机器学习', '计算机视觉'),
    ('人工智能', '自然语言处理')
]

for word1, word2 in pairs:
    if word1 in model.wv and word2 in model.wv:
        similarity = model.wv.similarity(word1, word2)
        print(f"'{word1}'和'{word2}'的相似度: {similarity:.4f}")

# 找出不匹配的词
print("\n=== 找出不匹配的词 ===")

word_sets = [
    ['机器学习', '深度学习', '计算机视觉', '人工智能'],
    ['机器学习', '监督学习', '无监督学习', '大数据'],
    ['自然语言处理', '计算机视觉', '语音识别', '大数据']
]

for words in word_sets:
    try:
        odd = model.wv.doesnt_match(words)
        print(f"不匹配的词: {odd} (从 {words})")
    except Exception as e:
        print(f"无法处理: {words}")

# 词向量的算术运算
print("\n=== 词向量算术运算 ===")

# 经典例子: 国王 - 男人 + 女人 ≈ 女王
try:
    result = model.wv.most_similar(
        positive=['机器学习', '深度学习'],
        negative=['人工智能'],
        topn=3
    )
    print("机器学习 + 深度学习 - 人工智能 ≈")
    for word, score in result:
        print(f"  {word}: {score:.4f}")
except:
    print("由于词汇量限制，无法进行此运算")

# 可视化词向量
print("\n=== 词向量可视化 ===")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 获取所有词的向量
words = list(model.wv.key_to_index.keys())
vectors = [model.wv[word] for word in words]

# 使用PCA降维到2D
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# 绘制词向量
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

# 添加词标签
for i, word in enumerate(words):
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))

plt.title('词向量可视化 (PCA降维)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# 训练参数对比
print("\n=== 不同训练参数对比 ===")

# CBOW vs Skip-gram
print("CBOW vs Skip-gram对比:")

# CBOW模型
model_cbow = Word2Vec(sentences=sentences, vector_size=50, window=3, 
                       min_count=1, sg=0, epochs=100)

# Skip-gram模型
model_sg = Word2Vec(sentences=sentences, vector_size=50, window=3, 
                     min_count=1, sg=1, epochs=100)

print(f"CBOW词汇表大小: {len(model_cbow.wv)}")
print(f"Skip-gram词汇表大小: {len(model_sg.wv)}")

# 对比相似词结果
target = '机器学习'
print(f"\n与'{target}'最相似的词:")
print(f"CBOW: {model_cbow.wv.most_similar(target, topn=3)}")
print(f"Skip-gram: {model_sg.wv.most_similar(target, topn=3)}")

# 清理文件
import os
for f in ['word2vec_model.model', 'word_vectors.kv']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除: {f}")

print("\nGensim词向量学习完成！")