# NLTK文本分类学习
# 主要内容：使用NLTK进行情感分析和文本分类

import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

print("=== 下载电影评论数据集 ===")
nltk.download('movie_reviews')

print("\n=== 加载数据集 ===")
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

print(f"文档总数: {len(documents)}")
print(f"正例数: {len([d for d in documents if d[1] == 'pos'])}")
print(f"负例数: {len([d for d in documents if d[1] == 'neg'])}")

print("\n=== 提取特征 ===")
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]

print("\n=== 划分训练集和测试集 ===")
train_set, test_set = featuresets[100:], featuresets[:100]

print("\n=== 训练朴素贝叶斯分类器 ===")
classifier = NaiveBayesClassifier.train(train_set)

print("\n=== 评估模型 ===")
print(f"准确率: {nltk_accuracy(classifier, test_set):.4f}")

print("\n=== 查看最有信息量的特征 ===")
classifier.show_most_informative_features(10)

print("\n=== 测试新文本 ===")
test_text = "This movie was amazing! The acting was brilliant."
test_features = document_features(test_text.split())
result = classifier.classify(test_features)
print(f"文本: {test_text}")
print(f"分类结果: {result}")