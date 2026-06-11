# NLTK自然语言处理学习
# 主要内容：分词、词性标注、命名实体识别、情感分析、文本分类

# 导入必要的库
import nltk
import numpy as np
import matplotlib.pyplot as plt

# 下载必要的NLTK数据
print("=== 下载NLTK数据 ===")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('maxent_ne_chunker_tab', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK数据下载完成")
except Exception as e:
    print(f"下载过程中出现问题: {e}")

# 文本分词
print("\n=== 文本分词 ===")

# 英文分词
text_en = "Natural language processing is a fascinating field of artificial intelligence. It helps computers understand human language."

# 句子分词
sentences = nltk.sent_tokenize(text_en)
print(f"句子分词结果 ({len(sentences)} 个句子):")
for i, sent in enumerate(sentences, 1):
    print(f"  句子 {i}: {sent}")

# 单词分词
words = nltk.word_tokenize(text_en)
print(f"\n单词分词结果 ({len(words)} 个词):")
print(f"  {words}")

# 中文分词（需要额外的库支持）
print("\n中文分词:")
text_cn = "自然语言处理是人工智能的重要分支。它让计算机能够理解和使用人类语言。"
print(f"  原文: {text_cn}")
print("  提示: 中文分词建议使用jieba库")

# 词性标注
print("\n=== 词性标注 ===")

text = "The quick brown fox jumps over the lazy dog."
words = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(words)

print(f"词性标注结果:")
for word, pos in pos_tags:
    print(f"  {word}: {pos}")

# 解释常见词性
print("\n常见词性标签:")
pos_meanings = {
    'NN': '名词(单数)',
    'NNS': '名词(复数)',
    'VB': '动词(原形)',
    'VBD': '动词(过去式)',
    'VBG': '动词(动名词)',
    'JJ': '形容词',
    'RB': '副词',
    'DT': '限定词',
    'IN': '介词',
    'CC': '连词'
}
for tag, meaning in pos_meanings.items():
    print(f"  {tag}: {meaning}")

# 命名实体识别
print("\n=== 命名实体识别 ===")

text_ner = "Apple Inc. is planning to open a new store in New York City. Tim Cook announced this on Monday."
words = nltk.word_tokenize(text_ner)
pos_tags = nltk.pos_tag(words)

# 命名实体识别
ne_chunks = nltk.ne_chunk(pos_tags)

print(f"命名实体识别结果:")
for chunk in ne_chunks:
    if hasattr(chunk, 'label'):
        entity = ' '.join(c[0] for c in chunk)
        entity_type = chunk.label()
        print(f"  {entity}: {entity_type}")

# 解释常见实体类型
print("\n常见实体类型:")
ne_meanings = {
    'ORGANIZATION': '组织机构',
    'GPE': '地理政治实体',
    'PERSON': '人名',
    'LOCATION': '位置',
    'DATE': '日期',
    'TIME': '时间',
    'MONEY': '金额',
    'PERCENT': '百分比'
}
for ne, meaning in ne_meanings.items():
    print(f"  {ne}: {meaning}")

# 情感分析
print("\n=== 情感分析 ===")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 创建情感分析器
sia = SentimentIntensityAnalyzer()

# 测试文本
test_texts = [
    "I love this product! It's amazing!",
    "This is the worst experience ever.",
    "It's okay, nothing special.",
    "I'm so happy with the service!",
    "Terrible, will never buy again."
]

print("情感分析结果:")
for text in test_texts:
    scores = sia.polarity_scores(text)
    print(f"\n  文本: {text}")
    print(f"  正向: {scores['pos']:.3f}")
    print(f"  中性: {scores['neu']:.3f}")
    print(f"  负向: {scores['neg']:.3f}")
    print(f"  复合: {scores['compound']:.3f}")
    
    # 判断情感倾向
    if scores['compound'] >= 0.05:
        sentiment = "正向"
    elif scores['compound'] <= -0.05:
        sentiment = "负向"
    else:
        sentiment = "中性"
    print(f"  情感: {sentiment}")

# 停用词
print("\n=== 停用词 ===")

from nltk.corpus import stopwords

# 英文停用词
stop_words_en = stopwords.words('english')
print(f"英文停用词数量: {len(stop_words_en)}")
print(f"前20个停用词: {stop_words_en[:20]}")

# 移除停用词
text = "This is a sample sentence showing how to remove stop words from text."
words = nltk.word_tokenize(text)
filtered_words = [word for word in words if word.lower() not in stop_words_en]

print(f"\n原始词数: {len(words)}")
print(f"移除停用词后: {len(filtered_words)}")
print(f"原始: {words}")
print(f"过滤后: {filtered_words}")

# 词形归一化
print("\n=== 词形归一化 ===")

from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

# 创建词形归一化工具
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# 测试词
test_words = ['running', 'ran', 'runs', 'easily', 'fairly', 'better', 'best']

print("词形归一化对比:")
print(f"{'原词':<12} {'Porter':<12} {'Lancaster':<12} {'Lemma':<12}")
print("-" * 50)
for word in test_words:
    porter_stem = porter.stem(word)
    lancaster_stem = lancaster.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')
    print(f"{word:<12} {porter_stem:<12} {lancaster_stem:<12} {lemma:<12}")

# 词性标注的词形归一化
print("\n词形归一化（考虑词性）:")
word = 'better'
print(f"  原词: {word}")
print(f"  名词形式: {lemmatizer.lemmatize(word, pos='n')}")
print(f"  形容词形式: {lemmatizer.lemmatize(word, pos='a')}")
print(f"  副词形式: {lemmatizer.lemmatize(word, pos='r')}")
print(f"  动词形式: {lemmatizer.lemmatize(word, pos='v')}")

# 词性还原 vs 词干提取
print("\n=== 词干提取 vs 词形还原 ===")
print("词干提取: 简单的规则去除词缀，结果可能不是完整单词")
print("词形还原: 基于词汇表和词性分析，返回单词的基本形式")

# 词频统计
print("\n=== 词频统计 ===")

from nltk import FreqDist

text = """Natural language processing is a subfield of artificial intelligence. 
It focuses on the interaction between computers and human language. 
NLP techniques are used in many applications such as sentiment analysis, 
machine translation, and text classification. 
Modern NLP relies heavily on deep learning."""

# 分词
words = nltk.word_tokenize(text.lower())

# 移除标点和停用词
words_clean = [word for word in words 
               if word.isalnum() and word not in stop_words_en]

# 词频统计
fdist = FreqDist(words_clean)
print(f"词频统计（前10）:")
for word, freq in fdist.most_common(10):
    print(f"  {word}: {freq}")

# 绘制词频分布
plt.figure(figsize=(10, 5))
fdist.plot(20, cumulative=False, title='词频分布（前20）')
plt.tight_layout()
plt.show()

# 文本预处理完整流程
print("\n=== 文本预处理完整流程 ===")

def preprocess_text(text, language='english'):
    """完整的文本预处理流程"""
    
    # 1. 句子分词
    sentences = nltk.sent_tokenize(text)
    print(f"  1. 句子分词: {len(sentences)} 个句子")
    
    # 2. 单词分词
    words = nltk.word_tokenize(text)
    print(f"  2. 单词分词: {len(words)} 个词")
    
    # 3. 转为小写
    words = [word.lower() for word in words]
    print(f"  3. 小写转换完成")
    
    # 4. 移除标点
    words = [word for word in words if word.isalnum()]
    print(f"  4. 移除标点: {len(words)} 个词")
    
    # 5. 移除停用词
    if language == 'english':
        stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]
    print(f"  5. 移除停用词: {len(words)} 个词")
    
    # 6. 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    print(f"  6. 词形还原完成")
    
    return words

# 测试预处理
test_text = """Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence!
It enables computers to understand and process human language. 
The applications of NLP are endless and exciting."""

print("文本预处理流程:")
processed_words = preprocess_text(test_text)
print(f"\n处理结果: {processed_words}")

# 文本分类
print("\n=== 文本分类 ===")

from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# 创建训练数据
positive_texts = [
    'I love this movie, it is amazing',
    'Great product, highly recommended',
    'Excellent service, very satisfied',
    'Best experience ever',
    'Wonderful, fantastic',
    'Awesome, will buy again'
]

negative_texts = [
    'Terrible, worst ever',
    'I hate this product',
    'Awful experience, very disappointed',
    'Worst purchase, do not buy',
    'Bad quality, not worth it',
    'Horrible service, very upset'
]

# 特征提取
def extract_features(text):
    """提取文本特征"""
    words = set(nltk.word_tokenize(text.lower()))
    features = {}
    for word in words:
        features[word] = True
    return features

# 准备训练数据
positive_features = [(extract_features(text), 'positive') for text in positive_texts]
negative_features = [(extract_features(text), 'negative') for text in negative_texts]
labeled_features = positive_features + negative_features

# 打乱数据
np.random.shuffle(labeled_features)

# 划分训练集和测试集
split = int(len(labeled_features) * 0.8)
train_set = labeled_features[:split]
test_set = labeled_features[split:]

# 训练朴素贝叶斯分类器
print("训练朴素贝叶斯分类器...")
classifier = NaiveBayesClassifier.train(train_set)

# 评估
acc = accuracy(classifier, test_set)
print(f"测试集准确率: {acc:.2%}")

# 显示最重要的特征
print("\n最重要的特征:")
for word, score in classifier.most_informative_features(5):
    print(f"  {word}: {score:.4f}")

# 预测新文本
test_texts = [
    "This is a great product, love it!",
    "Horrible experience, very bad.",
    "It's okay, nothing special."
]

print("\n预测新文本:")
for text in test_texts:
    features = extract_features(text)
    prediction = classifier.classify(features)
    print(f"  '{text}' -> {prediction}")

# 词袋模型
print("\n=== 词袋模型 (Bag of Words) ===")

documents = [
    "I love natural language processing",
    "Natural language processing is fun",
    "Machine learning is a subset of AI",
    "Deep learning is popular"
]

# 分词
tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in documents]

# 构建词袋
all_words = []
for doc in tokenized_docs:
    all_words.extend(doc)

unique_words = list(set(all_words))
print(f"词汇表: {unique_words}")
print(f"词汇表大小: {len(unique_words)}")

# 计算词频向量
print("\n词袋向量:")
for i, doc in enumerate(tokenized_docs):
    vector = [doc.count(word) for word in unique_words]
    print(f"  文档 {i+1}: {vector}")

# 总结
print("\n=== NLTK自然语言处理学习总结 ===")
print("1. 文本分词（句子和单词）")
print("2. 词性标注")
print("3. 命名实体识别")
print("4. 情感分析（VADER）")
print("5. 停用词处理")
print("6. 词形归一化（词干提取和词形还原）")
print("7. 词频统计")
print("8. 文本预处理流程")
print("9. 文本分类（朴素贝叶斯）")
print("10. 词袋模型")

print("\nNLTK自然语言处理学习完成！")
