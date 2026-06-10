# NLTK文本分类与情感分析学习
# 主要内容：文本分类、情感分析、词频分析、文本相似度

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

print("=== 下载必要资源 ===")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('movie_reviews')

print("\n=== 情感分析 ===")
sia = SentimentIntensityAnalyzer()

texts = [
    "I love this product! It's amazing!",
    "This is terrible, I hate it.",
    "It's okay, nothing special."
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"文本: {text}")
    print(f"  情感分数: {scores}")
    print()

print("\n=== 文本分类 ===")
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

print(f"文档数量: {len(documents)}")

print("\n=== 特征提取 ===")
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

classifier = NaiveBayesClassifier.train(train_set)
print(f"分类器准确率: {nltk.classify.accuracy(classifier, test_set):.4f}")

print("\n=== 最有信息量的特征 ===")
classifier.show_most_informative_features(10)

print("\n=== 词频分析 ===")
text = "Natural language processing is a subfield of artificial intelligence. Natural language processing helps computers understand human language."
words = word_tokenize(text.lower())
fdist = nltk.FreqDist(words)

print("词频分布:")
for word, freq in fdist.most_common(10):
    print(f"  {word}: {freq}")

print("\n=== 停用词过滤 ===")
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w not in stop_words]
print(f"原始词数: {len(words)}")
print(f"过滤后词数: {len(filtered_words)}")

print("\n=== 词形还原 ===")
lemmatizer = WordNetLemmatizer()
words_to_lemmatize = ['running', 'ran', 'runs', 'better', 'best']
lemmatized = [lemmatizer.lemmatize(word) for word in words_to_lemmatize]
print(f"原始: {words_to_lemmatize}")
print(f"还原: {lemmatized}")

print("\n=== N-gram分析 ===")
from nltk import ngrams

text = "I love natural language processing"
tokens = word_tokenize(text)

bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print(f"Bigrams: {bigrams}")
print(f"Trigrams: {trigrams}")

print("\n=== 文本相似度 ===")
from nltk.corpus import wordnet

def word_similarity(word1, word2):
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    
    if synsets1 and synsets2:
        return synsets1[0].wup_similarity(synsets2[0])
    return None

sim = word_similarity('car', 'automobile')
print(f"'car' 和 'automobile' 的相似度: {sim:.4f}")

print("\n=== 文本聚类 ===")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

documents = [
    "I love programming in Python",
    "Python is a great language",
    "I enjoy playing football",
    "Football is my favorite sport",
    "Machine learning is fascinating"
]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print("聚类结果:")
for i, doc in enumerate(documents):
    print(f"  文档{i+1}: 聚类{kmeans.labels_[i]}")