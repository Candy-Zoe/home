# NLTK文本处理基础学习
# 主要内容：分词、词性标注、停用词处理

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

print("=== 下载NLTK数据 ===")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

print("\n=== 句子分割 ===")
text = "Hello! This is NLTK. It is a great tool for natural language processing."
sentences = sent_tokenize(text)
print(f"原始文本: {text}")
print(f"句子分割: {sentences}")

print("\n=== 分词 ===")
words = word_tokenize(text)
print(f"分词结果: {words}")

print("\n=== 词性标注 ===")
tagged = pos_tag(words)
print(f"词性标注:")
for word, tag in tagged:
    print(f"  {word}/{tag}")

print("\n=== 停用词处理 ===")
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print(f"停用词集合(部分): {list(stop_words)[:10]}")
print(f"过滤后词: {filtered_words}")

print("\n=== 词干提取 ===")
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words = ["running", "ran", "runner", "runs"]
stemmed = [stemmer.stem(word) for word in words]
print(f"原始词: {words}")
print(f"词干提取: {stemmed}")

print("\n=== 词形还原 ===")
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
words = ["running", "ran", "better", "best"]
lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in words]
print(f"原始词: {words}")
print(f"词形还原: {lemmatized}")