# jieba文本处理应用学习
# 主要内容：文本统计、词云生成、情感分析基础

import jieba
from collections import Counter

print("=== 文本词频统计 ===")
text = """深度学习是机器学习的一个分支，它利用多层神经网络来模拟人脑的学习过程。
深度学习在图像识别、语音识别、自然语言处理等领域取得了巨大成功。
PyTorch和TensorFlow是目前最流行的深度学习框架。"""

words = jieba.lcut(text)
word_counts = Counter(words)

print(f"总词数: {len(words)}")
print(f"不同词数: {len(word_counts)}")
print(f"Top 10词频:")
for word, count in word_counts.most_common(10):
    print(f"  {word}: {count}")

print("\n=== 停用词过滤 ===")
stopwords = ['的', '是', '在', '等', '了', '和', '是', '它']
filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
filtered_counts = Counter(filtered_words)

print(f"过滤后Top 10词频:")
for word, count in filtered_counts.most_common(10):
    print(f"  {word}: {count}")

print("\n=== 词云生成 ===")
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(
    font_path='simhei.ttf',
    width=800,
    height=600,
    background_color='white'
).generate_from_frequencies(filtered_counts)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()