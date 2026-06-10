# jieba文本分析学习
# 主要内容：文本分词、词性标注、关键词提取、文本相似度

import jieba
import jieba.posseg as pseg
import jieba.analyse

print("=== 加载自定义词典 ===")
jieba.load_userdict('user_dict.txt')

print("\n=== 精确模式分词 ===")
text = "我爱自然语言处理和机器学习"
words = jieba.cut(text, cut_all=False)
print(f"精确模式: {', '.join(words)}")

print("\n=== 全模式分词 ===")
words = jieba.cut(text, cut_all=True)
print(f"全模式: {', '.join(words)}")

print("\n=== 搜索引擎模式 ===")
words = jieba.cut_for_search(text)
print(f"搜索引擎模式: {', '.join(words)}")

print("\n=== 词性标注 ===")
words = pseg.cut(text)
result = [(word, flag) for word, flag in words]
print(f"词性标注: {result}")

print("\n=== 关键词提取(TF-IDF) ===")
text = """自然语言处理是人工智能领域的一个重要方向，它研究如何让计算机理解和处理人类语言。
深度学习在自然语言处理中取得了巨大成功，特别是Transformer架构的出现。"""
keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
print("TF-IDF关键词:")
for word, weight in keywords:
    print(f"  {word}: {weight:.4f}")

print("\n=== 关键词提取(TextRank) ===")
keywords = jieba.analyse.textrank(text, topK=5, withWeight=True)
print("TextRank关键词:")
for word, weight in keywords:
    print(f"  {word}: {weight:.4f}")

print("\n=== 自定义停用词 ===")
stopwords = ['是', '的', '在', '了', '和']
words = jieba.cut(text)
filtered_words = [word for word in words if word not in stopwords]
print(f"过滤停用词后: {' '.join(filtered_words[:20])}...")

print("\n=== 文本相似度 ===")
from difflib import SequenceMatcher

text1 = "自然语言处理"
text2 = "自然语言理解"
similarity = SequenceMatcher(None, text1, text2).ratio()
print(f"文本相似度: {similarity:.4f}")

print("\n=== 创建用户词典文件 ===")
with open('user_dict.txt', 'w', encoding='utf-8') as f:
    f.write("自然语言处理 n\n机器学习 n\n深度学习 n\nTransformer n\n")
print("用户词典已创建")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('user_dict.txt'):
    os.remove('user_dict.txt')
    print("已删除用户词典")