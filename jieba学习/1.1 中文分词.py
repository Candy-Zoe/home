# jieba中文分词学习
# 主要内容：基本分词、自定义词典、词性标注

import jieba

print("=== 基本分词 ===")
text = "我爱自然语言处理"
result = jieba.cut(text)
print(f"精确模式: {' '.join(result)}")

result = jieba.cut(text, cut_all=True)
print(f"全模式: {' '.join(result)}")

result = jieba.cut_for_search(text)
print(f"搜索引擎模式: {' '.join(result)}")

print("\n=== 自定义词典 ===")
jieba.add_word('自然语言处理')
result = jieba.cut(text)
print(f"添加新词后: {' '.join(result)}")

print("\n=== 关键词提取 ===")
text = "机器学习是人工智能的核心技术之一，深度学习是机器学习的一个分支。"
keywords = jieba.analyse.extract_tags(text, topK=5)
print(f"关键词: {keywords}")

print("\n=== 词性标注 ===")
import jieba.posseg as pseg
words = pseg.cut(text)
for word, flag in words:
    print(f"{word}/{flag}")

print("\n=== 并行分词 ===")
jieba.enable_parallel(4)
result = jieba.cut(text)
print(f"并行分词结果: {' '.join(result)}")
jieba.disable_parallel()