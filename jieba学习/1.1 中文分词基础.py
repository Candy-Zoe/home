# jieba中文分词基础学习
# 主要内容：分词模式、自定义词典、关键词提取

# 导入jieba库
import jieba
import jieba.analyse

# 基本分词功能
print("=== 基本分词 ===")

# 待分词文本
text = "我爱自然语言处理，这是一个非常有趣的领域。"

# 默认分词（精确模式）
seg_list = jieba.cut(text, cut_all=False)
print(f"精确模式: {'/'.join(seg_list)}")

# 全模式分词
seg_list = jieba.cut(text, cut_all=True)
print(f"全模式: {'/'.join(seg_list)}")

# 搜索引擎模式
seg_list = jieba.cut_for_search(text)
print(f"搜索引擎模式: {'/'.join(seg_list)}")

# 词性标注
print("\n=== 词性标注 ===")

import jieba.posseg as pseg

words = pseg.cut(text)
for word, flag in words:
    print(f"{word}({flag})", end=" ")
print()

# 添加自定义词典
print("\n=== 自定义词典 ===")

# 定义自定义词典内容
custom_dict = """
自然语言处理 n
机器学习 n
深度学习 n
人工智能 n
"""

# 临时添加自定义词
jieba.add_word("自然语言处理", freq=10, tag="n")
jieba.add_word("机器学习", freq=10, tag="n")

# 再次分词
text2 = "机器学习和深度学习是人工智能的核心技术。"
seg_list = jieba.cut(text2)
print(f"添加自定义词后分词: {'/'.join(seg_list)}")

# 加载自定义词典文件
# jieba.load_userdict("custom_dict.txt")

# 关键词提取
print("\n=== 关键词提取 ===")

# 待提取关键词的文本
article = """
自然语言处理是计算机科学与人工智能领域的一个重要方向。
它研究如何使计算机能够理解、处理和生成人类语言。
机器学习和深度学习在自然语言处理中发挥着重要作用。
"""

# 使用TF-IDF提取关键词
keywords_tfidf = jieba.analyse.extract_tags(article, topK=5, withWeight=True)
print("TF-IDF关键词:")
for keyword, weight in keywords_tfidf:
    print(f"  {keyword}: {weight:.4f}")

# 使用TextRank提取关键词
keywords_textrank = jieba.analyse.textrank(article, topK=5, withWeight=True)
print("\nTextRank关键词:")
for keyword, weight in keywords_textrank:
    print(f"  {keyword}: {weight:.4f}")

# 并行分词
print("\n=== 并行分词 ===")

# 开启并行分词
jieba.enable_parallel(4)

# 大量文本分词
long_text = "自然语言处理" * 1000
seg_list = jieba.cut(long_text)
result = "/".join(seg_list)
print(f"并行分词完成，结果长度: {len(result)}")

# 关闭并行分词
jieba.disable_parallel()

# 调整分词粒度
print("\n=== 调整分词粒度 ===")

# 设置HMM模型
text3 = "他来到了网易杭研大厦"
seg_list = jieba.cut(text3, HMM=True)
print(f"HMM开启: {'/'.join(seg_list)}")

seg_list = jieba.cut(text3, HMM=False)
print(f"HMM关闭: {'/'.join(seg_list)}")

# 示例：完整的分词流程
print("\n=== 完整示例 ===")

# 原始文本
original_text = "深度学习正在改变人工智能的发展方向，自然语言处理是其中的重要应用领域。"

print(f"原始文本: {original_text}")

# 分词
seg_result = jieba.cut(original_text)
seg_str = "/".join(seg_result)
print(f"分词结果: {seg_str}")

# 词性标注
words = pseg.cut(original_text)
pos_result = " ".join([f"{w}/{f}" for w, f in words])
print(f"词性标注: {pos_result}")

# 关键词提取
keywords = jieba.analyse.extract_tags(original_text, topK=3)
print(f"关键词: {keywords}")