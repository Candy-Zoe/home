# spaCy工业级NLP学习
# 主要内容：分词、词性标注、命名实体、依存分析、词向量、文本相似度

# 导入必要的库
import spacy
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 加载spaCy模型
print("=== 加载spaCy模型 ===")
try:
    nlp = spacy.load("en_core_web_sm")
    print("en_core_web_sm 模型加载成功")
except OSError:
    print("未找到英文模型，尝试使用空白模型")
    nlp = spacy.blank("en")
    print("使用空白英文模型")

# 基本NLP处理
print("\n=== 基本NLP处理 ===")

text = "Apple Inc. is planning to open a new store in New York City. Tim Cook announced this exciting news on Monday."

# 处理文本
doc = nlp(text)

# 打印处理结果
print(f"原始文本: {text}")
print(f"\n分词和词性标注:")
print(f"{'Token':<15} {'POS':<8} {'Tag':<8} {'Dep':<10} {'Lemma':<15}")
print("-" * 60)
for token in doc:
    print(f"{token.text:<15} {token.pos_:<8} {token.tag_:<8} {token.dep_:<10} {token.lemma_:<15}")

# 分句
print("\n句子分割:")
for i, sent in enumerate(doc.sents, 1):
    print(f"  句子 {i}: {sent.text}")

# 命名实体识别
print("\n=== 命名实体识别 (NER) ===")

print("识别出的实体:")
for ent in doc.ents:
    print(f"  {ent.text:<20} {ent.label_:<15} ({spacy.explain(ent.label_)})")

# 解释常见实体类型
print("\n常见实体类型:")
entity_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'PRODUCT']
for ent_type in entity_types:
    try:
        explanation = spacy.explain(ent_type)
        print(f"  {ent_type}: {explanation}")
    except:
        pass

# 依存分析
print("\n=== 依存分析 ===")

print("依存关系:")
for token in doc:
    print(f"  {token.text} <--{token.dep_}-- {token.head.text}")

# 可视化依存关系（需要displacy）
print("\n依存关系可视化:")
try:
    from spacy import displacy
    
    # 在Jupyter中可以使用 displacy.render
    # 在脚本中使用 displacy.serve 或保存为HTML
    html = displacy.render(doc, style='dep', page=True)
    print("依存关系HTML已生成（displacy支持）")
    
    # 命名实体可视化
    html_ent = displacy.render(doc, style='ent', page=True)
    print("命名实体HTML已生成")
except Exception as e:
    print(f"displacy可视化需要特殊配置: {e}")

# 词向量
print("\n=== 词向量 ===")

try:
    # 加载包含词向量的模型
    nlp_vec = spacy.load("en_core_web_md")
    
    # 获取词向量
    word1 = nlp_vec("king")
    word2 = nlp_vec("queen")
    word3 = nlp_vec("man")
    word4 = nlp_vec("woman")
    
    print(f"king 向量维度: {len(word1.vector)}")
    print(f"queen 向量维度: {len(word2.vector)}")
    
    # 计算相似度
    sim_king_queen = word1.similarity(word2)
    sim_king_man = word1.similarity(word3)
    sim_queen_woman = word2.similarity(word4)
    
    print(f"\n相似度计算:")
    print(f"  king vs queen: {sim_king_queen:.4f}")
    print(f"  king vs man: {sim_king_man:.4f}")
    print(f"  queen vs woman: {sim_queen_woman:.4f}")
    
    # 经典示例: king - man + woman ≈ queen
    result = word1.vector - word3.vector + word4.vector
    # 计算与queen的相似度
    from numpy import dot
    from numpy.linalg import norm
    
    cos_sim = dot(result, word2.vector) / (norm(result) * norm(word2.vector))
    print(f"\n向量运算: king - man + woman ≈ queen")
    print(f"  结果与queen的余弦相似度: {cos_sim:.4f}")
    
except OSError:
    print("未找到包含词向量的模型(en_core_web_md)")
    print("使用 en_core_web_sm 进行基础功能演示")

# 文本相似度
print("\n=== 文本相似度 ===")

texts = [
    "I love this movie, it's amazing",
    "This film is wonderful, I really like it",
    "The weather is nice today",
    "It's raining heavily outside",
    "Natural language processing is fascinating",
    "NLP is a great field of study"
]

# 计算文档相似度
try:
    nlp_sim = spacy.load("en_core_web_md")
    docs = [nlp_sim(text) for text in texts]
    
    print("文本相似度矩阵:")
    n = len(texts)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = docs[i].similarity(docs[j])
            else:
                similarity_matrix[i][j] = 1.0
    
    # 打印矩阵
    print(f"{'':>30}", end="")
    for i in range(n):
        print(f"{texts[i][:8]:>10}", end="")
    print()
    
    for i in range(n):
        print(f"{texts[i][:30]:<30}", end="")
        for j in range(n):
            print(f"{similarity_matrix[i][j]:>10.3f}", end="")
        print()
    
    # 可视化相似度矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label='相似度')
    plt.title('文本相似度矩阵')
    plt.xticks(range(n), [t[:10] for t in texts], rotation=45, ha='right')
    plt.yticks(range(n), [t[:10] for t in texts])
    plt.tight_layout()
    plt.show()
    
except OSError:
    print("需要包含词向量的模型进行相似度计算")
    print("提示: python -m spacy download en_core_web_md")

# 规则匹配
print("\n=== 基于规则的匹配 (Matcher) ===")

from spacy.matcher import Matcher

# 创建Matcher
matcher = Matcher(nlp.vocab)

# 定义匹配模式
# 模式1: 形容词 + 名词
pattern1 = [
    {"POS": "ADJ"},
    {"POS": "NOUN"}
]

# 模式2: 名词 + 动词
pattern2 = [
    {"POS": "NOUN"},
    {"POS": "VERB"}
]

# 模式3: 数字
pattern3 = [
    {"LIKE_NUM": True}
]

# 添加模式
matcher.add("ADJ_NOUN", [pattern1])
matcher.add("NOUN_VERB", [pattern2])
matcher.add("NUMBER", [pattern3])

# 测试文本
test_text = "The quick brown fox jumps over 42 lazy dogs. Beautiful flowers bloom in spring."
test_doc = nlp(test_text)

print(f"测试文本: {test_text}")

# 查找匹配
matches = matcher(test_doc)
print("\n匹配结果:")
for match_id, start, end in matches:
    span = test_doc[start:end]
    rule_name = nlp.vocab.strings[match_id]
    print(f"  规则: {rule_name}, 匹配: '{span.text}' (位置: {start}-{end})")

# 短语匹配
print("\n=== 短语匹配 (PhraseMatcher) ===")

from spacy.matcher import PhraseMatcher

# 创建PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab)

# 定义要匹配的短语
phrases = ["New York", "machine learning", "natural language processing", 
           "artificial intelligence", "deep learning"]
patterns = [nlp.make_doc(phrase) for phrase in phrases]

# 添加模式
phrase_matcher.add("PHRASES", patterns)

# 测试文本
phrase_text = "Natural language processing and machine learning are branches of artificial intelligence. New York is a great city."
phrase_doc = nlp(phrase_text)

print(f"测试文本: {phrase_text}")

# 查找匹配
phrase_matches = phrase_matcher(phrase_doc)
print("\n短语匹配结果:")
for match_id, start, end in phrase_matches:
    span = phrase_doc[start:end]
    print(f"  匹配: '{span.text}' (位置: {start}-{end})")

# 实体链接
print("\n=== 实体链接 ===")

print("实体类型统计:")
entity_counts = Counter()
for ent in doc.ents:
    entity_counts[ent.label_] += 1

for ent_type, count in entity_counts.most_common():
    print(f"  {ent_type}: {count}")

# 自定义管道组件
print("\n=== 自定义管道组件 ===")

# 创建自定义组件
@spacy.registry.misc("spacy.text_length")
def text_length(doc):
    """计算文本长度"""
    doc.set_extension("text_length", len(doc.text), force=True)
    return doc

# 简化版：使用普通函数
def custom_component(doc):
    """自定义管道组件 - 计算文本长度"""
    doc._.text_length = len(doc.text)
    return doc

# 设置扩展属性
if not Doc.has_extension("text_length"):
    from spacy.tokens import Doc
    Doc.set_extension("text_length", default=0)

# 添加到管道
nlp.add_pipe(custom_component, last=True)

# 处理文本
custom_doc = nlp(text)
print(f"文本: {custom_doc.text[:50]}...")
print(f"文本长度: {custom_doc._.text_length}")

# 简化版：使用现有模型
print("\n=== 处理大量文本 ===")

# 处理多篇文档
sample_texts = [
    "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language created by Guido van Rossum.",
    "The Great Wall of China is one of the wonders of the world.",
    "Tesla's CEO Elon Musk announced new electric vehicles in 2024."
]

print("批量处理多篇文档:")
for i, text in enumerate(sample_texts, 1):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"\n  文档 {i}: {text}")
    print(f"    实体: {entities}")

# 文本相似度应用 - 文档检索
print("\n=== 文档检索示例 ===")

# 文档集合
corpus = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks for complex tasks.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables machines to see and interpret images.",
    "Reinforcement learning trains agents through rewards."
]

query = "I want to learn about AI and neural networks"

# 处理查询
query_doc = nlp(query)

# 计算每个文档与查询的相似度
print(f"查询: {query}")
print("\n文档相关性排序:")

similarities = []
for i, doc_text in enumerate(corpus):
    doc = nlp(doc_text)
    try:
        sim = query_doc.similarity(doc)
    except:
        sim = 0  # 如果没有词向量模型
    similarities.append((i, sim, doc_text))

# 按相似度排序
similarities.sort(key=lambda x: x[1], reverse=True)

for rank, (idx, sim, text) in enumerate(similarities, 1):
    print(f"  {rank}. 相似度: {sim:.4f} - {text}")

# 命名实体识别详细应用
print("\n=== NER应用示例 ===")

# 提取和组织实体
def extract_entities_by_type(text):
    """按类型提取实体"""
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

# 测试
news_text = """On Monday, Microsoft CEO Satya Nadella announced a new partnership with OpenAI 
in San Francisco. The $10 billion investment will be used to develop advanced AI technologies. 
The announcement was made at the company's annual conference in 2024."""

print("新闻文本:")
print(news_text)
print("\n提取的实体:")
entities = extract_entities_by_type(news_text)
for ent_type, ents in entities.items():
    print(f"  {ent_type}: {ents}")

# 总结
print("\n=== spaCy学习总结 ===")
print("1. 基本NLP处理（分词、词性标注、依存分析）")
print("2. 命名实体识别（NER）")
print("3. 词向量和语义相似度")
print("4. 文本相似度计算")
print("5. 基于规则的匹配（Matcher）")
print("6. 短语匹配（PhraseMatcher）")
print("7. 自定义管道组件")
print("8. 文档检索应用")
print("9. 实体提取和组织")
print("10. 工业级NLP处理")

# 模型下载提示
print("\n=== spaCy模型下载提示 ===")
print("下载模型命令:")
print("  python -m spacy download en_core_web_sm   # 英文小模型")
print("  python -m spacy download en_core_web_md   # 英文中等模型（含词向量）")
print("  python -m spacy download en_core_web_lg   # 英文大模型")
print("  python -m spacy download zh_core_web_sm   # 中文小模型")

print("\nspaCy工业级NLP学习完成！")
