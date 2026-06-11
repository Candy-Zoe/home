# spaCy自然语言处理基础学习
# 主要内容：分词、词性标注、命名实体识别、依存分析

# 导入spaCy库
import spacy

# 加载spaCy的中文模型
print("=== 加载spaCy模型 ===")

# 尝试加载中文模型
try:
    nlp = spacy.load('zh_core_web_sm')
    print("中文模型加载成功")
except:
    # 如果没有中文模型，下载模型
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'zh_core_web_sm'])
    nlp = spacy.load('zh_core_web_sm')

print(f"模型信息: {nlp.meta['name']}")
print(f"管道组件: {nlp.pipe_names}")

# 中文分词
print("\n=== 中文分词 ===")

# 处理中文文本
text = "自然语言处理是人工智能领域的重要研究方向。"

doc = nlp(text)

# 分词
print(f"原文: {text}")
print(f"分词结果: {[token.text for token in doc]}")

# 词性标注
print("\n=== 词性标注 ===")

# 处理中文文本进行词性标注
text_pos = "我爱学习机器学习和深度学习技术。"
doc_pos = nlp(text_pos)

print("词性标注结果:")
for token in doc_pos:
    # POS标签含义:
    # n: 名词, v: 动词, r: 代词, u: 助词, a: 形容词等
    print(f"  {token.text} - {token.pos_} ({token.pos})")

# 命名实体识别
print("\n=== 命名实体识别 ===")

# 处理包含命名实体的文本
text_ner = "马云创办了阿里巴巴集团，总部位于中国杭州市。"

doc_ner = nlp(text_ner)

print(f"文本: {text_ner}")
print("\n命名实体:")

for ent in doc_ner.ents:
    print(f"  {ent.text} - {ent.label_} ({spacy.explain(ent.label_)})")

# 可视化命名实体
from spacy import displacy

# 渲染命名实体
html = displacy.render(doc_ner, style='ent', page=True)
print("\n命名实体可视化已生成")

# 依存分析
print("\n=== 依存分析 ===")

# 处理文本进行依存分析
text_dep = "深度学习模型在图像识别领域取得了突破性进展。"

doc_dep = nlp(text_dep)

print("依存分析结果:")
for token in doc_dep:
    # 依存标签含义:
    # nsubj: 名词性主语, ROOT: 根动词, dobj: 直接宾语, det: 限定词等
    print(f"  {token.text:6s} -> {token.dep_:6s} ({spacy.explain(token.dep_)}): {token.head.text}")

# 可视化依存树
html_dep = displacy.render(doc_dep, style='dep', page=True, options={'distance': 100})
print("\n依存树可视化已生成")

# 句子边界检测
print("\n=== 句子边界检测 ===")

# 处理多句子文本
text_sent = "自然语言处理非常重要。它可以帮助计算机理解和生成人类语言。机器学习是实现NLP的关键技术。"

doc_sent = nlp(text_sent)

print(f"句子数量: {len(list(doc_sent.sents))}")
print("\n句子分割结果:")
for i, sent in enumerate(doc_sent.sents, 1):
    print(f"  句子{i}: {sent.text}")

# 词形还原和词干提取
print("\n=== 词形还原 ===")

# 处理英文文本进行词形还原
try:
    nlp_en = spacy.load('en_core_web_sm')
    text_en = "I am running and I ran yesterday."

    doc_en = nlp_en(text_en)

    print(f"原文: {text_en}")
    print("词形还原结果:")
    for token in doc_en:
        print(f"  {token.text} -> {token.lemma_}")
except:
    print("英文模型未安装，跳过英文示例")

# 词向量和相似度
print("\n=== 词向量和相似度 ===")

# 检查模型是否包含词向量
if nlp.meta.get('vectors', {}).get('size', 0) > 0:
    # 获取词向量
    text_sim = "深度学习"

    doc_sim = nlp(text_sim)

    print(f"词向量维度: {nlp.vocab.vectors_length}")

    # 计算相似度
    tokens = nlp("自然语言处理和机器学习")
    token1, token2 = list(tokens)[:2]

    print(f"\n'{token1.text}'和'{token2.text}'的相似度: {token1.similarity(token2):.4f}")
else:
    print("当前模型不包含词向量")

# 文档和词Span
print("\n=== 文档和Span ===")

# 处理文本
text_span = "自然语言处理技术正在快速发展。"

doc_full = nlp(text_span)

# 获取Span
span = doc_full[0:4]  # 前4个词
print(f"完整文档: {doc_full.text}")
print(f"Span: {span.text}")
print(f"Span起止: {span.start_char} - {span.end_char}")

# 自定义组件
print("\n=== 自定义组件 ===")

def custom_component(doc):
    """自定义组件：在分词后添加新属性"""
    # 标记包含"学习"的词
    for token in doc:
        if "学习" in token.text:
            token._.is_learning_related = True
        else:
            token._.is_learning_related = False
    return doc

# 添加自定义属性
from spacy.tokens import Token
Token.set_extension('is_learning_related', default=False)

# 添加组件到管道
nlp.add_pipe(custom_component, first=True)

# 处理文本
text_custom = "我喜欢学习机器学习和深度学习。"

doc_custom = nlp(text_custom)

print(f"文本: {text_custom}")
print("包含'学习'的词:")
for token in doc_custom:
    if token._.get('is_learning_related'):
        print(f"  {token.text}")

# 规则匹配
print("\n=== 规则匹配 ===")

from spacy.matcher import Matcher

# 创建匹配器
matcher = Matcher(nlp.vocab)

# 定义匹配模式
pattern = [
    {"TEXT": "机器学习"}
]
matcher.add("machine_learning", [pattern])

# 处理文本进行匹配
text_match = "机器学习和深度学习是人工智能的核心技术。"

doc_match = nlp(text_match)
matches = matcher(doc_match)

print(f"文本: {text_match}")
print(f"匹配结果数量: {len(matches)}")

for match_id, start, end in matches:
    span = doc_match[start:end]
    print(f"  匹配: {span.text}")

# 处理大规模文本
print("\n=== 批处理 ===")

# 创建多个文本
texts = [
    "自然语言处理是人工智能的重要分支。",
    "机器学习广泛应用于各个领域。",
    "深度学习在计算机视觉方面取得突破。"
]

# 批处理
docs = list(nlp.pipe(texts, batch_size=2))

print(f"处理文本数量: {len(docs)}")
for i, doc in enumerate(docs, 1):
    print(f"  文本{i}: {doc.text}")
    print(f"  分词: {[token.text for token in doc]}")

print("\nspaCy基础学习完成！")