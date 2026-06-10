# spaCy管道与扩展学习
# 主要内容：自定义管道组件、实体识别、文本分类、词向量

import spacy
from spacy import displacy

print("=== 加载模型 ===")
nlp = spacy.load("en_core_web_sm")

print("\n=== 处理文本 ===")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
print(f"文本: {doc.text}")

print("\n=== 分词和词性 ===")
for token in doc:
    print(f"{token.text:10} {token.pos_:8} {token.tag_:6} {token.dep_:10}")

print("\n=== 命名实体识别 ===")
print("命名实体:")
for ent in doc.ents:
    print(f"  {ent.text:20} {ent.label_:10} {spacy.explain(ent.label_)}")

print("\n=== 实体可视化 ===")
html = displacy.render(doc, style="ent")
print("实体可视化HTML已生成")

print("\n=== 依存关系可视化 ===")
html = displacy.render(doc, style="dep")
print("依存关系可视化HTML已生成")

print("\n=== 自定义管道组件 ===")
def custom_component(doc):
    for token in doc:
        token.set_extension("is_custom", default=False, force=True)
        if token.text.lower() in ["machine", "learning", "ai"]:
            token._.is_custom = True
    return doc

nlp.add_pipe(custom_component, last=True)
doc = nlp("Machine learning and AI are transforming the world.")
print("自定义属性:")
for token in doc:
    if token._.is_custom:
        print(f"  {token.text} -> is_custom=True")

print("\n=== 词向量 ===")
doc = nlp("cat dog apple banana")
print("词向量相似度:")
for token1 in doc:
    for token2 in doc:
        if token1 < token2:
            sim = token1.similarity(token2)
            print(f"  {token1.text} - {token2.text}: {sim:.4f}")

print("\n=== 文本分类 ===")
textcat = nlp.add_pipe("textcat")
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")
print("文本分类管道已添加")

print("\n=== 规则匹配 ===")
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "machine"}, {"LOWER": "learning"}]
matcher.add("MachineLearning", [pattern])

doc = nlp("Machine learning is fun. Machine Learning is powerful.")
matches = matcher(doc)
print("规则匹配结果:")
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"  匹配: {span.text} (位置: {start}-{end})")