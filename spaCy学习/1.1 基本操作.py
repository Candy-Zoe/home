# spaCy基本操作学习
# 主要内容：文档对象、分词、词性标注、命名实体识别

import spacy

print("=== 加载模型 ===")
nlp = spacy.load('en_core_web_sm')

print("\n=== 创建文档 ===")
text = "Apple Inc. is looking to buy a startup in California for $1 billion."
doc = nlp(text)

print("\n=== 分词和词性标注 ===")
print(f"文本: {text}")
print("\n分词结果:")
for token in doc:
    print(f"  {token.text} - 词性: {token.pos_}, 细粒度词性: {token.tag_}, 依存关系: {token.dep_}")

print("\n=== 命名实体识别 ===")
print("实体识别结果:")
for ent in doc.ents:
    print(f"  实体: {ent.text}, 类型: {ent.label_}, 描述: {spacy.explain(ent.label_)}")

print("\n=== 名词短语 ===")
noun_chunks = list(doc.noun_chunks)
print(f"名词短语: {[chunk.text for chunk in noun_chunks]}")

print("\n=== 可视化 ===")
from spacy import displacy
displacy.serve(doc, style='dep')