# spaCy文本处理与管道学习
# 主要内容：自定义管道、文本处理、向量表示

import spacy
import numpy as np

print("=== 加载模型 ===")
nlp = spacy.load('en_core_web_sm')

print("\n=== 词向量 ===")
text = "cat dog animal"
doc = nlp(text)
for token in doc:
    print(f"词: {token.text}, 有向量: {token.has_vector}")
    if token.has_vector:
        print(f"  向量维度: {token.vector.shape}")

print("\n=== 相似度计算 ===")
doc1 = nlp("cat")
doc2 = nlp("dog")
doc3 = nlp("car")
print(f"cat 和 dog 相似度: {doc1.similarity(doc2):.4f}")
print(f"cat 和 car 相似度: {doc1.similarity(doc3):.4f}")

print("\n=== 自定义管道组件 ===")
def custom_component(doc):
    doc._.custom_attr = "已处理"
    return doc

nlp.add_pipe(custom_component, last=True)
doc = nlp("Hello world")
print(f"自定义属性: {doc._.custom_attr}")

print("\n=== 移除管道组件 ===")
nlp.remove_pipe("custom_component")
print(f"管道组件: {nlp.pipe_names}")

print("\n=== 批量处理 ===")
texts = ["Hello world", "This is spaCy", "Natural language processing"]
for doc in nlp.pipe(texts):
    print(f"文本: {doc.text}, 词数: {len(doc)}")

print("\n=== 扩展属性 ===")
from spacy.tokens import Doc
Doc.set_extension("sentiment", default=0)

doc = nlp("This is great!")
doc._.sentiment = 1
print(f"文本: {doc.text}")
print(f"情感值: {doc._.sentiment}")