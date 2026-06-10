# spaCy高级NLP任务学习
# 主要内容：命名实体识别进阶、关系抽取、文本摘要、语义相似度

import spacy
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
from collections import Counter

print("=== 加载模型 ===")
nlp = spacy.load("en_core_web_sm")

print("\n=== 命名实体识别进阶 ===")
text = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
The company is headquartered in Cupertino, California.
Microsoft was founded by Bill Gates and Paul Allen in 1975.
"""

doc = nlp(text)
print("命名实体:")
for ent in doc.ents:
    print(f"  {ent.text:30} {ent.label_:15} {spacy.explain(ent.label_)}")

print("\n=== 实体关系抽取 ===")
def extract_relations(doc):
    relations = []
    for ent1 in doc.ents:
        for ent2 in doc.ents:
            if ent1 != ent2:
                if ent1.end < ent2.start:
                    between = doc[ent1.end:ent2.start]
                    if any(token.pos_ == "VERB" for token in between):
                        relations.append((ent1.text, ent2.text))
    return relations

relations = extract_relations(doc)
print(f"发现关系: {relations}")

print("\n=== 规则匹配 ===")
matcher = Matcher(nlp.vocab)

pattern = [{"LOWER": "apple"}, {"IS_TITLE": True, "OP": "?"}]
matcher.add("ApplePattern", [pattern])

matches = matcher(doc)
print("匹配结果:")
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"  {span.text}")

print("\n=== 短语匹配 ===")
phrase_matcher = PhraseMatcher(nlp.vocab)
terms = ["Apple Inc.", "Microsoft", "Google"]
patterns = [nlp(term) for term in terms]
phrase_matcher.add("TechCompanies", patterns)

matches = phrase_matcher(doc)
print("短语匹配:")
for match_id, start, end in matches:
    print(f"  {doc[start:end].text}")

print("\n=== 依存分析 ===")
sentence = "Apple was founded by Steve Jobs."
doc_sent = nlp(sentence)

print("依存关系:")
for token in doc_sent:
    print(f"  {token.text:15} {token.dep_:15} {token.head.text}")

print("\n=== 名词短语提取 ===")
doc = nlp("The quick brown fox jumps over the lazy dog.")
noun_chunks = list(doc.noun_chunks)
print(f"名词短语: {[chunk.text for chunk in noun_chunks]}")

print("\n=== 语义相似度 ===")
doc1 = nlp("I like cats")
doc2 = nlp("I love dogs")
doc3 = nlp("The weather is nice")

print(f"'{doc1}' vs '{doc2}': {doc1.similarity(doc2):.4f}")
print(f"'{doc1}' vs '{doc3}': {doc1.similarity(doc3):.4f}")

print("\n=== 词性标注详解 ===")
doc = nlp("The quick brown fox jumps over the lazy dog.")
print("词性标注:")
for token in doc:
    print(f"  {token.text:15} {token.pos_:10} {token.tag_:10} {spacy.explain(token.tag_)}")

print("\n=== 句法分析可视化 ===")
doc = nlp("This is a sentence.")
html = displacy.render(doc, style="dep", options={'compact': True})
print("依存关系可视化HTML已生成")

print("\n=== 文本预处理管道 ===")
def preprocess_text(text):
    doc = nlp(text)
    
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct]
    
    return tokens

text = "I am running and jumping over the fence!"
tokens = preprocess_text(text)
print(f"预处理结果: {tokens}")

print("\n=== 自定义组件 ===")
from spacy.language import Language

@Language.component("custom_component")
def custom_component(doc):
    doc._.custom_attr = "Custom value"
    return doc

Language.set_extension("custom_attr", default=None)
nlp.add_pipe("custom_component", last=True)

doc = nlp("Hello world")
print(f"自定义属性: {doc._.custom_attr}")