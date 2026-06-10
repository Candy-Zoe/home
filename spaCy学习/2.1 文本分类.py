# spaCy文本分类学习
# 主要内容：使用spaCy进行文本分类

import spacy
from spacy.training import Example
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
import random

print("=== 创建训练数据 ===")
train_data = [
    ("I love this movie!", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("This is the best film ever", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("Great performance!", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("Terrible movie", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
    ("I hate this film", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
    ("Awful experience", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
]

print("\n=== 创建空模型 ===")
nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"})

textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

print("\n=== 训练模型 ===")
optimizer = nlp.begin_training()
for i in range(10):
    random.shuffle(train_data)
    losses = {}
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer, losses=losses)
    print(f"轮次 {i+1}, 损失: {losses['textcat']:.4f}")

print("\n=== 测试模型 ===")
test_texts = ["This movie is amazing!", "I don't like this film", "Great!"]
for text in test_texts:
    doc = nlp(text)
    print(f"文本: {text}")
    print(f"  分类: {doc.cats}")
    print(f"  预测: {'POSITIVE' if doc.cats['POSITIVE'] > 0.5 else 'NEGATIVE'}")