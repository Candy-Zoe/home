# Hugging Face Transformers文本分类学习
# 主要内容：使用预训练模型进行情感分析

from transformers import pipeline

print("=== 情感分析 ===")
classifier = pipeline('sentiment-analysis')

texts = [
    "I love using Hugging Face Transformers! It's amazing.",
    "I hate waiting for models to download.",
    "The new update is okay, not great but not terrible."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"文本: {text}")
    print(f"  情感: {result['label']}, 置信度: {result['score']:.4f}")
    print()

print("\n=== 中文情感分析 ===")
chinese_classifier = pipeline('sentiment-analysis', model='uer/roberta-base-finetuned-dianping-chinese')

chinese_texts = [
    "这家餐厅的食物非常好吃，服务也很周到。",
    "电影很无聊，浪费时间。",
    "今天天气不错，心情很好。"
]

results = chinese_classifier(chinese_texts)
for text, result in zip(chinese_texts, results):
    print(f"文本: {text}")
    print(f"  情感: {result['label']}, 置信度: {result['score']:.4f}")
    print()