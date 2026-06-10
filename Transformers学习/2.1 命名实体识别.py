# Hugging Face Transformers命名实体识别学习
# 主要内容：使用预训练模型进行NER

from transformers import pipeline

print("=== 英文NER ===")
ner = pipeline('ner', grouped_entities=True)

text = "Apple Inc. is looking to buy a startup in California for $1 billion."
result = ner(text)

print(f"文本: {text}")
print("\n识别结果:")
for entity in result:
    print(f"  实体: {entity['word']}, 类型: {entity['entity_group']}, 置信度: {entity['score']:.4f}")

print("\n=== 中文NER ===")
chinese_ner = pipeline('ner', model='uer/roberta-base-finetuned-cluener2020-chinese')

chinese_text = "华为公司计划在深圳投资新建一个研发中心。"
result = chinese_ner(chinese_text)

print(f"文本: {chinese_text}")
print("\n识别结果:")
for entity in result:
    print(f"  实体: {entity['word']}, 类型: {entity['entity']}, 置信度: {entity['score']:.4f}")