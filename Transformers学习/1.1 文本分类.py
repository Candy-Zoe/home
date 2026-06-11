# Hugging Face Transformers文本分类学习
# 主要内容：使用预训练模型进行文本分类、情感分析

# 导入必要的库
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# 使用pipeline进行文本分类（情感分析）
print("=== 使用pipeline进行情感分析 ===")

# 创建情感分析pipeline
classifier = pipeline('sentiment-analysis')

# 测试文本
texts = [
    "I love using Transformers! It's so easy.",
    "I hate this movie. It was terrible.",
    "The weather is okay today.",
    "This restaurant is amazing!"
]

# 进行情感分析
results = classifier(texts)

# 打印结果
for text, result in zip(texts, results):
    print(f"文本: {text}")
    print(f"  标签: {result['label']}, 置信度: {result['score']:.4f}")
    print()

# 使用自定义模型和tokenizer
print("\n=== 使用自定义模型 ===")

# 选择预训练模型
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer加载成功: {tokenizer.__class__.__name__}")

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print(f"模型加载成功: {model.__class__.__name__}")

# 准备输入文本
text = "Transformers is awesome!"

# 对文本进行编码
inputs = tokenizer(text, return_tensors="pt")
print(f"\n输入编码:")
print(f"  input_ids: {inputs['input_ids']}")
print(f"  attention_mask: {inputs['attention_mask']}")

# 使用模型进行预测
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predicted_class = torch.argmax(logits).item()
probabilities = torch.nn.functional.softmax(logits, dim=1)

print(f"\n预测结果:")
print(f"  Logits: {logits}")
print(f"  预测类别索引: {predicted_class}")
print(f"  类别概率: {probabilities}")
print(f"  预测标签: {model.config.id2label[predicted_class]}")

# 批量处理文本
print("\n=== 批量处理 ===")

# 多个文本
texts = [
    "This is great!",
    "I don't like this product.",
    "Neutral statement here."
]

# 批量编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(f"批量输入形状:")
print(f"  input_ids: {inputs['input_ids'].shape}")
print(f"  attention_mask: {inputs['attention_mask'].shape}")

# 批量预测
with torch.no_grad():
    outputs = model(**inputs)

# 处理结果
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=1)
predictions = torch.argmax(logits, dim=1)

print("\n批量预测结果:")
for text, pred, prob in zip(texts, predictions, probabilities):
    label = model.config.id2label[pred.item()]
    score = prob[pred.item()].item()
    print(f"文本: {text}")
    print(f"  标签: {label}, 置信度: {score:.4f}")
    print()

# 多标签分类示例
print("\n=== 多标签分类 ===")

# 使用多标签分类模型
multi_classifier = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli")

# 待分类文本
text = "The new iPhone has amazing camera quality."

# 候选标签
candidate_labels = ["technology", "sports", "politics", "entertainment"]

# 进行零样本分类
result = multi_classifier(text, candidate_labels)
print(f"文本: {text}")
print(f"候选标签: {candidate_labels}")
print(f"分类结果:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.4f}")