# Hugging Face Transformers自定义模型训练学习
# 主要内容：使用Trainer API进行模型微调

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

print("=== 加载预训练模型 ===")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

print("\n=== 准备训练数据 ===")
class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128)
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label)
        }

train_texts = [
    "I love using Hugging Face!",
    "This is amazing!",
    "I hate this product.",
    "Terrible experience."
]
train_labels = [1, 1, 0, 0]

train_dataset = Dataset(train_texts, train_labels)

print("\n=== 设置训练参数 ===")
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
)

print("\n=== 创建Trainer ===")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("\n=== 开始训练 ===")
trainer.train()

print("\n=== 保存模型 ===")
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
print("模型已保存")

print("\n=== 清理测试文件 ===")
import shutil
import os
if os.path.exists('./results'):
    shutil.rmtree('./results')
if os.path.exists('./logs'):
    shutil.rmtree('./logs')
if os.path.exists('./fine_tuned_model'):
    shutil.rmtree('./fine_tuned_model')
print("测试文件已清理")