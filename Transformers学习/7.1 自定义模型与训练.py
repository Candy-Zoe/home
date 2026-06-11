# Transformers自定义模型与训练学习
# 主要内容：自定义数据集、自定义模型、训练循环、评估指标

# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 创建自定义数据集
print("=== 创建自定义数据集 ===")

# 示例数据
texts = [
    "这部电影太棒了，演员表演非常出色！",
    "剧情很精彩，推荐大家观看。",
    "画面精美，音乐动人，值得一看。",
    "非常失望，剧情拖沓，浪费时间。",
    "演员演技太差，不会再看了。",
    "故事很感人，让我哭了好几次。",
    "特效很棒，场面宏大。",
    "情节老套，毫无新意。",
    "导演手法独特，很有创意。",
    "浪费钱和时间，不推荐。"
]
labels = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]  # 1=正面评价，0=负面评价

# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

print(f"训练集大小: {len(train_texts)}")
print(f"测试集大小: {len(test_texts)}")
print(f"训练集标签分布: {np.bincount(train_labels)}")
print(f"测试集标签分布: {np.bincount(test_labels)}")

# 创建自定义Dataset类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载BERT tokenizer
print("\n=== 加载Tokenizer ===")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(f"Tokenizer词汇表大小: {tokenizer.vocab_size}")
print(f"特殊token: {tokenizer.special_tokens_map}")

# 创建数据集
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

# 查看数据示例
sample = train_dataset[0]
print(f"\n示例数据:")
print(f"  input_ids形状: {sample['input_ids'].shape}")
print(f"  attention_mask形状: {sample['attention_mask'].shape}")
print(f"  label: {sample['labels']}")
print(f"  解码文本: {tokenizer.decode(sample['input_ids'])}")

# 创建数据加载器
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 使用预训练模型进行分类
print("\n=== 使用预训练BERT进行分类 ===")

# 加载预训练模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"使用设备: {device}")

# 训练配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# 定义评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 训练模型
print("\n=== 开始训练 ===")
trainer.train()

# 评估模型
print("\n=== 评估模型 ===")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 进行预测
print("\n=== 进行预测 ===")
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

print(f"预测结果: {preds}")
print(f"真实标签: {labels}")
print(f"\n分类报告:")
print(classification_report(labels, preds))

# 创建自定义模型
print("\n=== 创建自定义模型 ===")

class CustomBERTModel(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomBERTModel, self).__init__()
        # 加载预训练BERT
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        # 冻结BERT参数（可选）
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # 自定义分类头
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, num_labels)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token的隐藏状态
        cls_output = outputs[1]
        
        # 自定义分类头
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

# 初始化自定义模型
custom_model = CustomBERTModel(num_labels=2).to(device)
print(f"自定义模型参数数量: {sum(p.numel() for p in custom_model.parameters())}")

# 手动训练循环
print("\n=== 手动训练循环 ===")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(custom_model.parameters(), lr=2e-5)

# 训练参数
num_epochs = 5
best_acc = 0.0

# 训练循环
for epoch in range(num_epochs):
    custom_model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in train_loader:
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = custom_model(input_ids, attention_mask)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_acc = correct / total
    train_loss = total_loss / len(train_loader)
    
    # 验证
    custom_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = custom_model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    val_loss = val_loss / len(test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
    print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(custom_model.state_dict(), 'best_custom_model.pth')
        print("  保存最佳模型")

# 加载最佳模型进行预测
print("\n=== 使用最佳模型进行预测 ===")
custom_model.load_state_dict(torch.load('best_custom_model.pth'))
custom_model.eval()

# 预测示例文本
test_samples = [
    "这部电影真的很好看！",
    "浪费时间，不推荐。",
    "值得一看的好电影。"
]

for text in test_samples:
    # 编码文本
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = custom_model(input_ids, attention_mask)
        _, pred = torch.max(outputs, dim=1)
    
    sentiment = "正面" if pred.item() == 1 else "负面"
    print(f"文本: '{text}'")
    print(f"  情感预测: {sentiment}")
    print(f"  预测概率: {torch.softmax(outputs, dim=1).cpu().numpy()[0]}")

# 混淆矩阵可视化
print("\n=== 混淆矩阵可视化 ===")

# 获取测试集预测结果
all_preds = []
all_labels = []

custom_model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = custom_model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

# 可视化
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['负面', '正面'])
plt.yticks(tick_marks, ['负面', '正面'])

# 在每个格子里显示数值
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# 清理临时文件
import os
if os.path.exists('best_custom_model.pth'):
    os.remove('best_custom_model.pth')
    print("已删除临时模型文件")

print("\nTransformers自定义模型与训练学习完成！")
