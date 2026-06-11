# Transformers微调技巧学习
# 主要内容：参数高效微调、LoRA、Adapter、梯度累积、学习率调度

# 导入必要的库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    TrainingArguments, Trainer, AutoConfig, AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 创建示例数据集
print("=== 创建示例数据集 ===")

# 示例数据
texts = [
    "这个产品质量很好，非常满意！",
    "服务态度很差，不会再买了。",
    "发货速度快，包装精美。",
    "价格太贵，不值这个钱。",
    "物流太慢，等了很久才到。",
    "客服很耐心，解决了我的问题。",
    "商品和描述不符，差评！",
    "性价比很高，推荐购买。",
    "快递员态度好，送货上门。",
    "质量一般，勉强能用。",
    "售后很好，退换货方便。",
    "颜色和图片不一样，失望。"
]
labels = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  # 1=正面，0=负面

# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

print(f"训练集大小: {len(train_texts)}")
print(f"测试集大小: {len(test_texts)}")

# 创建Dataset类
class ReviewDataset(Dataset):
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
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
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

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
test_dataset = ReviewDataset(test_texts, test_labels, tokenizer)

# 方法1：冻结部分层
print("\n=== 方法1：冻结部分层 ===")

model_freeze = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
)

# 冻结前6层
print("冻结前6层...")
for i, (name, param) in enumerate(model_freeze.named_parameters()):
    if 'bert.encoder.layer.' in name:
        layer_num = int(name.split('.')[3])
        if layer_num < 6:
            param.requires_grad = False

# 统计可训练参数
trainable_params = sum(p.numel() for p in model_freeze.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_freeze.parameters())
print(f"可训练参数: {trainable_params:,}")
print(f"总参数: {total_params:,}")
print(f"可训练比例: {trainable_params / total_params * 100:.2f}%")

# 方法2：使用学习率衰减
print("\n=== 方法2：学习率衰减 ===")

# 不同层设置不同学习率
optimizer_grouped_parameters = [
    {'params': model_freeze.bert.embeddings.parameters(), 'lr': 1e-5},
    {'params': model_freeze.bert.encoder.layer[:6].parameters(), 'lr': 1e-5},
    {'params': model_freeze.bert.encoder.layer[6:].parameters(), 'lr': 2e-5},
    {'params': model_freeze.classifier.parameters(), 'lr': 1e-4}
]

optimizer = AdamW(optimizer_grouped_parameters)

print("学习率设置:")
print("  - Embeddings层: 1e-5")
print("  - 前6层: 1e-5")
print("  - 后6层: 2e-5")
print("  - 分类头: 1e-4")

# 方法3：梯度累积
print("\n=== 方法3：梯度累积 ===")

# 梯度累积步数
gradient_accumulation_steps = 4
effective_batch_size = 2 * gradient_accumulation_steps  # batch_size=2
print(f"实际batch_size: {effective_batch_size}")
print(f"梯度累积步数: {gradient_accumulation_steps}")

# 方法4：学习率调度器
print("\n=== 方法4：学习率调度器 ===")

# 创建学习率调度器
num_train_steps = len(train_dataset) // effective_batch_size * 3  # 3 epochs
warmup_steps = int(num_train_steps * 0.1)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_train_steps
)

print(f"训练步数: {num_train_steps}")
print(f"预热步数: {warmup_steps}")

# 可视化学习率变化
lr_values = []
for _ in range(num_train_steps):
    lr_values.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.figure(figsize=(10, 5))
plt.plot(lr_values)
plt.xlabel('训练步数')
plt.ylabel('学习率')
plt.title('学习率调度曲线')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 方法5：参数高效微调 - LoRA
print("\n=== 方法5：LoRA参数高效微调 ===")

try:
    from peft import get_peft_model, LoraConfig, TaskType
    
    # 创建LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # 秩
        lora_alpha=32,  # 缩放因子
        target_modules=["query", "value"],  # 应用LoRA的层
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"]  # 保存的模块
    )
    
    # 创建模型
    model_lora = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=2
    )
    
    # 应用LoRA
    model_lora = get_peft_model(model_lora, lora_config)
    model_lora.print_trainable_parameters()
    
    # 训练配置
    training_args_lora = TrainingArguments(
        output_dir='./lora_results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_dir='./lora_logs',
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {'accuracy': accuracy_score(labels, preds)}
    
    trainer_lora = Trainer(
        model=model_lora,
        args=training_args_lora,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("\n开始LoRA微调...")
    trainer_lora.train()
    
except ImportError:
    print("peft库未安装，跳过LoRA示例")

# 方法6：Adapter微调
print("\n=== 方法6：Adapter微调 ===")

try:
    from peft import PrefixTuningConfig, PromptEncoderConfig
    
    # Prefix Tuning配置
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=20,
        encoder_hidden_size=128
    )
    
    model_prefix = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=2
    )
    
    model_prefix = get_peft_model(model_prefix, prefix_config)
    model_prefix.print_trainable_parameters()
    
except ImportError:
    print("peft库未安装，跳过Adapter示例")

# 方法7：使用梯度检查点
print("\n=== 方法7：梯度检查点 ===")

model_checkpoint = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
)

# 启用梯度检查点（节省内存）
model_checkpoint.gradient_checkpointing_enable()
print("已启用梯度检查点")

# 方法8：混合精度训练
print("\n=== 方法8：混合精度训练 ===")

training_args_fp16 = TrainingArguments(
    output_dir='./fp16_results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    fp16=True,  # 启用混合精度
    fp16_opt_level='O1',  # 优化级别
    learning_rate=2e-5,
    logging_dir='./fp16_logs',
    evaluation_strategy='epoch'
)

print("混合精度训练配置:")
print("  - fp16: True")
print("  - fp16_opt_level: O1")

# 方法9：Early Stopping
print("\n=== 方法9：Early Stopping ===")

training_args_early = TrainingArguments(
    output_dir='./early_results',
    num_train_epochs=10,  # 设置更多epoch
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    logging_dir='./early_logs',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    greater_is_better=True,
    patience=3  # 3个epoch没有提升则停止
)

print("Early Stopping配置:")
print("  - patience: 3")
print("  - metric_for_best_model: eval_accuracy")

# 方法10：数据增强
print("\n=== 方法10：数据增强 ===")

def augment_text(text, prob=0.1):
    """简单的数据增强"""
    words = list(text)
    augmented = []
    
    for i, char in enumerate(words):
        # 随机删除
        if np.random.random() < prob:
            continue
        # 随机替换（保持中文特征）
        elif np.random.random() < prob:
            # 简单替换为相似字符
            if char == '好':
                augmented.append('棒')
            elif char == '棒':
                augmented.append('好')
            elif char == '差':
                augmented.append('坏')
            elif char == '坏':
                augmented.append('差')
            else:
                augmented.append(char)
        else:
            augmented.append(char)
    
    return ''.join(augmented)

# 示例
print("原始文本: 这个产品质量很好")
print(f"增强后: {augment_text('这个产品质量很好')}")
print(f"增强后: {augment_text('这个产品质量很好')}")
print(f"增强后: {augment_text('这个产品质量很好')}")

# 方法11：正则化技巧
print("\n=== 方法11：正则化技巧 ===")

# Dropout正则化
model_reg = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
)

# 添加额外的Dropout层
class ModelWithDropout(nn.Module):
    def __init__(self, base_model, dropout_rate=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = self.dropout(outputs.logits)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

model_with_dropout = ModelWithDropout(model_reg)
print("已添加Dropout层，dropout_rate=0.3")

# 方法12：知识蒸馏
print("\n=== 方法12：知识蒸馏 ===")

# 定义教师模型（更大的模型）
teacher_model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
)

# 定义学生模型（更小的模型）
class StudentModel(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # 使用更小的隐藏层
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        return self.classifier(outputs[1])

student_model = StudentModel(num_labels=2)

# 蒸馏损失
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # 软标签损失
    soft_target_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / temperature, dim=-1),
        nn.functional.softmax(teacher_logits / temperature, dim=-1)
    ) * (temperature ** 2)
    
    # 硬标签损失
    hard_target_loss = nn.CrossEntropyLoss()(student_logits, labels)
    
    # 混合损失
    return alpha * soft_target_loss + (1 - alpha) * hard_target_loss

print("知识蒸馏损失函数已定义")
print("  - temperature: 2.0")
print("  - alpha: 0.5")

# 完整训练示例（使用多种技巧）
print("\n=== 完整训练示例 ===")

# 创建最终模型
final_model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
)

# 冻结部分层
for name, param in final_model.named_parameters():
    if 'bert.encoder.layer.' in name:
        layer_num = int(name.split('.')[3])
        if layer_num < 4:
            param.requires_grad = False

# 训练配置（组合多种技巧）
final_training_args = TrainingArguments(
    output_dir='./final_results',
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    logging_dir='./final_logs',
    logging_steps=5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    greater_is_better=True,
    fp16=True if torch.cuda.is_available() else False,
    remove_unused_columns=False
)

trainer_final = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("开始训练...")
trainer_final.train()

# 评估
eval_results = trainer_final.evaluate()
print(f"\n最终评估结果: {eval_results}")

# 预测示例
print("\n预测示例:")
test_texts_predict = [
    "非常满意这次购物体验！",
    "商品质量很差，很失望。",
    "物流很快，服务很好。"
]

for text in test_texts_predict:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = final_model(**encoding)
        pred = outputs.logits.argmax(-1).item()
        prob = torch.softmax(outputs.logits, dim=1).max().item()
    
    sentiment = "正面" if pred == 1 else "负面"
    print(f"'{text}' -> {sentiment} (概率: {prob:.4f})")

# 清理临时文件
import os
import shutil
for dir_name in ['lora_results', 'fp16_results', 'early_results', 'final_results', 
                 'lora_logs', 'fp16_logs', 'early_logs', 'final_logs']:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
        print(f"已删除目录: {dir_name}")

print("\nTransformers微调技巧学习完成！")
