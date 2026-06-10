# Transformers微调技巧学习
# 主要内容：参数高效微调、LoRA、PEFT、Prompt Tuning

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np

print("=== 加载模型和数据集 ===")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = load_dataset("imdb", split="train[:100]")

print("\n=== 数据预处理 ===")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("\n=== 传统微调 ===")
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("传统微调训练器已创建")

print("\n=== 参数高效微调 (PEFT) ===")
try:
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args_peft = TrainingArguments(
        output_dir="./results_peft",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs_peft",
        logging_steps=10,
        report_to="none"
    )
    
    trainer_peft = Trainer(
        model=model,
        args=training_args_peft,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("PEFT训练器已创建")
    
except ImportError:
    print("PEFT未安装，跳过参数高效微调示例")

print("\n=== 清理测试文件 ===")
import shutil
import os
for dir_name in ['./results', './logs', './results_peft', './logs_peft']:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
        print(f"已删除 {dir_name}")