# PyTorch循环神经网络文本分类学习
# 主要内容：LSTM/GRU网络构建、文本嵌入、序列分类

# 导入PyTorch库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 准备文本数据
print("=== 准备文本数据 ===")

# 示例文本数据（正面和负面评论）
texts = [
    "这部电影太棒了，非常推荐观看",
    "这个产品非常好用，质量很棒",
    "服务态度很差，不会再来",
    "太失望了，完全不推荐",
    "性价比很高，值得购买",
    "质量太差，不值这个价",
    "内容精彩，看得很过瘾",
    "无聊透顶，浪费时间",
    "超出预期，非常满意",
    "简直是垃圾，浪费钱"
]

# 标签：1表示正面，0表示负面
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

print(f"样本数量: {len(texts)}")
print(f"正面评论数: {sum(labels)}")
print(f"负面评论数: {len(labels) - sum(labels)}")

# 文本预处理
print("\n=== 文本预处理 ===")

def tokenize(text):
    """简单的中文分词（基于字符）"""
    return list(text)

# 构建词汇表
tokenized_texts = [tokenize(text) for text in texts]
all_tokens = [token for tokens in tokenized_texts for token in tokens]
vocab = Counter(all_tokens)

# 创建词汇表字典
vocab_dict = {'<PAD>': 0, '<UNK>': 1}  # PAD用于填充，UNK用于未知词
for word, count in vocab.most_common():
    vocab_dict[word] = len(vocab_dict)

print(f"词汇表大小: {len(vocab_dict)}")
print(f"词汇表示例: {dict(list(vocab_dict.items())[:10])}")

# 将文本转换为索引序列
def text_to_indices(text, vocab):
    """将文本转换为索引序列"""
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return indices

# 数据集类
print("\n=== 创建数据集 ===")

class TextDataset(Dataset):
    """文本数据集类"""

    def __init__(self, texts, labels, vocab, max_length=20):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 转换为索引
        indices = text_to_indices(text, self.vocab)

        # 填充或截断
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices), torch.tensor(label)

# 创建数据集和数据加载器
dataset = TextDataset(texts, labels, vocab_dict, max_length=20)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"数据集大小: {len(dataset)}")
print(f"批次数量: {len(dataloader)}")

# 查看一个批次
for batch_texts, batch_labels in dataloader:
    print(f"\n批次文本形状: {batch_texts.shape}")
    print(f"批次标签形状: {batch_labels.shape}")
    break

# 定义LSTM模型
print("\n=== 定义LSTM模型 ===")

class TextClassificationLSTM(nn.Module):
    """基于LSTM的文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 output_dim, dropout=0.5):
        super(TextClassificationLSTM, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向LSTM
        )

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2因为双向

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x形状: (batch_size, seq_length)

        # 嵌入层
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # LSTM层
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output: (batch_size, seq_length, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)

        # 合并双向的最后隐藏状态
        # hidden[-2,:,:] 是最后时刻前向的隐藏状态
        # hidden[-1,:,:] 是最后时刻后向的隐藏状态
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden_cat: (batch_size, hidden_dim * 2)

        # Dropout
        hidden_dropout = self.dropout(hidden_cat)

        # 全连接层
        output = self.fc(hidden_dropout)
        # output: (batch_size, output_dim)

        return output

# 定义GRU模型
class TextClassificationGRU(nn.Module):
    """基于GRU的文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 output_dim, dropout=0.5):
        super(TextClassificationGRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, hidden = self.gru(embedded)
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden_dropout = self.dropout(hidden_cat)
        output = self.fc(hidden_dropout)
        return output

# 模型参数
VOCAB_SIZE = len(vocab_dict)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
OUTPUT_DIM = 2  # 二分类
DROPOUT = 0.5

# 创建模型
lstm_model = TextClassificationLSTM(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT
)
gru_model = TextClassificationGRU(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT
)

print(f"LSTM模型结构:\n{lstm_model}")

# 计算模型参数数量
total_params = sum(p.numel() for p in lstm_model.parameters())
trainable_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"\n总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")

# 训练函数
print("\n=== 训练模型 ===")

def train_model(model, dataloader, epochs=50, lr=0.001):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for texts, labels in dataloader:
            # 前向传播
            outputs = model(texts)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算平均损失和准确率
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total

        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return losses, accuracies

# 训练LSTM模型
print("\n训练LSTM模型...")
lstm_losses, lstm_accs = train_model(lstm_model, dataloader, epochs=50)

# 训练GRU模型
print("\n训练GRU模型...")
gru_losses, gru_accs = train_model(gru_model, dataloader, epochs=50)

# 可视化训练过程
print("\n=== 可视化训练过程 ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(lstm_losses, label='LSTM')
axes[0].plot(gru_losses, label='GRU')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(lstm_accs, label='LSTM')
axes[1].plot(gru_accs, label='GRU')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('训练准确率')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 模型预测
print("\n=== 模型预测 ===")

def predict(model, text, vocab, max_length=20):
    """预测单条文本"""
    model.eval()
    indices = text_to_indices(text, vocab)

    # 填充或截断
    if len(indices) < max_length:
        indices = indices + [vocab['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]

    with torch.no_grad():
        x = torch.tensor([indices])
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    return pred, confidence

# 测试预测
test_texts = [
    "这部电影真的很棒，推荐观看",
    "质量太差，完全不推荐"
]

print("LSTM模型预测:")
for text in test_texts:
    pred, conf = predict(lstm_model, text, vocab_dict)
    label = "正面" if pred == 1 else "负面"
    print(f"  文本: {text}")
    print(f"  预测: {label}, 置信度: {conf:.4f}\n")

# 对比LSTM和GRU
print("\n=== LSTM vs GRU 对比 ===")

lstm_correct = 0
gru_correct = 0

lstm_model.eval()
gru_model.eval()

with torch.no_grad():
    for text, label in zip(texts, labels):
        # LSTM预测
        pred_lstm, _ = predict(lstm_model, text, vocab_dict)
        lstm_correct += (pred_lstm == label)

        # GRU预测
        pred_gru, _ = predict(gru_model, text, vocab_dict)
        gru_correct += (pred_gru == label)

print(f"LSTM准确率: {lstm_correct/len(labels)*100:.2f}%")
print(f"GRU准确率: {gru_correct/len(labels)*100:.2f}%")

# 保存和加载模型
print("\n=== 保存和加载模型 ===")

torch.save(lstm_model.state_dict(), 'lstm_text_classifier.pth')
print("LSTM模型已保存")

# 加载模型
loaded_model = TextClassificationLSTM(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT
)
loaded_model.load_state_dict(torch.load('lstm_text_classifier.pth'))
loaded_model.eval()
print("模型已加载")

print("\n学习完成！")