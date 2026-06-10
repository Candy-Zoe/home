# PyTorch循环神经网络学习
# 主要内容：使用PyTorch实现RNN进行序列数据处理

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建序列数据 ===")

def generate_sequence(length=100):
    x = np.linspace(0, 10*np.pi, length)
    y = np.sin(x) + np.random.normal(0, 0.1, length)
    return x, y

x, y = generate_sequence(200)
plt.plot(x, y)
plt.title('正弦波序列数据')
plt.show()

print("\n=== 准备训练数据 ===")
def create_dataset(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(2), torch.tensor(Y, dtype=torch.float32)

seq_length = 10
X, Y = create_dataset(y, seq_length)
print(f"输入形状: {X.shape}, 输出形状: {Y.shape}")

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

print("\n=== 定义RNN模型 ===")

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out

model = RNNModel()
print(model)

print("\n=== 训练模型 ===")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

plt.plot(loss_history)
plt.title('训练损失')
plt.show()

print("\n=== 预测 ===")
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze().numpy()

plt.figure(figsize=(10, 4))
plt.plot(y[train_size+seq_length:], label='真实值')
plt.plot(predictions, label='预测值')
plt.legend()
plt.title('RNN序列预测结果')
plt.show()