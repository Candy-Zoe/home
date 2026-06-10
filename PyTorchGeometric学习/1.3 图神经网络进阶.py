# PyTorch Geometric图神经网络进阶学习
# 主要内容：图注意力网络、图卷积网络进阶、图采样、异构图

import torch
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,pool
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

print("=== 加载数据集 ===")
dataset = KarateClub()
data = dataset[0]
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"特征维度: {data.num_node_features}")

print("\n=== 图注意力网络(GAT) ===")
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GATNet(data.num_node_features, 8, dataset.num_classes)
print("GAT模型已创建")

print("\n=== GraphSAGE网络 ===")
class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model_sage = SAGENet(data.num_node_features, 16, dataset.num_classes)
print("GraphSAGE模型已创建")

print("\n=== 图池化操作 ===")
class GNNWithPooling(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool = pool.global_mean_pool
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)
        return x

print("带池化的GNN已创建")

print("\n=== 图分割 ===")
from torch_geometric.loader import DataLoader

dataset_split = dataset.shuffle()
train_dataset = dataset_split[:int(len(dataset_split) * 0.8)]
test_dataset = dataset_split[int(len(dataset_split) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

print("\n=== 节点分类训练 ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATNet(data.num_node_features, 8, dataset.num_classes).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        model.eval()
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

print("\n=== 链接预测 ===")
from torch_geometric.nn import MLPPedge
from sklearn.metrics import roc_auc_score

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = MLPPedge(in_channels, hidden_channels, out_channels)
    
    def forward(self, x, edge_label_index):
        return self.mlp(x, edge_label_index)

print("链接预测模型已创建")