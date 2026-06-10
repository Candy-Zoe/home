# DGL图神经网络进阶学习
# 主要内容：消息传递、图采样、异构图、图transformer

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import citation_graph, karate_dataset
import matplotlib.pyplot as plt

print("=== 加载数据集 ===")
g = dgl.data.karate.KarateClubDataset()[0]
print(f"节点数: {g.num_nodes()}")
print(f"边数: {g.num_edges()}")

print("\n=== 定义消息传递函数 ===")
def message_func(edges):
    return {'msg': edges.src['h']}

def reduce_func(nodes):
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = dgl.nn.SAGEConv(in_dim, hidden_dim, aggregator_type='mean')
        self.conv2 = dgl.nn.SAGEConv(hidden_dim, out_dim, aggregator_type='mean')
    
    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

print("GraphSAGE模型已定义")

print("\n=== 图注意力网络(GAT) ===")
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4):
        super().__init__()
        self.conv1 = dgl.nn.GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.conv2 = dgl.nn.GATConv(hidden_dim * num_heads, out_dim, num_heads=1)
    
    def forward(self, g, features):
        h = self.conv1(g, features)
        h = h.flatten(1)
        h = F.elu(h)
        h = self.conv2(g, h)
        h = h.squeeze(1)
        return h

print("GAT模型已定义")

print("\n=== 异构图 ===")
import dgl.function as fn

# 创建异构图
hetero_graph = dgl.heterograph({
    ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('user', 'plays', 'game'): (torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])),
    ('game', 'played-by', 'user'): (torch.tensor([0, 1]), torch.tensor([0, 0, 1]))
})

print(f"异构图节点类型: {hetero_graph.ntypes}")
print(f"异构图边类型: {hetero_graph.etypes}")

print("\n=== 子图采样 ===")
sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5])
dataloader = dgl.dataloading.DataLoader(
    hetero_graph,
    {'user': torch.tensor([0, 1])},
    sampler,
    batch_size=2,
    shuffle=True,
    drop_last=False
)

print("子图采样器已创建")

print("\n=== 图卷积网络(GCN) ===")
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dgl.nn.GraphConv(hidden_dim, out_dim)
    
    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

print("GCN模型已定义")

print("\n=== 图 pooling 操作 ===")
class GlobalAttentionPooling(nn.Module):
    def __init__(self, gate_nn, feat_nn):
        super().__init__()
        self.gate = gate_nn
        self.feat = feat_nn
    
    def forward(self, g, features):
        with g.local_scope():
            g.ndata['h'] = self.feat(features)
            g.ndata['gate'] = self.gate(features)
            return dgl.nn.global_attention(g, g.ndata['gate'])

print("注意力池化已定义")

print("\n=== 训练图神经网络 ===")
features = torch.randn(g.num_nodes(), 10)
labels = torch.randint(0, 4, (g.num_nodes(),))
model = GraphSAGE(10, 16, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    logits = model(g, features)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

print("\n=== 图可视化 ===")
import networkx as nx

nx_g = g.to_networkx()
pos = nx.spring_layout(nx_g)
nx.draw(nx_g, pos, with_labels=True)
plt.title('Karate Club图结构')
plt.show()