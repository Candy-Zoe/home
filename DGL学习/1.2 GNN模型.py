# DGL GNN模型学习
# 主要内容：GCN、GAT、GraphSAGE

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv, SAGEConv

print("=== 创建示例图 ===")
g = dgl.graph(([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]))
g.ndata['feat'] = torch.randn(4, 5)

print("\n=== GCN模型 ===")
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
    
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

gcn_model = GCN(in_feats=5, hidden_size=16, num_classes=2)
output = gcn_model(g, g.ndata['feat'])
print(f"GCN输出形状: {output.shape}")

print("\n=== GAT模型 ===")
class GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=1)
        self.conv2 = GATConv(hidden_size, num_classes, num_heads=1)
    
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

gat_model = GAT(in_feats=5, hidden_size=16, num_classes=2)
output = gat_model(g, g.ndata['feat'])
print(f"GAT输出形状: {output.shape}")

print("\n=== GraphSAGE模型 ===")
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, num_classes, aggregator_type='mean')
    
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

sage_model = GraphSAGE(in_feats=5, hidden_size=16, num_classes=2)
output = sage_model(g, g.ndata['feat'])
print(f"GraphSAGE输出形状: {output.shape}")