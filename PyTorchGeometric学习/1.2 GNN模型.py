# PyTorch Geometric GNN模型学习
# 主要内容：GCN、GAT、GraphSAGE

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data

print("=== 创建示例图 ===")
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

print("\n=== GCN模型 ===")
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

gcn_model = GCN(num_features=2, hidden_dim=16, num_classes=2)
output = gcn_model(data)
print(f"GCN输出形状: {output.shape}")

print("\n=== GAT模型 ===")
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

gat_model = GAT(num_features=2, hidden_dim=16, num_classes=2)
output = gat_model(data)
print(f"GAT输出形状: {output.shape}")

print("\n=== GraphSAGE模型 ===")
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

sage_model = GraphSAGE(num_features=2, hidden_dim=16, num_classes=2)
output = sage_model(data)
print(f"GraphSAGE输出形状: {output.shape}")