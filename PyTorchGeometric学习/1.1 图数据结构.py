# PyTorch Geometric图数据结构学习
# 主要内容：Graph、Data对象、图操作

import torch
from torch_geometric.data import Data

print("=== 创建图数据 ===")
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(f"图数据: {data}")
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"特征维度: {data.num_features}")

print("\n=== 图数据属性 ===")
print(f"是否有孤立节点: {data.has_isolated_nodes()}")
print(f"是否有自环: {data.has_self_loops()}")
print(f"是否是无向图: {data.is_undirected()}")

print("\n=== 创建带边属性的图 ===")
edge_attr = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
print(f"带边属性的图: {data}")

print("\n=== 图可视化 ===")
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edges_from([(0, 1), (1, 2)])
    
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.title('简单图')
    plt.show()
except ImportError:
    print("networkx未安装，跳过可视化")