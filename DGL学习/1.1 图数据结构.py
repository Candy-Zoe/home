# DGL图数据结构学习
# 主要内容：Graph对象、节点和边操作

import dgl
import torch

print("=== 创建图 ===")
g = dgl.graph(([0, 1, 1, 2], [1, 0, 2, 1]))
print(f"图: {g}")
print(f"节点数: {g.num_nodes()}")
print(f"边数: {g.num_edges()}")

print("\n=== 添加节点特征 ===")
g.ndata['feat'] = torch.randn(3, 5)
print(f"节点特征形状: {g.ndata['feat'].shape}")
print(f"节点特征:\n{g.ndata['feat']}")

print("\n=== 添加边特征 ===")
g.edata['weight'] = torch.randn(4, 1)
print(f"边特征形状: {g.edata['weight'].shape}")
print(f"边特征:\n{g.edata['weight']}")

print("\n=== 图结构操作 ===")
print(f"节点0的出边: {g.out_edges(0)}")
print(f"节点1的入边: {g.in_edges(1)}")
print(f"节点0的出度: {g.out_degree(0)}")
print(f"节点1的入度: {g.in_degree(1)}")

print("\n=== 图可视化 ===")
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    
    nx_g = dgl.to_networkx(g)
    nx.draw(nx_g, with_labels=True, node_color='lightblue')
    plt.title('DGL图')
    plt.show()
except ImportError:
    print("networkx未安装，跳过可视化")

print("\n=== 批量图 ===")
g1 = dgl.graph(([0, 1], [1, 2]))
g2 = dgl.graph(([0, 1, 2], [1, 2, 0]))
bg = dgl.batch([g1, g2])
print(f"批量图节点数: {bg.num_nodes()}")
print(f"批量图边数: {bg.num_edges()}")