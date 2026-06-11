# DGL图数据结构学习
# 主要内容：DGLGraph创建、图操作、节点和边特征

# 导入必要的库
import dgl
import torch
import numpy as np

# 创建DGL图
print("=== 创建DGL图 ===")

# 创建一个简单的有向图
# 边列表: [(0, 1), (1, 2), (2, 3), (3, 0)]
u = torch.tensor([0, 1, 2, 3])
v = torch.tensor([1, 2, 3, 0])

g = dgl.graph((u, v))
print(f"图的类型: {type(g)}")
print(f"图的节点数: {g.num_nodes()}")
print(f"图的边数: {g.num_edges()}")
print(f"图是否有向: {g.is_directed()}")

# 创建无向图
g_undirected = dgl.to_bidirected(g)
print(f"\n无向图边数: {g_undirected.num_edges()}")

# 从邻接矩阵创建图
print("\n从邻接矩阵创建图:")
adj_matrix = torch.tensor([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
g_from_adj = dgl.from_scipy(sparse.csr_matrix(adj_matrix))
print(f"从邻接矩阵创建的图节点数: {g_from_adj.num_nodes()}")

# 添加节点和边特征
print("\n=== 添加节点和边特征 ===")

# 设置节点特征
g.ndata['feat'] = torch.randn(4, 5)  # 4个节点，每个节点5维特征
g.ndata['label'] = torch.tensor([0, 1, 0, 1])

# 设置边特征
g.edata['weight'] = torch.rand(g.num_edges())
g.edata['type'] = torch.tensor([0, 1, 0, 1])

print(f"节点特征: {g.ndata['feat'].shape}")
print(f"节点标签: {g.ndata['label']}")
print(f"边权重: {g.edata['weight']}")
print(f"边类型: {g.edata['type']}")

# 图操作
print("\n=== 图操作 ===")

# 获取出边
out_edges = g.out_edges(0)
print(f"节点0的出边: {out_edges}")

# 获取入边
in_edges = g.in_edges(0)
print(f"节点0的入边: {in_edges}")

# 获取邻接节点
neighbors = g.successors(0)
print(f"节点0的后继节点: {neighbors}")

# 图的度数
out_degree = g.out_degrees()
in_degree = g.in_degrees()
print(f"出度: {out_degree}")
print(f"入度: {in_degree}")

# 图的可视化
print("\n=== 图的可视化 ===")

try:
    import networkx as nx
    import matplotlib.pyplot as plt

    # 转换为networkx图
    nx_g = dgl.to_networkx(g)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(nx_g)
    nx.draw(nx_g, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=12, arrows=True)
    plt.title('DGL图可视化')
    plt.show()
except ImportError:
    print("需要安装networkx进行可视化")

# 批量图操作
print("\n=== 批量图操作 ===")

# 创建多个图
g1 = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
g2 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))

# 合并多个图
batch_g = dgl.batch([g1, g2])
print(f"批量图节点数: {batch_g.num_nodes()}")
print(f"批量图边数: {batch_g.num_edges()}")

# 设置批量图的节点特征
batch_g.ndata['feat'] = torch.randn(batch_g.num_nodes(), 3)
print(f"批量图节点特征形状: {batch_g.ndata['feat'].shape}")

# 分割批量图
graphs = dgl.unbatch(batch_g)
print(f"分割后的图数量: {len(graphs)}")

# 子图提取
print("\n=== 子图提取 ===")

# 提取节点子集的子图
subg = g.subgraph(torch.tensor([0, 1, 2]))
print(f"子图节点数: {subg.num_nodes()}")
print(f"子图边数: {subg.num_edges()}")

# 提取边子集的子图
edge_subg = g.edge_subgraph(torch.tensor([0, 1]))
print(f"边子图节点数: {edge_subg.num_nodes()}")
print(f"边子图边数: {edge_subg.num_edges()}")

# 图的属性检查
print("\n=== 图属性检查 ===")

print(f"图是否有自环: {g.has_self_loop()}")
print(f"图是否是简单图: {g.is_simple()}")

# 添加自环
g_with_self_loop = dgl.add_self_loop(g)
print(f"添加自环后的边数: {g_with_self_loop.num_edges()}")

# 删除自环
g_no_self_loop = dgl.remove_self_loop(g_with_self_loop)
print(f"删除自环后的边数: {g_no_self_loop.num_edges()}")

# 图的转换
print("\n=== 图的转换 ===")

# 转换为无向图
undirected_g = dgl.to_bidirected(g)
print(f"无向图边数: {undirected_g.num_edges()}")

# 转换为异构图（如果需要）
print("\n图数据结构学习完成！")