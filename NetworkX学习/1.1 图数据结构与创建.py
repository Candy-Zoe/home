# NetworkX图与网络分析学习
# 主要内容：图创建、图算法、最短路径、网络分析、可视化

# 导入NetworkX库
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 创建图
print("=== 创建图 ===")

# 创建无向图
G = nx.Graph()
print(f"无向图创建完成，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

# 创建有向图
D = nx.DiGraph()
print(f"有向图创建完成")

# 添加节点
print("\n=== 添加节点 ===")

# 添加单个节点
G.add_node(1)
print(f"添加单个节点后，节点数: {G.number_of_nodes()}")

# 添加多个节点
G.add_nodes_from([2, 3, 4, 5])
print(f"添加多个节点后，节点数: {G.number_of_nodes()}")

# 添加带属性的节点
G.add_node("A", type="category", value=10)
G.add_node("B", type="category", value=20)
G.add_node("C", type="category", value=30)
print(f"添加属性节点后，节点数: {G.number_of_nodes()}")

# 添加边
print("\n=== 添加边 ===")

# 添加单条边
G.add_edge(1, 2)
print(f"添加单条边后，边数: {G.number_of_edges()}")

# 添加多条边
G.add_edges_from([(1, 3), (2, 4), (3, 4), (4, 5)])
print(f"添加多条边后，边数: {G.number_of_edges()}")

# 添加带属性的边
G.add_edge("A", "B", weight=5.0)
G.add_edge("B", "C", weight=3.0)
G.add_edge("A", "C", weight=2.0)
print(f"添加属性边后，边数: {G.number_of_edges()}")

# 查看图信息
print("\n=== 查看图信息 ===")

print(f"所有节点: {list(G.nodes())}")
print(f"所有边: {list(G.edges())}")
print(f"节点数量: {G.number_of_nodes()}")
print(f"边数量: {G.number_of_edges()}")
print(f"所有邻居 (节点2): {list(G.neighbors(2))}")

# 获取节点属性
print("\n=== 节点属性 ===")
print(f"节点A的属性: {G.nodes['A']}")
print(f"节点B的属性: {dict(G.nodes['B'])}")

# 获取边属性
print("\n=== 边属性 ===")
print(f"边(A,B)的属性: {G.edges['A', 'B']}")

# 最短路径算法
print("\n=== 最短路径算法 ===")

# 创建示例图
G_shortest = nx.Graph()
G_shortest.add_edges_from([
    (1, 2, {'weight': 4}),
    (1, 3, {'weight': 2}),
    (2, 3, {'weight': 1}),
    (2, 4, {'weight': 5}),
    (3, 4, {'weight': 8}),
    (3, 5, {'weight': 10}),
    (4, 5, {'weight': 2}),
    (4, 6, {'weight': 3}),
    (5, 6, {'weight': 7})
])

# 无权重最短路径
shortest_path = nx.shortest_path(G_shortest, source=1, target=6)
print(f"从节点1到节点6的无权重最短路径: {shortest_path}")

# 带权重最短路径
weighted_path = nx.shortest_path(G_shortest, source=1, target=6, weight='weight')
print(f"从节点1到节点6的带权重最短路径: {weighted_path}")

# 计算所有节点对之间的距离
all_pairs = dict(nx.all_pairs_shortest_path(G_shortest))
print(f"\n所有节点对的最短路径长度:")
for source in sorted(all_pairs.keys()):
    for target in sorted(all_pairs[source].keys()):
        if source < target:
            path = all_pairs[source][target]
            length = len(path) - 1
            print(f"  {source} -> {target}: {path} (长度: {length})")

# 图的遍历
print("\n=== 图的遍历 ===")

# BFS广度优先搜索
print("BFS遍历 (从节点1开始):")
bfs_tree = list(nx.bfs_tree(G_shortest, source=1))
print(f"  {' -> '.join(map(str, bfs_tree[:10]))}...")

# DFS深度优先搜索
print("\nDFS遍历 (从节点1开始):")
dfs_tree = list(nx.dfs_preorder_nodes(G_shortest, source=1))
print(f"  {' -> '.join(map(str, dfs_tree))}")

# 连通性分析
print("\n=== 连通性分析 ===")

# 检查是否连通
print(f"图是否连通: {nx.is_connected(G_shortest)}")

# 找出所有连通分量
connected_components = list(nx.connected_components(G_shortest))
print(f"连通分量数量: {len(connected_components)}")
for i, component in enumerate(connected_components, 1):
    print(f"  分量{i}: {sorted(component)}")

# 连通分量个数
num_components = nx.number_connected_components(G_shortest)
print(f"连通分量个数: {num_components}")

# 网络中心性分析
print("\n=== 中心性分析 ===")

# 度中心性
degree_centrality = nx.degree_centrality(G_shortest)
print("度中心性 (每个节点的连接程度):")
for node, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True):
    print(f"  节点 {node}: {centrality:.3f}")

# 介数中心性
betweenness = nx.betweenness_centrality(G_shortest)
print("\n介数中心性 (节点作为桥梁的重要性):")
for node, centrality in sorted(betweenness.items(), key=lambda x: x[1], reverse=True):
    print(f"  节点 {node}: {centrality:.3f}")

# 接近中心性
closeness = nx.closeness_centrality(G_shortest)
print("\n接近中心性 (节点到其他节点的平均距离):")
for node, centrality in sorted(closeness.items(), key=lambda x: x[1], reverse=True):
    print(f"  节点 {node}: {centrality:.3f}")

# PageRank算法
print("\n=== PageRank算法 ===")

# 创建示例有向图
D_pagerank = nx.DiGraph()
D_pagerank.add_edges_from([
    (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 3),
    (3, 1), (3, 4),
    (4, 1), (4, 3),
    (5, 1), (5, 4),
    (6, 4)
])

# 计算PageRank
pagerank = nx.pagerank(D_pagerank, alpha=0.85)
print("PageRank得分 (节点重要性):")
for node, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
    print(f"  节点 {node}: {score:.4f}")

# 社区检测
print("\n=== 社区检测 ===")

# 创建示例图用于社区检测
G_community = nx.karate_club_graph()

# Louvain社区检测算法
try:
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G_community)
    print(f"检测到的社区数量: {len(communities)}")
    for i, community in enumerate(communities, 1):
        print(f"  社区 {i}: {sorted(community)[:10]}..." if len(community) > 10 else f"  社区 {i}: {sorted(community)}")
except:
    print("Louvain算法需要额外安装 python-louvain")

# Girvan-Newman算法
from networkx.algorithms.community import girvan_newman
gn = girvan_newman(G_community)
communities = next(gn)
print(f"\nGirvan-Newman社区检测 (2个社区):")
for i, community in enumerate(communities, 1):
    print(f"  社区 {i}: {sorted(list(community))[:10]}..." if len(community) > 10 else f"  社区 {i}: {sorted(list(community))}")

# 图的生成器
print("\n=== 图生成器 ===")

# 完全图 (所有节点都相连)
K_n = nx.complete_graph(5)
print(f"完全图K5的边数: {K_n.number_of_edges()}")

# 环形图
C_n = nx.cycle_graph(8)
print(f"环形图C8的边数: {C_n.number_of_edges()}")

# 网格图
grid = nx.grid_2d_graph(3, 4)
print(f"3x4网格图的节点数: {grid.number_of_nodes()}, 边数: {grid.number_of_edges()}")

# 随机图
G_random = nx.gnm_random_graph(20, 30, seed=42)
print(f"随机图 (20节点, 30边) 的平均度: {sum(dict(G_random.degree()).values()) / 20:.2f}")

# 小世界网络
WS = nx.watts_strogatz_graph(30, 5, 0.1, seed=42)
print(f"小世界网络的聚类系数: {nx.average_clustering(WS):.3f}")

# 无标度网络 (Barabási-Albert模型)
BA = nx.barabasi_albert_graph(100, 3, seed=42)
print(f"无标度网络的节点数: {BA.number_of_nodes()}, 边数: {BA.number_of_edges()}")

# 图算法
print("\n=== 图算法 ===")

# 最小生成树
G_mst = nx.Graph()
G_mst.add_weighted_edges_from([
    ('A', 'B', 1), ('A', 'C', 4), ('B', 'C', 2),
    ('B', 'D', 5), ('C', 'D', 3), ('C', 'E', 6),
    ('D', 'E', 7)
])
mst = nx.minimum_spanning_tree(G_mst)
print("最小生成树:")
for edge in mst.edges(data=True):
    print(f"  {edge[0]} - {edge[1]}: {edge[2]['weight']}")

# 欧拉路径
G_euler = nx.eulerian_graph = nx.Graph()
G_euler.add_edges_from([(1,2), (1,3), (2,3), (2,4), (3,4)])
print(f"\n图中是否存在欧拉路径: {nx.has_eulerian_path(G_euler)}")

# 图的可视化
print("\n=== 图可视化 ===")

# 创建用于可视化的图
G_vis = nx.complete_graph(5)
pos = nx.spring_layout(G_vis, seed=42)

# 设置图形
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
nx.draw(G_vis, pos, with_labels=True, node_color='lightblue',
        node_size=700, font_size=16, font_weight='bold', ax=ax)
plt.title('完全图K5可视化')
plt.tight_layout()
plt.show()

# 可视化带属性的图
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
pos = nx.spring_layout(G_shortest, seed=42)
edge_labels = nx.get_edge_attributes(G_shortest, 'weight')
nx.draw(G_shortest, pos, with_labels=True, node_color='lightgreen',
        node_size=700, font_size=14, font_weight='bold', ax=ax)
nx.draw_networkx_edge_labels(G_shortest, pos, edge_labels=edge_labels, ax=ax)
plt.title('带权重的图')
plt.tight_layout()
plt.show()

# 可视化社区结构
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
pos = nx.spring_layout(G_community, seed=42)
gn_communities = next(girvan_newman(G_community))
colors = ['red', 'blue', 'green', 'yellow']
for i, community in enumerate(gn_communities):
    nx.draw_networkx_nodes(G_community, pos, nodelist=list(community),
                           node_color=colors[i % len(colors)], ax=ax)
nx.draw_networkx_edges(G_community, pos, alpha=0.5, ax=ax)
nx.draw_networkx_labels(G_community, pos, font_size=10, ax=ax)
plt.title('社区检测可视化')
plt.tight_layout()
plt.show()

print("\nNetworkX图与网络分析学习完成！")
