# NetworkX图算法学习
# 主要内容：最短路径、网络流、匹配算法、拓扑排序

# 导入NetworkX库
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 创建示例图
print("=== 创建示例图 ===")

# 创建带权重的无向图
G = nx.Graph()
G.add_weighted_edges_from([
    ('A', 'B', 4), ('A', 'C', 2), ('A', 'D', 5),
    ('B', 'C', 1), ('B', 'E', 3), ('C', 'D', 3),
    ('C', 'E', 4), ('D', 'F', 2), ('E', 'F', 6),
    ('E', 'G', 5), ('F', 'G', 1)
])

print(f"图创建完成，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

# 最短路径算法
print("\n=== 最短路径算法 ===")

# 迪杰斯特拉算法
print("迪杰斯特拉算法 (最短路径):")
shortest_path = nx.dijkstra_path(G, source='A', target='G')
shortest_length = nx.dijkstra_path_length(G, source='A', target='G')
print(f"  从A到G的最短路径: {' -> '.join(shortest_path)}")
print(f"  最短路径长度: {shortest_length}")

# 所有节点对的最短路径
print("\n从A到所有节点的最短路径:")
all_shortest = nx.single_source_dijkstra(G, source='A')
for target in sorted(G.nodes()):
    if target != 'A':
        path, length = all_shortest[0][target], all_shortest[1][target]
        print(f"  A -> {target}: {' -> '.join(path)} (长度: {length})")

# 贝尔曼-福特算法
print("\n贝尔曼-福特算法:")
try:
    distance, path = nx.bellman_ford_path(G, source='A', target='G')
    print(f"  从A到G的最短路径: {' -> '.join(path)}")
    print(f"  最短路径长度: {distance}")
except nx.NetworkXUnbounded:
    print("  检测到负权重环")

# Floyd-Warshall算法 (所有节点对之间的最短路径)
print("\nFloyd-Warshall算法:")
all_pairs_length = dict(nx.floyd_warshall(G))
print("所有节点对之间的最短路径长度:")
for source in sorted(all_pairs_length.keys()):
    for target in sorted(all_pairs_length[source].keys()):
        if source < target:
            length = all_pairs_length[source][target]
            print(f"  {source} -> {target}: {length}")

# 生成树算法
print("\n=== 生成树算法 ===")

# 最小生成树
print("最小生成树 (Prim算法):")
mst = nx.minimum_spanning_tree(G)
mst_edges = list(mst.edges(data=True))
print(f"  MST边数: {len(mst_edges)}")
for edge in mst_edges:
    print(f"    {edge[0]} - {edge[1]}: 权重 {edge[2]['weight']}")

# 最大生成树
print("\n最大生成树:")
max_st = nx.maximum_spanning_tree(G)
max_st_edges = list(max_st.edges(data=True))
print(f"  最大ST边数: {len(max_st_edges)}")
for edge in max_st_edges:
    print(f"    {edge[0]} - {edge[1]}: 权重 {edge[2]['weight']}")

# 网络流算法
print("\n=== 网络流算法 ===")

# 创建有向网络流图
D = nx.DiGraph()
D.add_weighted_edges_from([
    ('S', 'A', 10), ('S', 'B', 8), ('S', 'C', 5),
    ('A', 'B', 2), ('A', 'D', 8), ('A', 'E', 4),
    ('B', 'C', 3), ('B', 'D', 5), ('B', 'E', 6),
    ('C', 'E', 4), ('D', 'T', 10), ('E', 'T', 12),
    ('D', 'E', 3)
])

# 设置源点和汇点
S = 'S'  # 源点 (Source)
T = 'T'  # 汇点 (Target)

# 最大流算法
print(f"从{S}到{T}的最大流:")
max_flow_value, max_flow_dict = nx.maximum_flow(D, S, T)
print(f"  最大流值: {max_flow_value}")
print("  各边的流量:")
for u in max_flow_dict:
    for v in max_flow_dict[u]:
        if max_flow_dict[u][v] > 0:
            print(f"    {u} -> {v}: {max_flow_dict[u][v]}")

# 最小割
print(f"\n最小割:")
min_cut_value, partition = nx.minimum_cut(D, S, T)
reachable, non_reachable = partition
print(f"  最小割值: {min_cut_value}")
print(f"  可达集: {reachable}")
print(f"  不可达集: {non_reachable}")

# 最小成本最大流
print("\n最小成本最大流:")
D_cost = nx.DiGraph()
D_cost.add_weighted_edges_from([
    ('S', 'A', 2), ('S', 'B', 1), ('A', 'B', 1),
    ('A', 'C', 3), ('B', 'C', 2), ('A', 'T', 4),
    ('B', 'T', 5), ('C', 'T', 6)
])
# 设置容量下界
flow_dict = nx.min_cost_flow_cost(D_cost)
print(f"  最小成本流: {flow_dict}")

# 匹配算法
print("\n=== 匹配算法 ===")

# 创建二部图
B = nx.Graph()
B.add_nodes_from(['A1', 'A2', 'A3'], bipartite=0)  # 左边的集合
B.add_nodes_from(['B1', 'B2', 'B3', 'B4'], bipartite=1)  # 右边的集合
B.add_edges_from([('A1', 'B1'), ('A1', 'B2'), ('A2', 'B2'),
                  ('A2', 'B3'), ('A3', 'B3'), ('A3', 'B4')])

# 最大匹配
print("二部图最大匹配:")
max_matching = nx.maximum_matching(B)
print(f"  最大匹配边数: {len(max_matching) // 2}")
for edge in max_matching:
    print(f"    {edge[0]} - {edge[1]}")

# Hopcroft-Karp算法
print("\nHopcroft-Karp算法:")
try:
    matching = nx.bipartite.maximum_matching(B)
    print(f"  匹配结果: {matching}")
except:
    print("  需要指定二部图集合")

# 完美匹配检查
print(f"\n是否存在完美匹配: {nx.is_perfect_matching(B, max_matching)}")

# 拓扑排序
print("\n=== 拓扑排序 ===")

# 创建有向无环图 (DAG)
DAG = nx.DiGraph()
DAG.add_edges_from([
    ('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'),
    ('C', 'F'), ('D', 'G'), ('E', 'G'), ('F', 'G')
])

# 拓扑排序
print("拓扑排序结果:")
topo_order = list(nx.topological_sort(DAG))
print(f"  {' -> '.join(topo_order)}")

# 检查是否为DAG
print(f"\n是否为有向无环图 (DAG): {nx.is_directed_acyclic_graph(DAG)}")

# 所有拓扑排序
print("\n所有可能的拓扑排序:")
all_topo_sorts = list(nx.all_topological_sorts(DAG))
for i, sort in enumerate(all_topo_sorts[:5], 1):  # 只显示前5个
    print(f"  {i}. {' -> '.join(sort)}")
if len(all_topo_sorts) > 5:
    print(f"  ... (共 {len(all_topo_sorts)} 种排序)")

# 关键路径分析
print("\n=== 关键路径分析 ===")

# 创建带权重的DAG
DAG_critical = nx.DiGraph()
DAG_critical.add_weighted_edges_from([
    ('A', 'B', 3), ('A', 'C', 2), ('B', 'D', 4), ('B', 'E', 2),
    ('C', 'D', 1), ('C', 'F', 3), ('D', 'G', 2), ('E', 'G', 1),
    ('F', 'G', 3)
])

# 关键路径
print("关键路径 (最长路径):")
try:
    critical_path = nx.dag_longest_path(DAG_critical)
    critical_path_length = nx.dag_longest_path_length(DAG_critical)
    print(f"  关键路径: {' -> '.join(critical_path)}")
    print(f"  关键路径长度: {critical_path_length}")
except:
    print("  DAG中没有简单路径")

# 强连通分量
print("\n=== 强连通分量 ===")

# 创建有向图
D_scc = nx.DiGraph()
D_scc.add_edges_from([
    (1, 2), (2, 3), (3, 1), (3, 4), (4, 5),
    (5, 6), (6, 4), (6, 7), (7, 8), (8, 6)
])

# Tarjan算法找强连通分量
print("Tarjan算法强连通分量:")
sccs = list(nx.strongly_connected_components(D_scc))
for i, scc in enumerate(sccs, 1):
    print(f"  分量 {i}: {sorted(scc)}")

# Kosaraju算法
print("\nKosaraju算法强连通分量:")
sccs_kosaraju = list(nx.kosaraju_strongly_connected_components(D_scc))
for i, scc in enumerate(sccs_kosaraju, 1):
    print(f"  分量 {i}: {sorted(scc)}")

# 可达性分析
print("\n=== 可达性分析 ===")

# 单源可达性
print("从节点A可达的节点:")
reachable_from_A = nx.descendants(DAG, 'A')
print(f"  {sorted(reachable_from_A)}")

# 所有节点对的可达性
print("\n可达性矩阵:")
reachability = nx.transitive_closure(DAG)
for node1 in sorted(DAG.nodes()):
    reachable = [node2 for node2 in DAG.nodes() if node2 in reachability[node1]]
    print(f"  {node1} 可达: {reachable}")

# 图分割
print("\n=== 图分割 ===")

# 创建示例图
G_partition = nx.Graph()
G_partition.add_edges_from([
    (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)
])

# Kernighan-Lin算法进行图分割
print("Kernighan-Lin图分割算法:")
try:
    partition = nx.community.kernighan_lin_bisection(G_partition)
    print(f"  分割结果:")
    for i, part in enumerate(partition, 1):
        print(f"    集合 {i}: {sorted(part)}")
except Exception as e:
    print(f"  算法执行出错: {e}")

# 可视化
print("\n=== 可视化 ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 最短路径可视化
ax1 = axes[0, 0]
pos = nx.spring_layout(G, seed=42)
shortest_edges = [(shortest_path[i], shortest_path[i+1])
                   for i in range(len(shortest_path)-1)]
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=700, font_size=12, ax=ax1)
nx.draw_networkx_edges(G, pos, edgelist=shortest_edges,
                       edge_color='red', width=3, ax=ax1)
ax1.set_title('最短路径 (A -> G)')

# 2. 最小生成树可视化
ax2 = axes[0, 1]
nx.draw(G, pos, with_labels=True, node_color='lightgreen',
        node_size=700, font_size=12, ax=ax2)
nx.draw_networkx_edges(G, pos, edgelist=list(mst.edges()),
                       edge_color='red', width=3, ax=ax2)
ax2.set_title('最小生成树')

# 3. 网络流可视化
ax3 = axes[1, 0]
pos_flow = nx.spring_layout(D, seed=42)
nx.draw(D, pos_flow, with_labels=True, node_color='lightyellow',
        node_size=700, font_size=12, ax=ax3)
edge_labels = {(u, v): f"{d['weight']}" for u, v, d in D.edges(data=True)}
nx.draw_networkx_edge_labels(D, pos_flow, edge_labels=edge_labels, ax=ax3)
ax3.set_title('网络流图')

# 4. 拓扑排序可视化
ax4 = axes[1, 1]
pos_dag = nx.spring_layout(DAG, seed=42)
nx.draw(DAG, pos_dag, with_labels=True, node_color='lightcoral',
        node_size=700, font_size=12, ax=ax4)
# 高亮拓扑排序顺序
node_colors = plt.cm.viridis(np.linspace(0, 1, len(DAG.nodes())))
color_map = {node: node_colors[topo_order.index(node)] for node in DAG.nodes()}
nx.draw(DAG, pos_dag, with_labels=True,
        node_color=[color_map[node] for node in DAG.nodes()],
        node_size=700, font_size=12, ax=ax4)
ax4.set_title('拓扑排序')

plt.tight_layout()
plt.show()

print("\nNetworkX图算法学习完成！")
