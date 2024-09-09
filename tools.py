import networkx as nx
import numpy as np
import torch


def gml_to_adjacency_matrix(gml_path):
    # 读取 GML 文件
    G = nx.read_gml(gml_path)

    # 生成邻接矩阵
    adj_matrix = nx.adjacency_matrix(G).todense()

    # 将邻接矩阵转换为 numpy 数组
    adj_matrix = np.array(adj_matrix)

    # 保存为 NPY 文件
    # np.save(npy_path, adj_matrix)
    return adj_matrix


def txt_to_adjacency_matrix(txt_path):
    # 读取边列表文件
    edges = []
    nodes = set()
    with open(txt_path, 'r') as file:
        for line in file:
            source, target = line.strip().split()
            edges.append((source, target))
            nodes.update([source, target])

    # 创建节点列表并构建节点索引
    nodes = sorted(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    # 初始化邻接矩阵
    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=int)

    # 填充邻接矩阵
    for source, target in edges:
        i, j = node_index[source], node_index[target]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # 若是无向图，则需要这一行；如果是有向图，则注释掉这一行

    return adj_matrix


def feature_sort(data):
    data = dict(sorted(data.items(), key=lambda item: int(item[0])))
    # 将排序后的字典的值转换为包含单独列表的 NumPy 数组
    data = np.array([[v] for v in data.values()])
    # 将 numpy 数组转换为 torch 张量
    feature = torch.from_numpy(data)

    feature = feature.squeeze(1)

    return feature


def label_sort(label):

    label.columns = ['Node', 'Label']
    # 根据节点号对数据进行排序
    label_sorted = label.sort_values(by='Node')
    # 重置索引，以确保排序后的数据索引是连续的
    label_sorted.reset_index(drop=True, inplace=True)

    # 提取排序后的标签（第二列）并转换为 PyTorch 张量
    label_values = label_sorted['Label'].values

    return label_values

def load_graph(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    G = nx.read_edgelist(path, create_using=nx.Graph())
    return G


def nx_graph_to_adjacency_matrix(G):
    """
    将NetworkX图转换为邻接矩阵（numpy数组形式）。

    Parameters:
    - G: NetworkX图对象

    Returns:
    - adjacency_matrix: numpy数组形式的邻接矩阵
    """
    # 获取图的节点数
    num_nodes = G.number_of_nodes()

    # 创建全零的邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 遍历图中的每条边，设置对应的邻接矩阵元素为1
    for edge in G.edges():
        try:
            source = int(edge[0])  # 将节点标识转换为整数
            target = int(edge[1])  # 将节点标识转换为整数
        except ValueError:
            # 如果不能转换为整数，保留原始标识或处理其他逻辑
            source = edge[0]
            target = edge[1]

        adjacency_matrix[source, target] = 1
        adjacency_matrix[target, source] = 1  # 如果是无向图，通常需要设置对称的边为1

    return adjacency_matrix
