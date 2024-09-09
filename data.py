import utils
import dgl
import torch
import pandas as pd
from tools import *
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import networkx as nx
# import os.pathpip
import os
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import numpy as np
from sklearn.preprocessing import StandardScaler




def get_train_dataset(pe_dim):

    G = load_graph('./dataset/Networks/training/Train_1000_41.txt')

    # 转换为邻接矩阵
    adj = nx_graph_to_adjacency_matrix(G)

    Label = pd.read_csv('./dataset/SIR results/Train_1000_4/BA_1000_4.csv')
    label = label_sort(Label)

    # 将邻接矩阵转换为稀疏的COO格式
    co_adj = sp.coo_matrix(adj)
    # 使用dgl.from_scipy_sparse_matrix创建DGL图对象
    g_dgl = dgl.from_scipy(co_adj)
    # # 这行代码使用 DGL 库的 to_bidirected 方法将图变为无向图，即在所有的边上添加反向边，使得图中每对连接的节点之间都有双向边。
    G_dgl = dgl.to_bidirected(g_dgl)

    dc = dict(G.degree())
    clustering_coefficient = nx.clustering(G)
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)
    # ks = nx.core_number(G)
    pagerank = nx.pagerank(G)

    # 合并五种中心性特征到一个字典
    feature = {}
    for node in G.nodes():
        feature[node] = [
            dc[node],
            clustering_coefficient[node],
            bc[node],
            cc[node],
            pagerank[node]
        ]

    features = feature_sort(feature)

    # 计算位置编码，G_dgl是目标网络，pe_dim是维度
    lpe = utils.laplacian_positional_encoding(G_dgl, pe_dim)

    # 中心性特征+位置编码特征 输入到编码器
    features = torch.cat((features, lpe), dim=1)

    # features = torch.tensor(features, dtype=torch.float32)
    features = features.clone().detach().float()
    labels = torch.tensor(label)

    adj = utils.sparse_mx_to_torch_sparse_tensor(co_adj)

    return adj, features, labels


def get_test_dataset(pe_dim):

###########
    file_path = './dataset/data/PGP.xlsx'
    # 读取 Excel 文件
    df = pd.read_excel(file_path, header=None, index_col=None)
    adj = df.to_numpy()
    # 将 DataFrame 转换为 numpy 数组作为邻接矩阵
    # adj = df.values
# 使用 NetworkX 将 DataFrame 转换为图
    G = nx.from_numpy_array(adj)
################################

    # 将邻接矩阵转换为稀疏的COO格式
    co_adj = sp.coo_matrix(adj)
    # 使用dgl.from_scipy_sparse_matrix创建DGL图对象
    g_dgl = dgl.from_scipy(co_adj)
    # # 这行代码使用 DGL 库的 to_bidirected 方法将图变为无向图，即在所有的边上添加反向边，使得图中每对连接的节点之间都有双向边。
    G_dgl = dgl.to_bidirected(g_dgl)


    dc = dict(G.degree())
    clustering_coefficient = nx.clustering(G)
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)
    # ks = nx.core_number(G)
    pagerank = nx.pagerank(G)

    # 合并五种中心性特征到一个字典
    feature = {}
    for node in G.nodes():
        feature[node] = [
            dc[node],
            clustering_coefficient[node],
            bc[node],
            cc[node],
            # ks[node],
            pagerank[node]
        ]


    features = feature_sort(feature)

    lpe = utils.laplacian_positional_encoding(G_dgl, pe_dim)
    features = torch.cat((features, lpe), dim=1)

    features = torch.tensor(features, dtype=torch.float32)

    adj = utils.sparse_mx_to_torch_sparse_tensor(co_adj)

    return adj, features




