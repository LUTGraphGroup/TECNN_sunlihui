import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix



# g: DGLGraph 对象，表示一个图结构。DGL 是深度学习图神经网络的一个库。
# pos_enc_dim: 位置编码的维度，表示我们希望计算出的拉普拉斯位置编码的特征向量数量
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adjacency_matrix(transpose=False, scipy_fmt="csr").astype(float)
    # 使用 DGL 的 adjacency_matrix_scipy 方法获得图的邻接矩阵 A，返回的是一个 SciPy 的稀疏矩阵表示。astype(float) 将其转换为浮点数类型。

    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    # 计算节点度数的逆平方根对角矩阵:

    L = sp.eye(g.number_of_nodes()) - N * A * N
    # 计算归一化的拉普拉斯矩阵

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    # 计算拉普拉斯矩阵的特征值和特征向量

    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # 排序特征值并选择相应的特征向量:

    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    # 提取位置编码:

    return lap_pos_enc



# adj：邻接矩阵，通常是一个二维的方阵 (N x N)，其中 N 是节点的数量。adj[i][j] 表示节点 i 和节点 j 之间的连接关系
# features：特征矩阵，形状为 (N x d)，其中 N 是节点的数量，d 是每个节点的特征维度
# K：传播步数，指示了要进行特征传播的次数
def re_features(adj, features, K):
    #传播之后的特征矩阵,size= (N, 1, K+1, d )
    # 这个张量将存储每个节点在每一步传播之后的特征。
    # 其形状为 (N, 1, K+1, d)，表示 N 个节点，每个节点在 K+1 个步骤中的特征，其中 d 是特征的维度。
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    # 将每个节点的初始特征（即 features 中的第 0 列）放入 nodes_features 的对应位置，即 nodes_features[:, 0, 0, :]
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]


    # 这里 x 是一个临时变量，用来存储每一轮传播后的特征。torch.zeros_like(features) 用来生成一个与 features 形状相同的零矩阵，
    # 以便与 features 相加，确保 x 是一个新的张量，不影响原始 features 的数据
    x = features + torch.zeros_like(features)


    # 进行特征传播:
    # 在每一轮循环中，当前的特征 x 通过邻接矩阵 adj 进行传播，计算公式是 x = torch.matmul(adj, x)，
    # 即将 adj 和 x 进行矩阵乘法。这表示的是一次特征传播，邻接矩阵 adj 作为传播权重 将特征传播到相邻节点上
    for i in range(K):

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]        

    # 最后，将 nodes_features 中多余的维度移除，使其成为一个更紧凑的张量。
    # squeeze 函数会移除张量中大小为1的维度，最终 nodes_features 的形状将是 (N, K+1, d)。
    nodes_features = nodes_features.squeeze()


    return nodes_features




def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix




