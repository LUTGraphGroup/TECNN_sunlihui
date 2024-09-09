from data import get_train_dataset, get_test_dataset
import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from model import TECNN
from test import *

import torch.utils.data as Data
import argparse




# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None) #模型的名称
    parser.add_argument('--dataset', type=str, default='pubmed',
                        help='Choose from {pubmed}') #数据集的名称
    # parser.add_argument('--device', type=int, default=1,
    #                     help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407, 
                        help='Random seed.') #随机种子 默认值为3407



    # model parameters
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated') # 计算的邻居跳数，默认值为7，用于确定图中的邻居层数

    parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size') # 位置编码的维度

    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='Hidden layer size') # 编码器的输入

    parser.add_argument('--ffn_dim', type=int, default=32,
                        help='ffn_dim layer size')  # 前馈层中间的大小

    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers') # Transformer层的数量

    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads') #Transformer头的数量

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout') #Dropout概率，防止过拟合

    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer') #注意力层中的Dropout概率


    # training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size') #批量大小

    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.') #epochs大小

    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling') #优化器学习率调度使用的总更新次数

    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps') #预热步骤的数量，用于学习率调度策略。

    parser.add_argument('--peak_lr', type=float, default=0.001,
                        help='learning rate') #学习率

    parser.add_argument('--end_lr', type=float, default=0.0001,
                        help='learning rate') #结束时的学习率

    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay') #权重衰减，类型为浮点数,用于防止过拟合的正则化技术。

    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping') #提前停止的耐心值

    return parser.parse_args()

args = parse_args()#调用之前定义的参数


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #使用cpu还是GPU
# device = args.device

random.seed(args.seed) #随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def PredictNode(is_train,name):
    """
    is_train:一个布尔值，表示是进行训练('True')还是测试('False')
    name: 用于测试时指定数据集的名称。
    """
   # model configuration
    model = TECNN(
                    hops=args.hops,#几跳邻居
                    n_class=1,
                    pe_dim=args.pe_dim, #位置编码的维度
                    input_dim=args.pe_dim+5, #节点自身的维度加上位置编码的维度
                    n_layers=args.n_layers, #transformer层数
                    num_heads=args.n_heads, #头数
                    hidden_dim=args.hidden_dim, #隐藏层维度
                    ffn_dim=args.ffn_dim, # 前馈层维度
                    dropout_rate=args.dropout,
                    attention_dropout_rate=args.attention_dropout).to(device)

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)

    loss_fun = nn.MSELoss()

    if is_train:
        # load train data
        # Load and pre-process data
        train_adj, train_features, labels = get_train_dataset(args.pe_dim)
        # 聚合几跳邻居
        train_hop_features = utils.re_features(train_adj, train_features, args.hops)  # return (N, hops+1, d)
        labels = labels.to(device)
        # 数据+标签 (一个矩阵对应一个标签)
        batch_data_train = Data.TensorDataset(train_hop_features, labels)
        # 加入batch_size
        train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle=True)

        print("------------train start---------------")
        for epoch in range(args.epochs): #开始一个循环，遍历指定的epoch次数
            # epoch_loss = 0.0 # 用于累计当前epoch的总损失
            # batch_count = 0 # 用于记录当前epoch中的批次数。
            for _, item in enumerate(train_data_loader): #开始内层循环，遍历训练数据加载器（train_data_loader）中的所有批次（batches）
                # 取出节点特征
                nodes_features = item[0].to(device)
                # 取出节点对应的标签
                labels = item[1].to(device)

                nodes_features = nodes_features.float()
                labels = labels.float()
                labels = labels.view(-1, 1)

                optimizer.zero_grad() #将优化器的梯度清零，以便在每次反向传播前清除旧的梯度信息。
                output = model(nodes_features) #将节点特征输入模型，获取输出结果，并将输出转换为浮点型。
                output = output.float()
                loss = loss_fun(output, labels) #计算模型输出与真实标签之间的损失。

                loss.backward(retain_graph=True) #对损失进行反向传播，计算梯度。
                optimizer.step()

                # epoch_loss += loss.item() #累计当前批次的损失值到epoch_loss
                # batch_count += 1

                state = {'net': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch + 1}

                torch.save(state, './dataset/model_train/' + "train_1000" + '.pth')
            # avg_loss = epoch_loss / batch_count # 计算当前epoch的平均损失值。
            print('Epoch:', epoch + 1)
            print('Train Loss: %.4f' % loss.item())
            # print(print('Train Loss: %.4f' % avg_loss))
        print("------------train end---------------")


    else:
        # load test data
        test_adj, test_features = get_test_dataset(args.pe_dim) #加载测试数据

        test_hop_features = utils.re_features(test_adj, test_features, args.hops) #几跳邻居

        # 这段代码加载训练好的模型权重。torch.load函数从指定路径加载模型检查点文件，
        # 然后使用model.load_state_dict将权重加载到当前模型实例中。
        checkpoint = torch.load('./dataset/model_train/' + "train_1000" + '.pth')
        model.load_state_dict(checkpoint['net'])
        # 使用模型对测试数据进行预测
        y_pred = model(test_hop_features)# test_data_loader



        # 评估预测结果
        value = sckendall_test(y_pred, name) #肯德尔系数


        return value



if __name__ == '__main__':
    is_train = False
    # is_train = True
    if is_train:
        # 训练模型
        PredictNode(is_train, None)
    else:
        # 测试模型
        name = "PGP"
        pre = PredictNode(is_train, name)
        print(pre)

