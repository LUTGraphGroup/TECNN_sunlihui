import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)



def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


#####################    我的模型
class TECNN(nn.Module):
    def __init__(
        self,
        hops, 
        n_class,
        pe_dim,
        input_dim,
        n_layers,
        num_heads,
        hidden_dim,
        ffn_dim,
        dropout_rate=0.1,
        attention_dropout_rate=0.1
    ):
        super().__init__()

        self.seq_len = hops+1
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        
        self.n_layers = n_layers
        self.n_class = n_class

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)

        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

   

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))

        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        self.Linear1 = nn.Linear(int(self.hidden_dim/2), self.n_class)

        self.scaling = nn.Parameter(torch.ones(1) * 0.5)


        self.apply(lambda module: init_params(module, n_layers=n_layers))


        # CNN 自己定义的
        self.cnn = CNN()
        self.FC = nn.Linear(532,1)
        # self.fc_xiaorong = nn；、.Linear(160,1)

    def forward(self, batched_data):
        #初始维度经过一层线性层 ，降维
        tensor = self.att_embeddings_nope(batched_data)  #16

        
        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        # 层正则化
        output = self.final_ln(tensor) #16  #64,8,20
#### 消融##############
        # x = output.reshape(output.shape[0], -1)
        # x = self.fc_xiaorong(x) # 64,160
############################################
        #
        # return x


        # target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        # 从output张量中提取特定维度的数据，然后扩展和重复这些数据，以生成新的张量target
        # output[:,0,:]：这个操作从output张量中提取第二个维度（dim=1）的第一个切片。
# 假设output的维度是 (batch_size, seq_len, feature_size)，那么结果的维度是 (batch_size, feature_size)。
        # .unsqueeze(1)：在第二个维度（dim=1）插入一个新的维度。结果的维度变成 (batch_size, 1, feature_size)
        # .repeat(1, self.seq_len-1, 1)：沿着第二个维度重复数据 self.seq_len-1 次。结果的维度变成 (batch_size, self.seq_len-1, feature_size)。


        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)
#         # 将output张量按照指定的维度和大小进行拆分
#         # 拆分后的split_tensor是一个包含两个张量的元组
# #第一个张量的维度是 (batch_size, 1, feature_size)。第二个张量的维度是 (batch_size, self.seq_len-1, feature_size)。
#
#         # 提取出的节点自身的特征
        node_tensor = split_tensor[0]   #1000,1,521
#         # neighbor_tensor = split_tensor[1]   #1000,7,521
#         # 这里的out是编码器出来的 节点自身+几跳邻居
        fea_cnn = self.cnn(output)
#
        fea_cnn = fea_cnn.unsqueeze(1)
#
        x = torch.cat((node_tensor, fea_cnn), dim=2)
#
        x = x.squeeze(1)
#
        x = self.FC(x)

        return x
##############################################################################



############  编码器
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        # 层正则化
        y = self.self_attention_norm(x)  # 1000,8,512
        # 进入多头注意力
        y = self.self_attention(y, y, y, attn_bias)

        # drop
        y = self.self_attention_dropout(y)
        # 残差
        x = x + y



        # 层正则化
        y = self.ffn_norm(x)

        # 前馈层
        y = self.ffn(y)
        # drop
        y = self.ffn_dropout(y)
        # 残差
        x = x + y
        return x
#########################################################################


############### 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        # 获取q的形状维度
        orig_q_size = q.size()  # 1000,8,512

        # dk=投影特征维度 除以 多头数
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k) # 1000.,8,8,64
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        # transpose函数 将指定的两个列互换位置
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale

        # q点乘k
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]  # 1000,8,8,8
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3) ################################### 原本的
        # x = torch.nn.Softmax(dim=3)(x) # 自己加的

        x = self.att_dropout(x)
        # q与k点乘完的值 在点乘 v
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x
########################################################################



################# 前馈层
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)  # 512-to-1024
        x = self.gelu(x)  # 激活函数
        x = self.layer2(x)  # 1024-to-512
        return x




################    二维CNN
class CNN(nn.Module):
    def __init__(self, ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1,padding=2)

        # self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.MaxPool = nn.AdaptiveAvgPool2d((3, 3))
        # self.MaxPool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(576, 512)


    def forward(self, x):
        x = x.unsqueeze(1)
        # x = x.float()  # 避免类型不同报错
        x = F.relu(self.conv1(x)) #(batch size,第一层卷积输出通道，输出维度跟输入维度一样)
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x





