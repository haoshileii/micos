import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)
# 定义自己的网络的时候需要继承torch的nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法
class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        # 调用父类的构造函数
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        # 定义全连接层（线性层）可以理解为矩阵运算的函数映射；input_dims： 上层神经元个数【每个输入样本的大小】；hidden_dims： 本层神经元个数【每个输出样本的大小】
        # y=xA(T)+b
        self.input_fc = nn.Linear(input_dims, hidden_dims)#定义全连接层，数据输入之后进行矩阵计算，然后输出
        #第二个参数是一个长度为10的list, 不是一个数组，要注意

        ##########测试LSTM效果
        self.lstm = nn.LSTM(64, 128, 3)
        self.lstmbi = nn.LSTM(64, 64, 3, bidirectional=True)
        self.gru = nn.GRU(64, 128, 3)
        #self.gate = torch.nn.Linear(seq_len*output_dims + seq_len * hidden_dims, 2)


        ###########
        self.input_fc2 = nn.Linear(input_dims, 320)  #
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size = 3
        )
        decoder_layer = TransformerEncoderLayer(d_model=64, nhead=16,
                                                dim_feedforward=320, dropout=0.1)
        self.pee = LearnablePositionalEncoding(d_model=64,)
        self.transformer = TransformerEncoder(decoder_layer, 2)
        #self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.input_x_a = nn.Linear(output_dims+hidden_dims, output_dims+hidden_dims)
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0

        #x2 = self.input_fc2(x).to(x.device)
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x1 = x.transpose(1, 2).to(x.device)  # B x Ch x T
        x1 = self.repr_dropout(self.feature_extractor(x1))  # B x Co x T
        #输入要求：B x Co x T 输出结果；B x Co x T 效果：对每个维度长度上进行平均池化
        x1 = F.avg_pool1d(x1, kernel_size=7, stride=1, padding=3)  # B*co*T
        x1 = x1.transpose(1, 2)# B  x T*co
        #print(x.shape)

        x2 = x.to(x.device)  # B x T*ch

        ######如果用LSTM
        # x2 = x2.transpose(0, 1)#Tx B*ch
        # x2, (hn, cn) = self.lstm(x2)   #Tx B*ch
        # x2 = x2.transpose(0, 1)#B*T*ch
        ######如果用LSTM

        ######如果用BI-LSTM
        # x2 = x2.transpose(0, 1)  # Tx B*ch
        # x2, (hn, cn) = self.lstmbi(x2)  # Tx B*ch
        # x2 = x2.transpose(0, 1)  # B*T*ch
        ######如果用BI-LSTM

        ######如果用GRU
        x2 = x2.transpose(0, 1)  # Tx B*ch
        x2, hn = self.gru(x2)  # Tx B*ch

        x2 = x2.transpose(0, 1)  # B*T*ch
        x2 = x2.transpose(1, 2) # B*ch*T
        x2 = F.avg_pool1d(x2, kernel_size=7, stride=1, padding=3)  # B*ch*T
        x2 = x2.transpose(1, 2)  # B*T*ch
        ######如果用GRU



        ######如果用TRANSFOMER
        # x2 = self.pee(x2)#B x T x Co
        # x2 = self.transformer(x2)#transformer输入要求：B x T x Co
        ######如果用TRANSFOMER
        # x1 = x1.reshape(x1.shape[0], -1)
        # x2 = x2.reshape(x2.shape[0], -1)
        # gate = F.softmax(self.gate(torch.cat([x1, x2], dim=-1)), dim=-1)
        # x1 = x1*gate[:, 0:1]
        # x1 = x1.reshape(x1.shape[0], -1, 320)
        # x2 = x2 * gate[:, 1:2]
        # x2 = x2.reshape(x2.shape[0], -1, 64)

        #x_a = torch.cat([x1, x2], dim=-1).to(x.device)
        #x_a = self.input_x_a(x_a)
        #x = x.transpose(1, 2)  # B x T x Co
        x_a = x1.to(x.device)
        return x_a


class FixedPositionalEncoding(nn.Module):
    def __init__(self, seq_len=20000, d_model=500, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(seq_len, d_model)  # positional encoding
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):

        x = x + self.pe[:, 0:x.shape[1], :]
        return x


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, seq_len=20000, d_model=500):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.empty(1, seq_len, d_model))  # requires_grad automatically set to True
        # 初始化方式为均匀分布，也可以用trunc_normal_，目前这种初始化方式可能导致位置编码起不到太大作用（因为值域太小，不同位置的编码区别不大）
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):  #（batch-size，ts-len, channle）
        x = x + self.pe[:, 0:x.shape[1], :]
        return x