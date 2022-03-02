import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# def spatial_dynamic_block(input, pool_size, d_prime):
#     ''' Create a spatial_dynamic_block
#     Args:
#         input: input tensor (BATCH, input_dims, length)
#         pool_size: pooling window for AvgPooling1D layer
#         comr_factor: the compression faction for hidden dimensions
#     Returns: a keras tensor
#     '''
#     m = F.avg_pool1d(input, kernel_size=5, stride=1, padding=2)
#     m = nn.Linear(input_dims, hidden_dims)
#     filters = input.shape[-1] # channel_axis = -1 for TF
#
#     se = Dense(d_prime,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
#     se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
#     se = multiply([input, se])
#     return se
class SamePadConv(nn.Module):#
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        # 感受野是固定值
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.linear0 = nn.Linear(in_channels, in_channels)
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        #i=10的时候final=True
        self.BN1_64_1 = nn.BatchNorm1d(64)
        self.BN1_64_2 = nn.BatchNorm1d(64)
        #self.BN2_64 = nn.BatchNorm1d(64)
        self.BN2_320 = nn.BatchNorm1d(320)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        #前面的块projector都是none,下面的一行代码执行前面一部分
        residual = x if self.projector is None else self.projector(x)
        #实现网络的非线性功能
        # x = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)#B*co*T
        # x = self.linear0(x.transpose(1,2)).transpose(1,2)
        # channels_BN = residual.size(1)
        #j = nn.BatchNorm1d(channels_BN).to(device=x.device)
        #x = x.to('cpu')
        # x = j(x)
        # print('----')
        #
        # print(residual.size(1))
        # print(x.size(1))

        # if channels_BN==64:
        #     x = self.BN1_64_1(x)
        #     x = F.gelu(x)
        #     x = self.conv1(x)
        #     x = self.BN1_64_1(x)
        #     x = F.gelu(x)
        #     x = self.conv2(x)
        # else:
        #     x = self.BN1_64_1(x)
        #     # print('BN2_64')
        #     # print(x.size(1))
        #     x = F.gelu(x)
        #     # print('gelu')
        #     # print(x.size(1))
        #     x = self.conv1(x)
        #     # print('conv1(')
        #     # print(x.size(1))
        #     x = self.BN2_320(x)
        #     # print('BN2_320')
        #     # print(x.size(1))
        #     x = F.gelu(x)
        #     # print('gelu(x)')
        #     # print(x.size(1))
        #     x = self.conv2(x)
        #     # print('conv2')
        #     # print(x.size(1))
        x = F.gelu(x)
        # print('gelu')
        # print(x.size(1))
        x = self.conv1(x)
        # print('conv1(x)')
        # print(x.size(1))
        # # x = self.BN2(x)
        x = F.gelu(x)
        # print('gelu(x)')
        # print(x.size(1))
        x = self.conv2(x)
        # print('conv2(x)')
        # print(x.size(1))

        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        #ker = [2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
