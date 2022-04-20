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
class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.gru = nn.GRU(64, 128, 3)
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
        self.input_x_a = nn.Linear(output_dims+hidden_dims, output_dims+hidden_dims)
        self.repr_dropout = nn.Dropout(p=0.1)
    def forward(self, x, mask=None):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)
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
        x1 = x.transpose(1, 2).to(x.device)  # B x Ch x T
        x1 = self.repr_dropout(self.feature_extractor(x1))  # B x Co x T
        x1 = F.avg_pool1d(x1, kernel_size=7, stride=1, padding=3)  # B*co*T
        x1 = x1.transpose(1, 2)# B  x T*co
        x2 = x.to(x.device)  # B x T*ch
        x2 = x2.transpose(0, 1)  # Tx B*ch
        x2, hn = self.gru(x2)  # Tx B*ch
        x2 = x2.transpose(0, 1)  # B*T*ch
        x2 = x2.transpose(1, 2) # B*ch*T
        x2 = F.avg_pool1d(x2, kernel_size=7, stride=1, padding=3)  # B*ch*T
        x2 = x2.transpose(1, 2)  # B*T*ch
        x_a = torch.cat([x1, x2], dim=-1).to(x.device)
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
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):  #（batch-size，ts-len, channle）
        x = x + self.pe[:, 0:x.shape[1], :]
        return x