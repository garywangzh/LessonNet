import math
import json
import torch
import torch.nn as nn
import numpy as np
import os

class PositionalEncoding(nn.Module):
    """固定位置编码（Transformer 标准实现）"""
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttentionLayer(nn.Module):
    """单层交叉注意力：Q 来自中心，K,V 来自完整序列"""
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, q, kv):
        # q: [B, 1, d_model], kv: [B, T, d_model]
        attn_out, _ = self.attn(q, kv, kv)        # 交叉注意力
        q = self.norm1(q + self.dropout1(attn_out))

        ff_out = self.linear2(self.dropout3(torch.relu(self.linear1(q))))
        q = self.norm2(q + self.dropout2(ff_out))
        return q


class TemporalTransformerScorer(nn.Module):
    """
    基于 Transformer 的时序重要性评分模型。
    以序列中心片段为 Query，两侧上下文为 Key/Value，
    通过多层交叉注意力聚合上下文信息，最终输出中心片段的分数。

    参数：
        input_dim    : 输入特征维度（4 或 5）
        d_model      : Transformer 内部维度
        nhead        : 多头注意力头数
        num_layers   : 交叉注意力层数（可通过此参数便捷修改模型深度）
        dim_feedforward : 前馈网络维度
        dropout      : dropout 比率
        use_pos_enc  : 是否使用位置编码（建议开启）
    """
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, use_pos_enc=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout) if use_pos_enc else nn.Identity()

        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: [B, T, C]  其中 T 为序列长度（建议为奇数，中心位置为 T//2）
        返回: [B, 1]   中心片段的重要性分数（0~1）
        """
        B, T, _ = x.shape
        center_idx = T // 2

        # 投影到 d_model
        x = self.input_proj(x)                 # [B, T, d_model]
        x = self.pos_enc(x)                    # [B, T, d_model]

        # 提取中心 Query
        q = x[:, center_idx:center_idx+1, :]   # [B, 1, d_model]
        kv = x                                 # [B, T, d_model]

        # 逐层交叉注意力
        for layer in self.layers:
            q = layer(q, kv)                   # [B, 1, d_model]

        # 输出分数
        score = self.output_proj(q)            # [B, 1, 1]
        score = score.squeeze(-1)              # [B, 1]
        return torch.sigmoid(score)            # [B, 1]