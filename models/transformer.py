import torch
from torch_geometric import nn
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 特征聚合层
        self.feature_pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size=1026, seq_len=8, input_dim=5)
        Returns:
            features: (batch_size, d_model)
        """
        # 维度投影和位置编码
        x = self.input_proj(x)  # (1026,8,5) -> (1026,8,d_model)
        x = self.pos_encoder(x)  # 添加位置信息

        # Transformer编码
        encoded = self.transformer_encoder(x)  # (1026,8,d_model)

        # 特征聚合：沿时间维度平均池化
        features = self.feature_pool(encoded.permute(0, 2, 1))  # (1026,d_model)
        return features