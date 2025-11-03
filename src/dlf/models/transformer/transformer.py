import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=8, num_layers=3,
                 output_dim=1, seq_length=100, pred_length=20, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.d_model = d_model
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length=seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim * pred_length)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (batch_size, seq_length, input_dim)
        batch_size = src.size(0)

        # Input projection
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # Transpose for positional encoding (seq_length, batch_size, d_model)
        src = src.transpose(0, 1)
        src = self.pos_encoding(src)

        # Transpose back for transformer (batch_size, seq_length, d_model)
        src = src.transpose(0, 1)
        src = self.dropout(src)

        # Transformer encoding
        output = self.transformer_encoder(src)

        # Use the last token's representation for prediction
        last_token = output[:, -1, :]  # (batch_size, d_model)

        # Output projection
        prediction = self.output_projection(last_token)  # (batch_size, output_dim * pred_length)
        prediction = prediction.view(batch_size, self.pred_length, -1)

        return prediction
