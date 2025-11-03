import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

from dlf.models.transformer.transformer import PositionalEncoding


class AutoregressiveTransformer(nn.Module):
    """Autoregressive Transformer for time series forecasting"""

    def __init__(self, input_dim=1, d_model=64, nhead=8, num_layers=3,
                 dim_feedforward=256, dropout=0.1, max_len=5000):
        super(AutoregressiveTransformer, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, tgt):
        """
        Args:
            src: Source sequence (seq_len, batch_size, input_dim)
            tgt: Target sequence (seq_len, batch_size, input_dim)
        """
        seq_len = tgt.size(0)

        # Create causal mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)

        # Project inputs to d_model dimensions
        src = self.input_projection(src) * math.sqrt(self.d_model)
        tgt = self.input_projection(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Apply dropout
        src = self.dropout(src)
        tgt = self.dropout(tgt)

        # Pass through transformer decoder
        output = self.transformer_decoder(tgt, src, tgt_mask=tgt_mask)

        # Project back to input dimension
        output = self.output_projection(output)

        return output