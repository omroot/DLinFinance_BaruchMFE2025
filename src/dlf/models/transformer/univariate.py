import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length, mode='vanilla'):
        """
        Dataset for time series forecasting

        Args:
            data: 1D array of time series data
            seq_length: Length of input sequence
            pred_length: Length of prediction sequence
            mode: 'vanilla' for non-autoregressive, 'autoregressive' for AR transformer
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.mode = mode

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        if self.mode == 'vanilla':
            # For vanilla transformer: input sequence -> output sequence
            x = self.data[idx:idx + self.seq_length]
            y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]
            return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y).unsqueeze(-1)

        elif self.mode == 'autoregressive':
            # For autoregressive transformer: src, tgt_input, tgt_output
            src = self.data[idx:idx + self.seq_length]
            tgt_full = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]

            # For teacher forcing: input is shifted right with start token (use last src value)
            tgt_input = np.concatenate([[src[-1]], tgt_full[:-1]])
            tgt_output = tgt_full

            return (torch.FloatTensor(src).unsqueeze(-1),
                    torch.FloatTensor(tgt_input).unsqueeze(-1),
                    torch.FloatTensor(tgt_output).unsqueeze(-1))

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def generate_synthetic_data(length=2000, noise_level=0.1):
    """Generate synthetic time series data with trends and seasonality"""
    t = np.linspace(0, 4 * np.pi, length)

    # Multiple sine waves with different frequencies and phases
    trend = 0.01 * t
    seasonal1 = np.sin(t) * 2
    seasonal2 = np.sin(2 * t + np.pi / 4) * 0.5
    seasonal3 = np.sin(0.5 * t + np.pi / 2) * 1.5

    # Add noise
    noise = np.random.normal(0, noise_level, length)

    data = trend + seasonal1 + seasonal2 + seasonal3 + noise
    return data


def create_data_loaders(data, seq_length, pred_length, train_ratio=0.8,
                        batch_size=32, mode='vanilla'):
    """
    Create train and validation data loaders

    Args:
        data: Time series data array
        seq_length: Input sequence length
        pred_length: Prediction sequence length
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for data loaders
        mode: 'vanilla' or 'autoregressive'
    """
    # Split data
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, seq_length, pred_length, mode=mode)
    val_dataset = TimeSeriesDataset(val_data, seq_length, pred_length, mode=mode)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, train_data, val_data