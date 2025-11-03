import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math


def train_model(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):
        # Move to device and transpose for transformer format (seq_len, batch_size, features)
        src = src.transpose(0, 1).to(device)  # (seq_len, batch_size, input_dim)
        tgt_input = tgt_input.transpose(0, 1).to(device)
        tgt_output = tgt_output.transpose(0, 1).to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt_input)

        # Calculate loss
        loss = criterion(output, tgt_output)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def train_autoregressive_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    """
    Complete training function for autoregressive transformer

    Args:
        model: AutoregressiveTransformer model
        train_loader: Training data loader (should return src, tgt_input, tgt_output)
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses = []
    val_losses = []

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(epochs):
        # Training
        train_loss = train_model(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

    return train_losses, val_losses


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate the model on validation data"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt_input, tgt_output in data_loader:
            src = src.transpose(0, 1).to(device)
            tgt_input = tgt_input.transpose(0, 1).to(device)
            tgt_output = tgt_output.transpose(0, 1).to(device)

            output = model(src, tgt_input)
            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(data_loader)