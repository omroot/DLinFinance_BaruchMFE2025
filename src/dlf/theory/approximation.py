"""
Utility functions for HW01 Part 1: Deep vs Shallow Networks
============================================================
This module contains all the helper functions for demonstrating
the approximation power of deep vs shallow networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from typing import Tuple, List, Dict


# ============================================================================
# Data Generation Functions
# ============================================================================

def h_function(a: float, b: float) -> float:
    """
    Basic non-linear function for composition.
    h(a,b) = tanh(a + b^2)

    Args:
        a, b: Input values

    Returns:
        Non-linear combination
    """
    return np.tanh(a + b**2)


def compositional_function(x: np.ndarray) -> np.ndarray:
    """
    Hierarchical compositional function following a binary tree structure:

    f(x1,...,x8) = h3(h21(h11(x1,x2), h12(x3,x4)), h22(h13(x5,x6), h14(x7,x8)))

    This represents a 3-level binary tree where each node computes h(left, right).

    Args:
        x: Input array of shape (n_samples, 8)

    Returns:
        Output array of shape (n_samples,)
    """
    assert x.shape[1] == 8, "Input must have 8 features"

    # Level 1: Leaf computations (4 nodes)
    h11 = h_function(x[:, 0], x[:, 1])
    h12 = h_function(x[:, 2], x[:, 3])
    h13 = h_function(x[:, 4], x[:, 5])
    h14 = h_function(x[:, 6], x[:, 7])

    # Level 2: Internal nodes (2 nodes)
    h21 = h_function(h11, h12)
    h22 = h_function(h13, h14)

    # Level 3: Root (1 node)
    h3 = h_function(h21, h22)

    return h3


def generate_compositional_data(n_samples: int = 1000,
                               noise_std: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset based on compositional function.

    Args:
        n_samples: Number of samples to generate
        noise_std: Standard deviation of Gaussian noise

    Returns:
        X: Input features of shape (n_samples, 8)
        y: Target values of shape (n_samples,)
    """
    # Generate input features from standard normal
    X = np.random.randn(n_samples, 8)

    # Compute target using compositional function
    y = compositional_function(X)

    # Add Gaussian noise
    y += np.random.randn(n_samples) * noise_std

    return X, y


def non_compositional_function(x: np.ndarray) -> np.ndarray:
    """
    Non-compositional function using random Fourier features.
    f(x) = sum_i w_i * sin(omega_i * x_i)

    Args:
        x: Input array of shape (n_samples, 8)

    Returns:
        Output array of shape (n_samples,)
    """
    # Fixed random weights and frequencies for reproducibility
    np.random.seed(123)
    w = np.random.randn(8)
    omega = np.random.randn(8) * 2

    result = np.sum(w * np.sin(omega * x), axis=1)
    return result


# ============================================================================
# Neural Network Architectures
# ============================================================================

class ShallowNetwork(nn.Module):
    """
    Shallow neural network: Input -> Hidden -> Output
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 100,
                 activation: str = 'relu'):
        super(ShallowNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepNetwork(nn.Module):
    """
    Deep neural network matching compositional structure: Input -> Layer1 -> Layer2 -> Layer3 -> Output
    Architecture: 8 -> hidden1 -> hidden2 -> hidden3 -> 1
    """
    def __init__(self, input_dim: int = 8, hidden_dims: List[int] = [16, 8, 4],
                 activation: str = 'relu'):
        super(DeepNetwork, self).__init__()

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation)
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training Function
# ============================================================================

def train_network(model: nn.Module,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 lr: float = 0.001,
                 verbose: bool = False,
                 device: str = None) -> Dict:
    """
    Train a neural network and track performance.

    Args:
        model: PyTorch model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        verbose: Whether to print training progress
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.

    Returns:
        Dictionary containing training history and final metrics
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Move model to device
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
    }

    # Training loop
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(X_train)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    training_time = time.time() - start_time

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train_loss = criterion(model(X_train_t), y_train_t).item()
        final_test_loss = criterion(model(X_test_t), y_test_t).item()

    results = {
        'model': model,
        'history': history,
        'train_loss': final_train_loss,
        'test_loss': final_test_loss,
        'training_time': training_time,
        'num_parameters': model.count_parameters()
    }

    return results
