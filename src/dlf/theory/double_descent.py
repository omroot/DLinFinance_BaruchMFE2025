"""
Utility functions for HW01 Part 2: Double Descent Phenomenon
=============================================================
This module contains all helper functions for demonstrating the double descent
phenomenon in both linear regression and neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from typing import Tuple, List, Dict


# ============================================================================
# Linear Regression Functions
# ============================================================================

def generate_linear_regression_data(n: int = 100,
                                   d: int = 200,
                                   sigma: float = 0.2,
                                   beta_decay: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate linear regression data: y = X @ beta + epsilon

    Args:
        n: Number of samples
        d: Number of features
        sigma: Noise standard deviation
        beta_decay: If True, use decaying coefficients beta_j ~ 1/j^2

    Returns:
        X: Design matrix (n x d)
        y: Response vector (n,)
        beta_true: True coefficients (d,)
    """
    # Generate design matrix from standard normal
    X = np.random.randn(n, d)

    # Generate true coefficients
    if beta_decay:
        # Decaying coefficients: beta_j ~ 1/j^2
        beta_true = np.random.randn(d) / (np.arange(1, d + 1) ** 2)
    else:
        # Standard normal coefficients
        beta_true = np.random.randn(d)

    # Generate response with Gaussian noise
    y = X @ beta_true + np.random.randn(n) * sigma

    return X, y, beta_true


def fit_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit least squares: beta_hat = (X^T X)^{-1} X^T y
    Only works when X has full column rank (p < n)

    Args:
        X: Design matrix (n x p)
        y: Response vector (n,)

    Returns:
        beta_hat: Estimated coefficients (p,)
    """
    try:
        beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
        return beta_hat
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if matrix is singular
        return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_minimum_norm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit minimum-norm interpolator: beta_hat = X^T (X X^T)^{-1} y
    This is the minimum L2 norm solution among all interpolating solutions.
    Works when p >= n (overparameterized regime)

    Args:
        X: Design matrix (n x p)
        y: Response vector (n,)

    Returns:
        beta_hat: Estimated coefficients (p,)
    """
    # Using Moore-Penrose pseudoinverse: X^+ = X^T (X X^T)^{-1}
    beta_hat = X.T @ np.linalg.solve(X @ X.T, y)
    return beta_hat


def compute_risk(X: np.ndarray, y: np.ndarray, beta_hat: np.ndarray) -> float:
    """
    Compute mean squared error: (1/n) ||y - X beta_hat||^2

    Args:
        X: Design matrix (n x p)
        y: Response vector (n,)
        beta_hat: Estimated coefficients (p,)

    Returns:
        MSE risk
    """
    predictions = X @ beta_hat
    risk = np.mean((y - predictions) ** 2)
    return risk


def double_descent_experiment(n: int = 100,
                             d: int = 200,
                             p_values: List[int] = None,
                             sigma: float = 0.2,
                             n_trials: int = 50,
                             seed: int = 42) -> Dict:
    """
    Run double descent experiment by varying number of selected features.

    Args:
        n: Number of training samples
        d: Total number of features
        p_values: List of feature counts to try
        sigma: Noise standard deviation
        n_trials: Number of random trials to average over
        seed: Random seed for reproducibility

    Returns:
        Dictionary with results
    """
    if p_values is None:
        p_values = list(range(10, 151, 5))

    results = {
        'p_values': p_values,
        'train_risks': [],
        'test_risks': [],
        'train_risks_std': [],
        'test_risks_std': []
    }

    print(f"\nRunning experiment with n={n}, d={d}, sigma={sigma}")
    print(f"Number of trials: {n_trials}")
    print(f"Testing {len(p_values)} different feature counts from {min(p_values)} to {max(p_values)}")
    print("-" * 80)

    for p in p_values:
        train_risks_trials = []
        test_risks_trials = []

        for trial in range(n_trials):
            # Generate train and test data
            X_full_train, y_train, beta_true = generate_linear_regression_data(n, d, sigma)
            X_full_test, y_test, _ = generate_linear_regression_data(n, d, sigma, beta_decay=False)
            # Use same beta for test
            y_test = X_full_test @ beta_true + np.random.randn(n) * sigma

            # Randomly select p features
            np.random.seed(seed + trial)
            selected_features = np.random.choice(d, size=p, replace=False)
            X_train = X_full_train[:, selected_features]
            X_test = X_full_test[:, selected_features]

            # Fit model
            if p < n:
                # Underdetermined: use standard least squares
                beta_hat = fit_least_squares(X_train, y_train)
            else:
                # Overdetermined: use minimum-norm interpolator
                beta_hat = fit_minimum_norm(X_train, y_train)

            # Compute risks
            train_risk = compute_risk(X_train, y_train, beta_hat)
            test_risk = compute_risk(X_test, y_test, beta_hat)

            train_risks_trials.append(train_risk)
            test_risks_trials.append(test_risk)

        # Store mean and std
        results['train_risks'].append(np.mean(train_risks_trials))
        results['test_risks'].append(np.mean(test_risks_trials))
        results['train_risks_std'].append(np.std(train_risks_trials))
        results['test_risks_std'].append(np.std(test_risks_trials))

        if p % 20 == 0 or p == n or p == n + 1:
            print(f"p={p:3d}: Train Risk={np.mean(train_risks_trials):.6f}, "
                  f"Test Risk={np.mean(test_risks_trials):.6f}")

    return results


# ============================================================================
# Neural Network Functions
# ============================================================================

def load_fashion_mnist(subset_size: int = 1000,
                      test_size: int = 500,
                      data_dir: str = './data'):
    """
    Load Fashion-MNIST dataset with optional subset.

    Args:
        subset_size: Number of training samples to use
        test_size: Number of test samples to use
        data_dir: Directory to store data

    Returns:
        Train and test datasets
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load training data
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download and load test data
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create subsets
    if subset_size < len(train_dataset):
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    if test_size < len(test_dataset):
        indices = np.random.choice(len(test_dataset), test_size, replace=False)
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    return train_dataset, test_dataset


class FullyConnectedNet(nn.Module):
    """
    Fully connected neural network with variable width.
    Architecture: Input (784) -> Hidden1 -> Hidden2 -> ... -> Output (10)
    """
    def __init__(self, input_dim: int = 784,
                 hidden_dims: List[int] = [100, 100],
                 output_dim: int = 10,
                 activation: str = 'relu'):
        super(FullyConnectedNet, self).__init__()

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

        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_neural_network(model: nn.Module,
                        train_dataset,
                        test_dataset,
                        epochs: int = 100,
                        batch_size: int = 32,
                        lr: float = 0.001,
                        patience: int = 20,
                        verbose: bool = True,
                        device: str = None) -> Dict:
    """
    Train a neural network and track performance.

    Args:
        model: PyTorch model
        train_dataset, test_dataset: PyTorch datasets
        epochs: Maximum number of epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        verbose: Whether to print progress
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.

    Returns:
        Dictionary with training history and metrics
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Move model to device
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
    }

    best_test_acc = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_loss /= len(train_dataset)
        train_acc = 100. * train_correct / train_total

        # Test phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()

        test_loss /= len(test_dataset)
        test_acc = 100. * test_correct / test_total

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, "
                  f"Test Acc: {test_acc:.2f}%")

    training_time = time.time() - start_time

    results = {
        'model': model,
        'history': history,
        'best_test_acc': best_test_acc,
        'final_train_acc': train_acc,
        'final_test_acc': test_acc,
        'training_time': training_time,
        'num_parameters': model.count_parameters(),
        'epochs_trained': len(history['train_loss'])
    }

    return results
