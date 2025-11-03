import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def evaluate_model_on_loader(model, data_loader, criterion, device):
    """Evaluate the model on a data loader"""
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


def predict_sequence_autoregressive(model, src_seq, pred_len, device):
    """Generate predictions autoregressively"""
    model.eval()
    predictions = []

    # Convert to tensor and prepare for model input
    src = torch.FloatTensor(src_seq).unsqueeze(-1).unsqueeze(1)  # (seq_len, 1, 1)
    src = src.to(device)

    # Start with the last value of source sequence as the first decoder input
    current_input = src[-1:, :, :]  # (1, 1, 1) - last value from source

    with torch.no_grad():
        for _ in range(pred_len):
            # Predict next value
            output = model(src, current_input)
            next_val = output[-1:, :, :]  # Get the last (most recent) prediction
            predictions.append(next_val.cpu().numpy()[0, 0, 0])

            # Update decoder input: append the new prediction
            current_input = torch.cat([current_input, next_val], dim=0)

            # Keep only the most recent pred_len values to prevent memory issues
            if current_input.size(0) > pred_len:
                current_input = current_input[-pred_len:, :, :]

    return np.array(predictions)


def walk_forward_validation(data, seq_len, pred_len, initial_train_size,
                            model_params, train_params, device):
    """Perform walk-forward validation"""
    from dlf.models.transformer.ar_transformer import AutoregressiveTransformer
    from dlf.models.transformer.ar_trainer import train_autoregressive_model
    from dlf.models.transformer.univariate import TimeSeriesDataset

    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    n_splits = min(10, len(data) - initial_train_size - pred_len + 1)  # Limit splits for efficiency
    predictions_all = []
    actuals_all = []
    train_sizes = []

    print(f"Starting walk-forward validation with {n_splits} splits...")

    for i in range(n_splits):
        train_end = initial_train_size + i
        test_start = train_end
        test_end = test_start + pred_len

        if test_end > len(data_scaled):
            break

        # Split data
        train_data = data_scaled[:train_end]
        test_data = data_scaled[test_start:test_end]

        print(f"Split {i + 1}/{n_splits}: Train size = {len(train_data)}, Test size = {len(test_data)}")

        # Create datasets for autoregressive mode
        train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len, mode='autoregressive')
        train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'],
                                  shuffle=True, drop_last=True)

        # Initialize model
        model = AutoregressiveTransformer(**model_params).to(device)

        # Train model (reduced epochs for efficiency in walk-forward)
        train_epochs = min(train_params['epochs'], 50)
        train_losses, _ = train_autoregressive_model(
            model, train_loader, train_loader,  # Use train_loader for both train and val
            epochs=train_epochs,
            learning_rate=train_params['lr']
        )

        # Make predictions
        src_seq = train_data[-seq_len:]
        predictions = predict_sequence_autoregressive(model, src_seq, pred_len, device)

        # Store results
        predictions_all.extend(predictions)
        actuals_all.extend(test_data)
        train_sizes.append(len(train_data))

        print(f"  Final train loss: {train_losses[-1]:.6f}")
        print(f"  Predictions: {predictions[:3]}... (showing first 3)")
        print(f"  Actuals:     {test_data[:3]}... (showing first 3)")
        print()

    # Convert back to original scale
    predictions_all = scaler.inverse_transform(np.array(predictions_all).reshape(-1, 1)).flatten()
    actuals_all = scaler.inverse_transform(np.array(actuals_all).reshape(-1, 1)).flatten()

    return np.array(predictions_all), np.array(actuals_all), train_sizes


def evaluate_single_prediction(model, data, seq_len, pred_len, device, scaler=None):
    """
    Make a single prediction using the most recent data

    Args:
        model: Trained autoregressive transformer
        data: Time series data
        seq_len: Length of input sequence
        pred_len: Number of steps to predict
        device: PyTorch device
        scaler: Optional StandardScaler for normalization

    Returns:
        predictions: Array of predicted values
        input_sequence: The input sequence used for prediction
    """
    model.eval()

    # Prepare data
    if scaler is not None:
        data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    else:
        data_scaled = data

    # Get the most recent sequence
    src_seq = data_scaled[-seq_len:]

    # Make prediction
    predictions = predict_sequence_autoregressive(model, src_seq, pred_len, device)

    # Convert back to original scale if scaler provided
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        src_seq = scaler.inverse_transform(src_seq.reshape(-1, 1)).flatten()

    return predictions, src_seq


def plot_results(actuals, predictions, train_sizes, original_data, initial_train_size):
    """Plot the results of walk-forward validation"""

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f"\nOut-of-Sample Performance Metrics:")
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Full time series with predictions
    axes[0, 0].plot(original_data, label='Original Data', alpha=0.7)
    axes[0, 0].axvline(x=initial_train_size, color='red', linestyle='--',
                       label='Start of Predictions', alpha=0.7)

    # Plot predictions at their correct positions
    pred_positions = np.arange(initial_train_size, initial_train_size + len(predictions))
    axes[0, 0].plot(pred_positions, predictions, 'r-', label='Predictions', alpha=0.8)
    axes[0, 0].plot(pred_positions, actuals, 'g-', label='Actuals', alpha=0.8)

    axes[0, 0].set_title('Time Series with Walk-Forward Predictions')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Predictions vs Actuals scatter
    axes[0, 1].scatter(actuals, predictions, alpha=0.6)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Predictions vs Actuals')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Residuals
    residuals = actuals - predictions
    axes[1, 0].plot(residuals, alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Prediction Residuals')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Training set sizes
    axes[1, 1].plot(train_sizes)
    axes[1, 1].set_xlabel('Validation Step')
    axes[1, 1].set_ylabel('Training Set Size')
    axes[1, 1].set_title('Training Set Size Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()