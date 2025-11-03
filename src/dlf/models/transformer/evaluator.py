import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def evaluate_model(model, test_data, seq_length, pred_length, device, num_predictions=5, scaler=None):
    """
    Evaluate vanilla transformer model and generate predictions

    Args:
        model: Trained TimeSeriesTransformer
        test_data: Test data array
        seq_length: Input sequence length
        pred_length: Prediction sequence length
        device: PyTorch device
        num_predictions: Number of predictions to make
        scaler: Optional StandardScaler for data normalization

    Returns:
        predictions: List of prediction arrays
        actuals: List of actual value arrays
    """
    model.eval()
    predictions = []
    actuals = []

    # Prepare data
    if scaler is not None:
        test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    else:
        test_data_scaled = test_data

    with torch.no_grad():
        for i in range(num_predictions):
            # Calculate start index for this prediction
            start_idx = len(test_data_scaled) - seq_length - pred_length * (num_predictions - i)

            if start_idx < 0:
                continue

            # Get input and target sequences
            input_seq = test_data_scaled[start_idx:start_idx + seq_length]
            actual_seq = test_data_scaled[start_idx + seq_length:start_idx + seq_length + pred_length]

            # Convert to tensor and make prediction
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(input_tensor)
            pred = pred.squeeze().cpu().numpy()

            # Convert back to original scale if scaler provided
            if scaler is not None:
                if pred.ndim == 0:  # Single value
                    pred = np.array([pred])
                pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                actual_seq = scaler.inverse_transform(actual_seq.reshape(-1, 1)).flatten()

            predictions.append(pred)
            actuals.append(actual_seq)

    return predictions, actuals


def predict_future(model, data, seq_length, pred_length, device, scaler=None):
    """
    Make a prediction for future values using the most recent data

    Args:
        model: Trained TimeSeriesTransformer
        data: Time series data
        seq_length: Input sequence length
        pred_length: Number of future steps to predict
        device: PyTorch device
        scaler: Optional StandardScaler for normalization

    Returns:
        prediction: Array of predicted future values
        input_sequence: The input sequence used for prediction
    """
    model.eval()

    # Prepare data
    if scaler is not None:
        data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    else:
        data_scaled = data

    # Get the most recent sequence
    input_seq = data_scaled[-seq_length:]

    with torch.no_grad():
        # Convert to tensor and make prediction
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)
        prediction = model(input_tensor)
        prediction = prediction.squeeze().cpu().numpy()

        # Convert back to original scale if scaler provided
        if scaler is not None:
            if prediction.ndim == 0:  # Single value
                prediction = np.array([prediction])
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
            input_seq = scaler.inverse_transform(input_seq.reshape(-1, 1)).flatten()

    return prediction, input_seq


def walk_forward_validation_vanilla(data, seq_len, pred_len, initial_train_size,
                                    model_params, train_params, device):
    """
    Perform walk-forward validation for vanilla transformer

    Args:
        data: Time series data
        seq_len: Input sequence length
        pred_len: Prediction length
        initial_train_size: Initial training set size
        model_params: Dictionary of model parameters
        train_params: Dictionary of training parameters
        device: PyTorch device

    Returns:
        predictions_all: Array of all predictions
        actuals_all: Array of all actual values
        train_sizes: List of training set sizes
    """
    from dlf.models.transformer.transformer import TimeSeriesTransformer
    from dlf.models.transformer.trainer import train_model
    from dlf.models.transformer.univariate import TimeSeriesDataset

    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    n_splits = min(10, len(data) - initial_train_size - pred_len + 1)  # Limit splits for efficiency
    predictions_all = []
    actuals_all = []
    train_sizes = []

    print(f"Starting vanilla transformer walk-forward validation with {n_splits} splits...")

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

        # Create datasets for vanilla mode
        train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len, mode='vanilla')
        val_dataset = TimeSeriesDataset(train_data, seq_len, pred_len, mode='vanilla')  # Use train data for val

        train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'],
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'],
                                shuffle=False, drop_last=True)

        # Initialize model
        model = TimeSeriesTransformer(**model_params).to(device)

        # Train model (reduced epochs for efficiency in walk-forward)
        train_epochs = min(train_params['epochs'], 50)
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            epochs=train_epochs,
            learning_rate=train_params['lr']
        )

        # Make prediction using the most recent sequence
        input_seq = train_data[-seq_len:]
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)

        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = prediction.squeeze().cpu().numpy()

        # Store results
        predictions_all.extend(prediction)
        actuals_all.extend(test_data)
        train_sizes.append(len(train_data))

        print(f"  Final train loss: {train_losses[-1]:.6f}")
        print(f"  Predictions: {prediction[:3]}... (showing first 3)")
        print(f"  Actuals:     {test_data[:3]}... (showing first 3)")
        print()

    # Convert back to original scale
    predictions_all = scaler.inverse_transform(np.array(predictions_all).reshape(-1, 1)).flatten()
    actuals_all = scaler.inverse_transform(np.array(actuals_all).reshape(-1, 1)).flatten()

    return np.array(predictions_all), np.array(actuals_all), train_sizes


def plot_results(data, train_losses, val_losses, predictions, actuals, seq_length, pred_length):
    """Plot training results and predictions for vanilla transformer"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Original time series
    axes[0, 0].plot(data)
    axes[0, 0].set_title('Synthetic Time Series Data')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True)

    # Plot 2: Training and validation loss
    axes[0, 1].plot(train_losses, label='Training Loss')
    axes[0, 1].plot(val_losses, label='Validation Loss')
    axes[0, 1].set_title('Training Progress')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Predictions vs Actuals
    axes[1, 0].set_title('Predictions vs Actuals')
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        x_pred = range(i * pred_length, (i + 1) * pred_length)
        x_actual = range(i * pred_length, (i + 1) * pred_length)

        axes[1, 0].plot(x_actual, actual, 'o-', color=colors[i % len(colors)],
                        alpha=0.7, label=f'Actual {i + 1}', markersize=4)
        axes[1, 0].plot(x_pred, pred, 's--', color=colors[i % len(colors)],
                        alpha=0.9, label=f'Predicted {i + 1}', markersize=4)

    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Value')

    # Plot 4: Recent data with predictions
    recent_data = data[-200:]
    axes[1, 1].plot(recent_data, label='Recent Data', alpha=0.7)

    if predictions:
        start_pred = len(recent_data) - pred_length
        axes[1, 1].plot(range(start_pred, len(recent_data)),
                        predictions[-1], 'r--', linewidth=2,
                        label='Latest Prediction', marker='o', markersize=4)

    axes[1, 1].set_title('Recent Data with Latest Prediction')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Value')

    plt.tight_layout()
    plt.show()


def calculate_metrics(predictions, actuals):
    """Calculate performance metrics"""
    predictions_flat = np.concatenate(predictions) if isinstance(predictions[0], np.ndarray) else np.array(predictions)
    actuals_flat = np.concatenate(actuals) if isinstance(actuals[0], np.ndarray) else np.array(actuals)

    mse = mean_squared_error(actuals_flat, predictions_flat)
    mae = mean_absolute_error(actuals_flat, predictions_flat)
    rmse = np.sqrt(mse)

    print(f"\nPerformance Metrics:")
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {'mse': mse, 'mae': mae, 'rmse': rmse}