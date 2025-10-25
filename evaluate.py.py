# TemporalClock/src/evaluate.py

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from models import TemporalClockModel
from features import process_features
from utils import mae, rmse, get_logger
from train import CONFIG  # Reuse config from train script

def evaluate_model(model, test_loader, device, scaler):
    """
    Evaluates the model on the test dataset.

    Args:
        model (TemporalClockModel): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to run evaluation on.
        scaler (StandardScaler): The scaler used for inverse transforming predictions.

    Returns:
        tuple: A tuple containing true values, mean predictions, and std deviations.
    """
    model.eval()
    all_preds_mean = []
    all_preds_std = []
    all_true = []

    with torch.no_grad():
        for X_ts_batch, X_spec_batch, y_batch in test_loader:
            X_ts_batch = X_ts_batch.to(device).float()
            X_spec_batch = X_spec_batch.to(device).float()
            
            mean, log_var = model(X_ts_batch, X_spec_batch)
            std = torch.exp(0.5 * log_var)

            # Inverse transform to original scale
            mean_inv = scaler.inverse_transform(mean.cpu().numpy().reshape(-1, 1)).reshape(mean.shape)
            std_inv = scaler.scale_ * std.cpu().numpy() # Scale standard deviation
            
            all_preds_mean.append(mean_inv)
            all_preds_std.append(std_inv)
            all_true.append(y_batch.numpy())

    return np.concatenate(all_true), np.concatenate(all_preds_mean), np.concatenate(all_preds_std)

def main():
    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data and scaler
    feature_data = process_features(CONFIG["data_dir"])
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(feature_data['X_test_ts']),
        torch.from_numpy(feature_data['X_test_spec']),
        torch.from_numpy(feature_data['y_test'])
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # Initialize and load model
    model = TemporalClockModel(
        ts_input_dim=1,
        spec_input_dim=feature_data['X_test_spec'].shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        lstm_hidden_dim=CONFIG["lstm_hidden_dim"]
    ).to(device)
    
    model_path = os.path.join(CONFIG["experiments_dir"], CONFIG["model_name"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Model loaded from {model_path}")
    
    # Evaluate
    y_true, y_pred_mean, y_pred_std = evaluate_model(model, test_loader, device, feature_data['scaler'])
    
    # Calculate metrics
    mae_score = mae(y_true, y_pred_mean)
    rmse_score = rmse(y_true, y_pred_mean)
    logger.info(f"Test MAE: {mae_score:.6f}")
    logger.info(f"Test RMSE: {rmse_score:.6f}")
    
    # Calibration check
    within_one_std = np.sum((y_true >= y_pred_mean - y_pred_std) & (y_true <= y_pred_mean + y_pred_std))
    coverage_one_std = (within_one_std / y_true.size) * 100
    logger.info(f"Uncertainty Coverage (1-sigma): {coverage_one_std:.2f}% (Expected: ~68%)")

    # Plotting
    results_dir = os.path.join(CONFIG["experiments_dir"], "results")
    os.makedirs(results_dir, exist_ok=True)

    # Plot prediction vs true for a sample
    plt.figure(figsize=(15, 7))
    sample_idx = 0
    time_steps = np.arange(y_true.shape[1])
    plt.plot(time_steps, y_true[sample_idx, :, 0], label="True Drift", color='blue')
    plt.plot(time_steps, y_pred_mean[sample_idx, :, 0], label="Predicted Drift", color='red')
    plt.fill_between(time_steps,
                     y_pred_mean[sample_idx, :, 0] - y_pred_std[sample_idx, :, 0],
                     y_pred_mean[sample_idx, :, 0] + y_pred_std[sample_idx, :, 0],
                     color='red', alpha=0.2, label="1-sigma Uncertainty")
    plt.title("Predicted vs True Drift with Uncertainty")
    plt.xlabel("Time Step")
    plt.ylabel("Drift (microseconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "prediction_vs_true.png"))
    plt.show()

if __name__ == '__main__':
    main()