# TemporalClock/src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from models import TemporalClockModel
from features import process_features
from utils import nll_loss, get_logger

# Configuration
CONFIG = {
    "data_dir": "../data/processed/",
    "experiments_dir": "../experiments/",
    "model_name": "best_model.pth",
    "batch_size": 32,
    "epochs": 50,
    "lr": 0.001,
    "patience": 5,
    "d_model": 128,
    "nhead": 4,
    "num_encoder_layers": 2,
    "lstm_hidden_dim": 64,
}

def run_epoch(model, dataloader, optimizer, criterion, is_training, device):
    """Runs a single epoch of training or validation."""
    epoch_loss = 0
    model.train() if is_training else model.eval()
    
    progress_bar = tqdm(dataloader, desc="Training" if is_training else "Validation")
    
    for X_ts_batch, X_spec_batch, y_batch in progress_bar:
        X_ts_batch = X_ts_batch.to(device).float()
        X_spec_batch = X_spec_batch.to(device).float()
        y_batch = y_batch.to(device).float()

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            mean, log_var = model(X_ts_batch, X_spec_batch)
            loss = criterion(mean, log_var, y_batch)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)


def main():
    """Main training loop."""
    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    feature_data = process_features(CONFIG["data_dir"])
    
    train_dataset = TensorDataset(
        torch.from_numpy(feature_data['X_train_ts']),
        torch.from_numpy(feature_data['X_train_spec']),
        torch.from_numpy(feature_data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.from_numpy(feature_data['X_val_ts']),
        torch.from_numpy(feature_data['X_val_spec']),
        torch.from_numpy(feature_data['y_val'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Initialize model
    model = TemporalClockModel(
        ts_input_dim=1,
        spec_input_dim=feature_data['X_train_spec'].shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        lstm_hidden_dim=CONFIG["lstm_hidden_dim"]
    ).to(device)

    logger.info(f"Model initialized with {model.count_parameters():,} trainable parameters.")

    # Optimizer, Scheduler, and Loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=CONFIG["patience"] // 2, factor=0.5)
    criterion = nll_loss

    # MLflow setup
    mlflow.set_experiment("TemporalClock")
    with mlflow.start_run() as run:
        mlflow.log_params(CONFIG)
        logger.info("Starting training...")

        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(CONFIG["epochs"]):
            train_loss = run_epoch(model, train_loader, optimizer, criterion, is_training=True, device=device)
            val_loss = run_epoch(model, val_loader, None, criterion, is_training=False, device=device)
            
            scheduler.step(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                # Save best model
                save_path = os.path.join(CONFIG["experiments_dir"], CONFIG["model_name"])
                os.makedirs(CONFIG["experiments_dir"], exist_ok=True)
                torch.save(model.state_dict(), save_path)
                mlflow.pytorch.log_model(model, "model")
                logger.info(f"Best model saved to {save_path}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= CONFIG["patience"]:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == '__main__':
    main()