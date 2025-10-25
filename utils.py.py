# TemporalClock/src/utils.py

import numpy as np
import torch
import logging
import sys

def mae(y_true, y_pred):
    """Calculates Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def nll_loss(mean, log_var, target):
    """
    Calculates the Negative Log-Likelihood loss for a Gaussian distribution.
    This loss function encourages the model to produce accurate means and
    calibrated uncertainty estimates.
    
    Loss = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
    """
    # Clamp log_var for stability
    log_var = torch.clamp(log_var, -10, 10)
    
    # Ensure target has the same shape as mean and log_var
    if target.shape != mean.shape:
        target = target.view_as(mean)
        
    # The precision term (inverse of variance)
    precision = torch.exp(-log_var)
    
    # Squared error term
    error = (target - mean) ** 2
    
    # The loss
    loss = 0.5 * (log_var + error * precision)
    
    return loss.mean()


def get_logger(name):
    """
    Configures and returns a logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Visualization functions can be added here as needed
# For example, plot_spectral_residuals, etc.