# TemporalClock/src/preprocess.py

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_samples=1000, seq_length=512, seed=42):
    """
    Generates synthetic high-frequency clock drift data.

    This function simulates clock drift with several components:
    1.  A baseline linear drift.
    2.  A slow-moving thermal drift component (sinusoidal).
    3.  Multi-frequency periodic interference (e.g., from power supplies).
    4.  Random Gaussian noise.
    5.  Optional environmental effects (temperature, voltage).

    Args:
        num_samples (int): The number of time-series sequences to generate.
        seq_length (int): The length of each individual time-series sequence.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The generated clock drift data (num_samples, seq_length, 1).
            - np.ndarray: The associated metadata/features (num_samples, seq_length, 2).
    """
    np.random.seed(seed)
    
    # Time vector
    t = np.linspace(0, 100, seq_length)
    
    all_sequences = []
    all_metadata = []

    for i in range(num_samples):
        # 1. Baseline linear drift (random slope)
        linear_drift = np.random.uniform(0.01, 0.05) * t
        
        # 2. Slow thermal drift (low-frequency sine wave)
        thermal_drift_amp = np.random.uniform(0.5, 1.5)
        thermal_drift_freq = np.random.uniform(0.01, 0.03)
        thermal_drift = thermal_drift_amp * np.sin(2 * np.pi * thermal_drift_freq * t + np.random.uniform(0, 2*np.pi))

        # 3. Periodic interference (multiple higher frequencies)
        interference = np.zeros(seq_length)
        for _ in range(np.random.randint(2, 5)):
            amp = np.random.uniform(0.1, 0.5)
            freq = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2 * np.pi)
            interference += amp * np.sin(2 * np.pi * freq * t + phase)
            
        # 4. Random Gaussian noise
        noise = np.random.normal(0, 0.1, seq_length)
        
        # 5. Environmental effects (simulated)
        temperature = 25.0 + 5.0 * np.sin(2 * np.pi * 0.02 * t + np.random.uniform(0, 2*np.pi)) + np.random.normal(0, 0.5, seq_length)
        voltage = 5.0 + 0.1 * np.sin(2 * np.pi * 0.1 * t + np.random.uniform(0, 2*np.pi)) + np.random.normal(0, 0.02, seq_length)

        # Combine all components
        drift = linear_drift + thermal_drift + interference + noise
        
        all_sequences.append(drift)
        all_metadata.append(np.stack([temperature, voltage], axis=1))

    # Reshape for model input (samples, seq_length, features)
    sequences_np = np.array(all_sequences).reshape(num_samples, seq_length, 1)
    metadata_np = np.array(all_metadata)
    
    return sequences_np, metadata_np


def main():
    """Main function to generate, split, and save data."""
    print("Generating synthetic clock drift data...")
    
    # Parameters
    NUM_SAMPLES = 2000
    SEQ_LENGTH = 1024
    
    # Create directories if they don't exist
    processed_dir = '../data/processed'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Generate data
    sequences, metadata = generate_synthetic_data(NUM_SAMPLES, SEQ_LENGTH)
    
    print(f"Generated {sequences.shape[0]} sequences of length {sequences.shape[1]}")

    # Split data: 80% train, 10% validation, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, metadata, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Save to .npy files
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
    
    print(f"Data saved to {processed_dir}")

    # Visualize a sample sequence
    plt.figure(figsize=(12, 6))
    plt.plot(X_train[0, :, 0])
    plt.title("Sample Synthetic Clock Drift Sequence")
    plt.xlabel("Time Step")
    plt.ylabel("Drift (microseconds)")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()