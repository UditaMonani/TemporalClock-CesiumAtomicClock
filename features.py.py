# TemporalClock/src/features.py

import numpy as np
import os
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

def compute_fft_features(sequence, sample_rate=1.0):
    """
    Computes spectral features from a time-series sequence using FFT.

    Args:
        sequence (np.ndarray): A 1D time-series sequence.
        sample_rate (float): The sample rate of the sequence.

    Returns:
        dict: A dictionary containing spectral features:
            - 'dominant_freq': The frequency with the highest amplitude.
            - 'spectral_entropy': The entropy of the power spectrum.
            - 'band_power': Power in low, mid, and high frequency bands.
    """
    n = len(sequence)
    yf = rfft(sequence)
    xf = rfftfreq(n, 1 / sample_rate)
    
    power_spectrum = np.abs(yf)**2
    
    # Dominant frequency
    dominant_freq = xf[np.argmax(power_spectrum)]
    
    # Spectral entropy
    power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
    spectral_entropy = entropy(power_spectrum_norm)
    
    # Band power
    low_band = np.sum(power_spectrum[xf < 0.1 * (sample_rate / 2)])
    mid_band = np.sum(power_spectrum[(xf >= 0.1 * (sample_rate / 2)) & (xf < 0.4 * (sample_rate / 2))])
    high_band = np.sum(power_spectrum[xf >= 0.4 * (sample_rate / 2)])
    
    total_power = np.sum(power_spectrum)
    band_power = np.array([low_band, mid_band, high_band]) / total_power if total_power > 0 else np.zeros(3)
    
    return {
        'dominant_freq': dominant_freq,
        'spectral_entropy': spectral_entropy,
        'band_power': band_power
    }

def get_spectrogram(sequence, nperseg=64, noverlap=32):
    """
    Generates a spectrogram from a time-series sequence.
    This is a simplified version; for production, consider using scipy.signal.spectrogram.

    Args:
        sequence (np.ndarray): 1D time-series data.
        nperseg (int): Length of each segment.
        noverlap (int): Overlap between segments.

    Returns:
        np.ndarray: The spectrogram tensor.
    """
    step = nperseg - noverlap
    shape = ( (sequence.shape[0] - noverlap) // step, nperseg)
    strides = (sequence.strides[0] * step, sequence.strides[0])
    
    # Create overlapping windows
    windowed_data = np.lib.stride_tricks.as_strided(sequence, shape=shape, strides=strides)
    
    # Apply a window function (e.g., Hann)
    windowed_data = windowed_data * np.hanning(nperseg)
    
    # Compute FFT on each window
    spectrogram = np.abs(rfft(windowed_data, axis=1))**2
    
    # Log scale for better feature representation
    spectrogram = np.log1p(spectrogram)
    
    return spectrogram.T # (freq_bins, time_steps)


def process_features(data_dir='../data/processed/'):
    """
    Loads raw data, computes features, normalizes, and returns tensors.

    Args:
        data_dir (str): Directory containing the processed .npy files.

    Returns:
        dict: A dictionary containing all processed data splits:
            - 'X_train_ts', 'X_val_ts', 'X_test_ts': Time-series data.
            - 'X_train_spec', 'X_val_spec', 'X_test_spec': Spectrogram data.
            - 'y_train', 'y_val', 'y_test': Target data (which is the input shifted).
    """
    print("Processing features...")
    
    # Load data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))

    # For forecasting, the target is a shifted version of the input
    y_train, y_val, y_test = X_train, X_val, X_test

    # Normalize time-series data
    scaler = StandardScaler()
    X_train_ts = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val_ts = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_ts = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # Generate spectrograms
    X_train_spec = np.array([get_spectrogram(seq.flatten()) for seq in X_train_ts])
    X_val_spec = np.array([get_spectrogram(seq.flatten()) for seq in X_val_ts])
    X_test_spec = np.array([get_spectrogram(seq.flatten()) for seq in X_test_ts])

    print(f"Time-series shape: {X_train_ts.shape}")
    print(f"Spectrogram shape: {X_train_spec.shape}")
    
    return {
        'X_train_ts': X_train_ts, 'X_val_ts': X_val_ts, 'X_test_ts': X_test_ts,
        'X_train_spec': X_train_spec, 'X_val_spec': X_val_spec, 'X_test_spec': X_test_spec,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'scaler': scaler # Return scaler to inverse transform predictions
    }

if __name__ == '__main__':
    feature_data = process_features()
    # You can add visualization of a spectrogram here
    plt.figure(figsize=(10, 6))
    plt.imshow(feature_data['X_train_spec'][0], aspect='auto', origin='lower')
    plt.title('Sample Spectrogram')
    plt.ylabel('Frequency Bins')
    plt.xlabel('Time Steps')
    plt.colorbar(label='Log Power')
    plt.show()