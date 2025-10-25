# TemporalClock/README.md
# TemporalClock: Ultra-Precision Temporal Drift Prediction

## Concept
**TemporalClock** is a deep learning project designed to forecast microsecond-level clock drift and synchronization errors with high precision. Inspired by the stability and complex dynamics of Cesium atomic clocks, this model aims to provide not only accurate predictions but also reliable uncertainty estimates.

## Why Precision Matters
In many critical systems, time is the most important measurement. Ultra-precise clock synchronization is fundamental for:
- **Satellite Navigation & Communication (e.g., DRDO, ISRO, NASA):** Ensuring accurate location data and coherent communication links.
- **Defense and Aerospace (e.g., DRDO):** Synchronizing radar arrays, electronic warfare systems, and secure communication networks.
- **Telecommunications:** Maintaining the stability of 5G/6G networks.
- **Financial Trading:** High-frequency trading relies on microsecond-level timestamps.

This project tackles the challenge of predicting clock drift, which is affected by complex factors like thermal variations, periodic interference, and long-term aging effects.

## Architecture: A Hybrid Approach
To capture the intricate dynamics of clock drift, TemporalClock uses a novel hybrid architecture that fuses time-domain and frequency-domain features.

**Pipeline Diagram:**
[Synthetic Data] → [FFT Features] → [1D CNN] → [LSTM] → [Transformer] → [Predicted Drift + Uncertainty]


**Code**
- **FFT-based Spectral Features:** The raw time-series is transformed into the frequency domain to extract features like dominant frequencies, spectral entropy, and spectrograms. This explicitly models periodic drifts.
- **1D CNN Encoder:** A lightweight convolutional neural network processes the spectrogram to learn local patterns in the frequency domain over time.
- **LSTM Block:** A Long Short-Term Memory network captures short-term temporal dependencies and local trends in the time-series data.
- **Transformer Encoder:** A Transformer block with multi-head self-attention is used to model long-range dependencies and global context across the entire sequence.
- **Uncertainty Head:** The model outputs two values for each time step: the mean prediction (μ) and the log-variance (σ²). This allows us to quantify the model's confidence, which is crucial for risk assessment in critical systems.

## Dataset
The model is trained on synthetic data generated to mimic the behavior of Cesium atomic clocks. The data includes:
- Baseline linear drift
- Slow thermal drift components
- Multi-frequency periodic interference
- Random Gaussian noise

This approach allows for controlled experimentation and a robust understanding of the model's capabilities.

## How to Train and Evaluate

### 1. Setup
Clone the repository and install the required packages:

git clone https://your-repo-link/TemporalClock.git
cd TemporalClock
pip install -r requirements.txt

### **2. Generate Data**
First, run the preprocessing script to generate and save the synthetic dataset.
code
Bash
python src/preprocess.py

### **3. Train the Model**
Start the training process. This script will log all parameters and metrics to MLflow.
code
Bash
python src/train.py```
You can view the experiments by running `mlflow ui` in your terminal.

### 4. Evaluate the Model
After training, evaluate the best model on the test set. This will print metrics and save result plots to `/experiments/results/`.
```bash

