# TemporalClock/src/models.py

import torch
import torch.nn as nn
import math

class TemporalClockModel(nn.Module):
    """
    A hybrid deep learning model for time-series forecasting.

    This model combines a 1D CNN for spectral features, an LSTM for short-term
    temporal patterns, and a Transformer Encoder for long-range dependencies.
    It outputs both a mean prediction and a log-variance for uncertainty estimation.
    """
    def __init__(self, ts_input_dim, spec_input_dim, d_model=128, nhead=4, num_encoder_layers=2, dim_feedforward=256, lstm_hidden_dim=64, lstm_layers=1, dropout=0.1):
        """
        Args:
            ts_input_dim (int): Input dimension of the time-series data (usually 1).
            spec_input_dim (int): Number of frequency bins in the spectrogram.
            d_model (int): The number of expected features in the transformer encoder/decoder inputs.
            nhead (int): The number of heads in the multi-head attention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the transformer encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            lstm_hidden_dim (int): The number of features in the LSTM hidden state.
            lstm_layers (int): Number of recurrent layers for the LSTM.
            dropout (float): The dropout value.
        """
        super(TemporalClockModel, self).__init__()

        self.d_model = d_model

        # 1. 1D CNN Encoder for Spectrogram
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=spec_input_dim, out_channels=d_model // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=d_model // 2, out_channels=d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 2. LSTM Block for Time-Series
        self.ts_input_projection = nn.Linear(ts_input_dim, d_model)
        self.lstm = nn.LSTM(d_model, lstm_hidden_dim, lstm_layers, batch_first=True, bidirectional=True)
        self.lstm_output_projection = nn.Linear(lstm_hidden_dim * 2, d_model) # Bidirectional

        # 3. Transformer Encoder Block for Long-Range Dependencies
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 4. Dense Regression Head
        self.fc_mean = nn.Linear(d_model, 1)
        self.fc_log_var = nn.Linear(d_model, 1) # Output log-variance for stability

        self.init_weights()

    def init_weights(self):
        """Initializes weights for the linear and embedding layers."""
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-initrange, initrange)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, ts_data, spec_data):
        """
        Forward pass of the model.
        
        Args:
            ts_data (torch.Tensor): Time-series data with shape (batch, seq_len, ts_input_dim).
            spec_data (torch.Tensor): Spectrogram data with shape (batch, freq_bins, time_steps).
        
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Mean prediction (batch, seq_len, 1).
                - torch.Tensor: Log-variance prediction (batch, seq_len, 1).
        """
        # CNN path for spectrogram
        spec_features = self.cnn_encoder(spec_data) # (batch, d_model, reduced_time_steps)
        spec_features = spec_features.permute(0, 2, 1) # (batch, reduced_time_steps, d_model)
        # Upsample to match sequence length (simple repeat)
        scale_factor = ts_data.size(1) // spec_features.size(1)
        spec_features_upsampled = spec_features.repeat_interleave(scale_factor, dim=1)
        # Pad if necessary
        if spec_features_upsampled.size(1) < ts_data.size(1):
            padding_size = ts_data.size(1) - spec_features_upsampled.size(1)
            padding = torch.zeros(spec_features_upsampled.size(0), padding_size, self.d_model, device=ts_data.device)
            spec_features_upsampled = torch.cat([spec_features_upsampled, padding], dim=1)

        # LSTM path for time-series
        ts_proj = self.ts_input_projection(ts_data)
        lstm_out, _ = self.lstm(ts_proj)
        lstm_features = self.lstm_output_projection(lstm_out)

        # Combine features
        combined_features = ts_proj + lstm_features + spec_features_upsampled
        combined_features = self.pos_encoder(combined_features * math.sqrt(self.d_model))
        
        # Transformer path
        transformer_out = self.transformer_encoder(combined_features)
        
        # Regression head
        mean = self.fc_mean(transformer_out)
        log_var = self.fc_log_var(transformer_out)
        
        return mean, log_var

    def count_parameters(self):
        """Counts the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)