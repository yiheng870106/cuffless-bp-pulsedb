from __future__ import annotations

import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, input_dim=1250 * 2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_branch(kernel_size):
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

        self.branch1 = conv_branch(5)
        self.branch2 = conv_branch(7)
        self.branch3 = conv_branch(15)

        self.fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        b1 = self.branch1(x).squeeze(-1)
        b2 = self.branch2(x).squeeze(-1)
        b3 = self.branch3(x).squeeze(-1)

        x = torch.cat([b1, b2, b3], dim=1)
        return self.fc(x)

class RNN(nn.Module):
    def __init__(self, model_type="GRU", hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.model_type = model_type

        rnn_dropout = dropout if num_layers > 1 else 0.0

        if model_type == "RNN":
            self.rnn = nn.RNN(input_size=2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_size=2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        else:
            raise ValueError("model_type must be 'RNN', 'LSTM', or 'GRU'")

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        if self.model_type == "LSTM":
            _, (hn, _) = self.rnn(x)
        else:
            _, hn = self.rnn(x)

        return self.fc(hn[-1])

class Transformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, 1250, d_model))
        self.input_fc = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)
    def forward(self, x):
        x = x.transpose(1,2)          # (B, seq_len, channels)
        x = self.input_fc(x) + self.pos_emb
        x = self.transformer(x)       # [B, seq_len, d_model]
        x = x.mean(dim=1)
        # x = x[-1,:,:]                 # last timestep
        return self.fc(x)

class Transformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size=9, stride=5, padding=4),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.downsample(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

class FCAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(1250,625),
            nn.ReLU(),
            nn.Linear(625, latent_dim)
        )
        # Decoder (mirror)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 625),
            nn.ReLU(),
            nn.Linear(625, 1250)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
