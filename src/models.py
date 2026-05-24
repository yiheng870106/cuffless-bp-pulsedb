from __future__ import annotations

import numpy as np
import pandas as pd
from kneed import KneeLocator
import time
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1250*2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # SBP, DBP
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()

        # Branch 1 — small receptive field
        self.branch1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Branch 2 — medium receptive field
        self.branch2 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Branch 3 — large receptive field
        self.branch3 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # total output: 32 + 32 + 32 = 96
        self.fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # SBP, DBP
        )

    def forward(self, x):
        b1 = self.branch1(x).squeeze(-1)  # [batch, 32]
        b2 = self.branch2(x).squeeze(-1)  # [batch, 32]
        b3 = self.branch3(x).squeeze(-1)  # [batch, 32]

        # return self.fc(b2)

        x = torch.cat([b1, b2, b3], dim=1)  # [batch, 96]
        return self.fc(x)

class RNN(nn.Module):
    def __init__(self, model_type='RNN', hidden_size=64, num_layers=1):
        super().__init__()
        self.model_type = model_type
        if model_type=='RNN':
            self.rnn = nn.RNN(2, hidden_size, num_layers, batch_first=True)
        elif model_type=='LSTM':
            self.rnn = nn.LSTM(2, hidden_size, num_layers, batch_first=True)
        elif model_type=='GRU':
            self.rnn = nn.GRU(2, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("model_type must be 'RNN','LSTM', or 'GRU'")
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = x.transpose(1,2)  # (B, seq_len, channels)
        if self.model_type=='LSTM':
            out, (hn, cn) = self.rnn(x)
        else:
            out, hn = self.rnn(x)
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
