from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
import time
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FIG_DIR = Path("/content/drive/MyDrive/cuffless-bp-pulsedb/figures")
metrics_path = Path("/content/drive/MyDrive/cuffless-bp-pulsedb/results/model_metrics.csv")

def plot_true_vs_pred(test, pred, model_name, split_name, variable_name):
  plt.figure(figsize=(6,6))
  plt.scatter(test, pred, alpha=0.3, s=8)
  plt.plot([min(test), max(test)], [min(test), max(test)], 'r--')  # y=x line
  plt.xlabel('True')
  plt.ylabel('Predicted')
  plt.title(f"True vs. Predicted {model_name}_{variable_name}_{split_name}")
  plt.savefig(f"{FIG_DIR}/True vs. Predicted {model_name}_{variable_name}_{split_name}.png", dpi=300, bbox_inches="tight")
  plt.show()

def make_features(ECG, PPG):
    return np.concatenate([ECG, PPG], axis=1)

def compute_bp_metrics(model_name, split_name, y_true, y_pred, train_time, n_epochs=None):
    metrics = []

    for j, target in enumerate(["SBP", "DBP"]):
        err = y_pred[:, j] - y_true[:, j]

        metrics.append({
            "Model": model_name,
            "Split": split_name,
            "Variable": target,
            "ME": np.mean(err),
            "SDE": np.std(err, ddof=1),
            "MAE": np.mean(np.abs(err)),
            "R2": r2_score(y_true[:, j], y_pred[:, j]),
            "Time": train_time,
            "Number of epochs": n_epochs
        })

    metrics.to_csv(
        metrics_path,
        mode="a",
        header=not metrics_path.exists(),
        index=False
    )

    return pd.DataFrame(metrics)

class PulseDataset(Dataset):
    def __init__(self, ECG, PPG, df_labels):
        self.ECG = ECG.astype('float32')
        self.PPG = PPG.astype('float32')
        self.labels = df_labels[['SBP','DBP']].values.astype('float32')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = np.stack([self.ECG[idx], self.PPG[idx]], axis=0)  # shape=(2, seq_len)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

def evaluate_loop(name, model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()*X.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss

def train_model(name, model, train_loader, val_loader, epochs=100, lr=0.001, loss_fn = nn.MSELoss()):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # early stopping setting
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    tol = 0.0001
    all_training_loss = []
    all_val_loss = []
    n_epochs = epochs

    start = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*X.size(0)
        train_loss /= len(train_loader.dataset)
        all_training_loss.append(train_loss)

        # validation
        val_loss = evaluate_loop(name, model, val_loader, loss_fn)
        all_val_loss.append(val_loss)

        # early stopping
        if best_val_loss - val_loss >= tol:
          best_val_loss = val_loss
          patience_counter = 0
          torch.save(model.state_dict(), "best_model.pt")
        else:
          patience_counter += 1
          if patience_counter >= patience:
            n_epochs = epoch+1 - patience
            break
    end = time.time()
    model.load_state_dict(torch.load("best_model.pt"))

    return model, end - start, n_epochs, all_training_loss, all_val_loss

def evaluate_model(name, model_class, train_loader, val_loader, test_loader, test_loader_free):
    model = model_class()
    trained_model, time, n_epochs, all_training_loss, all_val_loss = train_model(name, model, train_loader, val_loader)

    # predict
    trained_model.eval()
    preds = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            out = trained_model(X).cpu().numpy()
            preds.append(out)
    y_pred = np.vstack(preds)

    SBP_pred = y_pred[:,0]
    DBP_pred = y_pred[:,1]

    # predict calfree
    trained_model.eval()
    preds_free = []
    with torch.no_grad():
        for X, y in test_loader_free:
            X = X.to(device)
            out = trained_model(X).cpu().numpy()
            preds_free.append(out)
    y_pred_free = np.vstack(preds_free)

    SBP_pred_free = y_pred_free[:,0]
    DBP_pred_free = y_pred_free[:,1]

    del model
    del trained_model

    return SBP_pred, DBP_pred, SBP_pred_free, DBP_pred_free, time, n_epochs, all_training_loss, all_val_loss

# fully connected autoencoder (AE)
from src.models import FCAutoencoder

def evaluate_loop_AE(model, val_loader, loss_fn = nn.MSELoss()):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (X,) in val_loader:
            X = X.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, X).item()*X.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss

def train_model_AE(model, train_loader, epochs=100, lr=0.001, loss_fn = nn.MSELoss()):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # early stopping setting
    prev_loss = float('inf')
    patience = 10
    patience_counter = 0
    n_epochs = epochs

    start = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for (X,) in train_loader:
            X = X.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*X.size(0)
        train_loss /= len(train_loader.dataset)

        # convergence
        if prev_loss > train_loss:
          prev_loss = train_loss
          patience_counter = 0
          torch.save(model.state_dict(), "best_model.pt")
        else:
          patience_counter += 1
          if patience_counter >= patience:
            n_epochs = epoch+1 - patience
            break
    end = time.time()
    model.load_state_dict(torch.load("best_model.pt"))

    return model, evaluate_loop_AE(model,train_loader), end - start, n_epochs

def elbow_AE(latent_list, ABP_loader, ABP_org):
  results = []
  ABP_gen = {}

  for ld in latent_list:
      # print("Training AE with latent dimension:", ld)
      model = FCAutoencoder(latent_dim=ld)
      trained_model, mse, training_time, epochs = train_model_AE(model, ABP_loader)
      # print(f"MSE: {mse}, time: {training_time}, epochs: {epochs}")

      results.append({
        'latent_dim': ld,
        'mse': mse,
      })

      #generate one reconstructed ABP signal
      trained_model.eval()
      with torch.no_grad():
          x0 = torch.from_numpy(ABP_org).float().to(device)
          x0 = x0.unsqueeze(0)

          x_rec = trained_model(x0).cpu().squeeze(0).numpy()  # remove batch dim

          ABP_gen[ld] = {
              "latent_dim": ld,
              "generated": x_rec
          }

  return results, ABP_gen
