from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import mat73
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_RANGES = {
    "Height": (120, 220),
    "Weight": (25, 200),
    "BMI": (13, 60),
    "SBP": (70, 250),
    "DBP": (30, 150),
}

def load_mat_file(file_path: str | Path):
    data_dict = mat73.loadmat(str(file_path))["Subset"]

    ECG = data_dict["Signals"][:, 0, :].astype("float32")
    PPG = data_dict["Signals"][:, 1, :].astype("float32")
    ABP = data_dict["Signals"][:, 2, :].astype("float32")

    df = pd.DataFrame(
        {
            "Age": data_dict["Age"].tolist(),
            "BMI": data_dict["BMI"].tolist(),
            "DBP": data_dict["DBP"].tolist(),
            "Gender": [1 if x[0] == "M" else 0 for x in data_dict["Gender"]],
            "Height": data_dict["Height"].tolist(),
            "SBP": data_dict["SBP"].tolist(),
            "Subject": [x[0] for x in data_dict["Subject"]],
            "Weight": data_dict["Weight"].tolist(),
        }
    )
    return df, ECG, PPG, ABP

def valid_range_mask(df: pd.DataFrame, ranges: dict | None = None) -> np.ndarray:
    ranges = DEFAULT_RANGES if ranges is None else ranges
    mask = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in ranges.items():
        mask &= df[col].between(lo, hi).to_numpy()
    return mask

def apply_mask(df: pd.DataFrame, ECG: np.ndarray, PPG: np.ndarray, ABP: np.ndarray, mask: np.ndarray):
    df_clean = df.loc[mask].reset_index(drop=True)
    ECG_clean = ECG[mask]
    PPG_clean = PPG[mask]
    ABP_clean = ABP[mask]
    return df_clean, ECG_clean, PPG_clean, ABP_clean

def summarize_split(df: pd.DataFrame, split: str, stage: str):
    return {
        "split": split,
        "stage": stage,
        "n_segments": int(len(df)),
        "n_subjects": int(df["Subject"].nunique()),
        "male_share": float(df["Gender"].mean()),
        "mean_age": float(df["Age"].mean()),
        "mean_sbp": float(df["SBP"].mean()),
        "mean_dbp": float(df["DBP"].mean()),
    }

def summarize_indices(df: pd.DataFrame, idx: np.ndarray, split: str, stage: str = "ready"):
    return summarize_split(df.iloc[idx], split, stage)

def make_group_split(df: pd.DataFrame, group_col: str = "Subject", test_size: float = 0.1, random_state: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))
    return np.sort(train_idx), np.sort(val_idx)

def compute_signal_stats(ECG: np.ndarray, PPG: np.ndarray, idx: np.ndarray, batch_size: int = 10000):
    def _mean_std(arr: np.ndarray):
        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            batch = arr[batch_idx].astype(np.float64, copy=False)
            total_sum += batch.sum()
            total_sq_sum += np.square(batch).sum()
            total_count += batch.size
        mean = total_sum / total_count
        var = max(total_sq_sum / total_count - mean**2, 1e-12)
        return float(mean), float(np.sqrt(var))

    ecg_mean, ecg_std = _mean_std(ECG)
    ppg_mean, ppg_std = _mean_std(PPG)

    return {
        "ecg_mean": ecg_mean,
        "ecg_std": ecg_std,
        "ppg_mean": ppg_mean,
        "ppg_std": ppg_std,
    }
