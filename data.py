import pandas as pd
import numpy as np

# import torch
# from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, RobustScaler

import pickle
import os

# class TimeSeriesDataset(Dataset):
#     def __init__(self, X, y, seq_len):
#         self.X = torch.tensor(X).float()
#         self.y = torch.tensor(y).float()
#         self.seq_len = seq_len
    
#     def __len__(self):
#         return len(self.X) - self.seq_len 
        
#     def __getitem__(self, idx):
#         return (self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len])

def add_lag_features(df, target_cols, lags=[1, 3, 12]):
    """    
    Args:
        df: DataFrame with time series data
        target_cols: Columns to create lags for
        lags: [1, 3, 12] - month, quarter, year lags
    """
    df = df.copy()
    
    for col in target_cols:
        # Simple lags: 1, 3, 12
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Drop rows with NaN from lag_12 (first 12 rows)
    df = df.dropna()
    
    return df
    
def add_seasonal_features(df):
    """
    Add cyclical seasonal encoding.
    
    Based on:
    - Hyndman & Athanasopoulos (2021): Seasonal decomposition
    - Makridakis et al. (2020): M4 competition best practices
    """
    df = df.copy()
    
    # Cyclical encoding (better than one-hot for neural networks)
    month = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    quarter = df.index.quarter  
    df['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
    
    # Philippine tax calendar indicators
    # April 15 - individual tax deadline
    # April 30 - corporate tax deadline
    df['is_tax_season'] = df.index.month.isin([3, 4, 5]).astype(int)
    
    # December year-end collections
    df['is_year_end'] = (df.index.month == 12).astype(int)
    
    return df

def transform_data(data, save_path="Transforms/default/scaler.pkl"):
    """Transform data and save scaler"""
    
    # Convert to absolute path if it's relative
    if not os.path.isabs(save_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, save_path)
    
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)

    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    
    return data_scaled, scaler

def inverse_transform(data, scaler=None, load_path=None):
    """Inverse transform using a scaler object or a saved scaler file."""
    if scaler is None:
        if load_path is None:
            raise ValueError("Must provide either a scaler object or a load_path.")
        if not os.path.isabs(load_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            load_path = os.path.join(base_dir, load_path)
        with open(load_path, "rb") as f:
            scaler = pickle.load(f)
    return scaler.inverse_transform(data)

def split_data(X, use_val=True):
    if use_val:
        train_size = int(0.6 * len(X))
        val_size = int(0.2 * len(X))

        train = X[:train_size]
        val = X[train_size:train_size + val_size]
        test = X[train_size + val_size:]

        return train, val, test

def pairwise_corr(df, feature_cols, target_series, threshold=0.85):
    corr_matrix = df[feature_cols].corr().abs()
    target_corr = df[feature_cols].corrwith(target_series).abs().to_dict()
    
    to_drop = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if corr_matrix.iloc[i, j] > threshold:
                fi, fj = feature_cols[i], feature_cols[j]
                if fi in to_drop or fj in to_drop:
                    continue
                ci = target_corr.get(fi, 0) if not np.isnan(target_corr.get(fi, 0)) else 0
                cj = target_corr.get(fj, 0) if not np.isnan(target_corr.get(fj, 0)) else 0
                if ci < cj:
                    to_drop.add(fi)
                    print(f"    Dropping {fi:30s} (r={corr_matrix.iloc[i,j]:.2f} with {fj}, target_r={ci:.3f})")
                else:
                    to_drop.add(fj)
                    print(f"    Dropping {fj:30s} (r={corr_matrix.iloc[i,j]:.2f} with {fi}, target_r={cj:.3f})")
    
    kept = [f for f in feature_cols if f not in to_drop]
    print(f"    Kept {len(kept)}/{len(feature_cols)} features")
    return kept

def target_corr(df, feature_cols, target_series, min_corr=0.3):
    corr = df[feature_cols].corrwith(target_series).abs().sort_values(ascending=False)
    
    selected = []
    for feat, c in corr.items():
        if np.isnan(c):
            continue
        marker = "✓" if c >= min_corr else "✗"
        print(f"    {marker} {feat:30s} r={c:.3f}")
        if c >= min_corr:
            selected.append(feat)
    return selected