"""
Preprocessing pipeline used in Phase 1:
- Missing values → mean
- Duplicates removal
- Binary dtype conversion
- Min–Max scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(df[numeric].mean())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def convert_binary_to_bool(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            df[col] = df[col].astype(bool)
    return df


def minmax_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    numeric = df.select_dtypes(include=[np.number, "bool"]).columns
    df[numeric] = scaler.fit_transform(df[numeric])
    return df, scaler


def preprocess(df: pd.DataFrame):
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = convert_binary_to_bool(df)
    df, scaler = minmax_scale(df)
    return df, scaler
