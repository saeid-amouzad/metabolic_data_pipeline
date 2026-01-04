"""

save feature matrix
IO → how features are stored

"""

# src/features/io.py

import pandas as pd
from pathlib import Path


def save_feature_matrix_csv(
    X: pd.DataFrame,
    path: str = "data/processed/features_with_target.csv"
) -> None:
    """
    Save feature matrix as CSV (human-readable).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(path, index=False)


def load_feature_matrix_csv(
    path: str = "data/processed/features.csv"
) -> pd.DataFrame:
    return pd.read_csv(path)

def save_feature_matrix_parquet(
    X: pd.DataFrame,
    path: str = "data/processed/features.parquet"
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(path, index=False)


def load_feature_matrix_parquet(
    path: str = "data/processed/features.parquet"
) -> pd.DataFrame:
    return pd.read_parquet(path)