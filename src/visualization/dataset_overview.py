# src/visualization/dataset_overview.py
"""
STEP A: Dataset-level overview & quality checks

"""


import os
import sqlite3
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

from src.utils.validators import validate_file_exists
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class DatasetSummary:
    dataset_name: str
    n_rows: int
    n_columns: int
    columns: list
    missing_values: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]]
    date_range: Optional[Dict[str, str]]


class DatasetOverviewAnalyzer:
    """
    Dataset-level overview & quality checks.

    Supports:
    - CSV datasets
    - SQLite tables (e.g. analysis_master)
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        csv_path: Optional[str] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        self.dataset_name = dataset_name

        if csv_path:
            validate_file_exists(csv_path)
            self.df = self._load_csv(csv_path)
        elif db_path and table_name:
            validate_file_exists(db_path)
            self.df = self._load_from_db(db_path, table_name)
        else:
            raise ValueError(
                "Either csv_path or (db_path + table_name) must be provided"
            )

        assert not self.df.empty, "Loaded dataset is empty"

    # -----------------------------
    # Loaders
    # -----------------------------
    def _load_csv(self, path: str) -> pd.DataFrame:
        logger.info("Loading CSV dataset: %s", path)
        return pd.read_csv(path)

    def _load_from_db(self, db_path: str, table: str) -> pd.DataFrame:
        logger.info("Loading table '%s' from SQLite: %s", table, db_path)
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {table}", conn)

    # -----------------------------
    # Analysis helpers
    # -----------------------------
    def _missing_values(self) -> Dict[str, int]:
        return self.df.isna().sum().to_dict()

    def _numeric_summary(self) -> Dict[str, Dict[str, float]]:
        numeric_df = self.df.select_dtypes(include=["int", "float"])
        if numeric_df.empty:
            return {}

        desc = numeric_df.describe().T
        return desc[["mean", "std", "min", "max"]].round(4).to_dict(orient="index")

    def _date_range(self) -> Optional[Dict[str, str]]:
        if "registry_date" not in self.df.columns:
            return None

        dates = pd.to_datetime(self.df["registry_date"], errors="coerce")
        return {
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
        }

    # -----------------------------
    # Public API
    # -----------------------------
    def generate_summary(self) -> DatasetSummary:
        logger.info("Generating dataset overview: %s", self.dataset_name)

        return DatasetSummary(
            dataset_name=self.dataset_name,
            n_rows=self.df.shape[0],
            n_columns=self.df.shape[1],
            columns=list(self.df.columns),
            missing_values=self._missing_values(),
            numeric_summary=self._numeric_summary(),
            date_range=self._date_range(),
        )


# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":

    analyzer = DatasetOverviewAnalyzer(
        dataset_name="Integrated cohort (DB)",
        db_path="data/database.db",
        table_name="analysis_master",
    )

    summary = analyzer.generate_summary()

    print("\nDATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows: {summary.n_rows}")
    print(f"Columns: {summary.n_columns}")
    print(f"Registry date range: {summary.date_range}")
