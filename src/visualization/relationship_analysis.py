"""
STEP C: Relationship analysis

Purpose:
- Visual exploration of relationship between diabetes and features
- NO statistical tests here (handled elsewhere)

Supported:
- Continuous numeric features:
  * Histogram (by class)
  * Boxplot
  * Scatter (jittered)

- Categorical features (SNPs, binary, race, smoking):
  * Grouped bar chart (frequency or proportion)

"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Optional
from src.utils.validators import validate_file_exists
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RelationshipAnalyzer:

    TARGET_COL = "diabetes"

    IGNORE_COLS = {"location", "patient_id", "registry_date"}
    SNP_PREFIX = "rs"
    BINARY_COLS = {"gender", "hypertension", "heart_disease"}

    RACE_MAPPING = {
        1: "African American",
        2: "Asian",
        3: "Caucasian",
        4: "Hispanic",
        5: "Other",
    }

    SMOKING_MAPPING = {
        1: "No Info",
        2: "Current",
        3: "Ever",
        4: "Former",
        5: "Never",
        6: "Not current",
    }

    # --------------------------------------------------
    def __init__(
        self,
        *,
        csv_path: Optional[str] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        if csv_path:
            validate_file_exists(csv_path)
            self.df = pd.read_csv(csv_path)
        elif db_path and table_name:
            validate_file_exists(db_path)
            self.df = self._load_from_db(db_path, table_name)
        else:
            raise ValueError("Provide csv_path OR db_path + table_name")

        assert self.TARGET_COL in self.df.columns, "Target column missing"

    def _load_from_db(self, db_path: str, table: str) -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {table}", conn)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def analyze(
        self,
        *,
        feature: str,
        plot_type: str,
        save_path: str,
    ):
        logger.info("Step C: feature=%s plot=%s", feature, plot_type)

        if feature in self.IGNORE_COLS:
            raise ValueError(f"Ignored feature: {feature}")

        # ---- Encoded categorical features ----
        if feature in ("race (encoded)", "smoking (encoded)"):
            return self._plot_encoded_categorical(feature, save_path)

        # ---- SNPs ----
        if feature.startswith(self.SNP_PREFIX):
            return self._plot_grouped_bar(feature, save_path)

        # ---- Binary clinical ----
        if feature in self.BINARY_COLS:
            return self._plot_grouped_bar(feature, save_path)

        # ---- Continuous numeric ----
        return self._plot_numeric(feature, plot_type, save_path)

    # --------------------------------------------------
    # Numeric features
    # --------------------------------------------------
    def _plot_numeric(self, feature: str, plot_type: str, save_path: str):
        assert feature in self.df.columns, f"{feature} not found"

        df = self.df[[feature, self.TARGET_COL]].dropna()
        assert not df.empty, "No data after filtering"

        if plot_type == "hist":
            plt.figure()
            plt.hist(df[df[self.TARGET_COL] == 0][feature],
                     bins=30, alpha=0.6, label="No diabetes")
            plt.hist(df[df[self.TARGET_COL] == 1][feature],
                     bins=30, alpha=0.6, label="Diabetes")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.legend()
            plt.title(f"{feature} distribution by diabetes")

        elif plot_type == "box":
            plt.figure()
            plt.boxplot(
                [
                    df[df[self.TARGET_COL] == 0][feature],
                    df[df[self.TARGET_COL] == 1][feature],
                ],
                labels=["No diabetes", "Diabetes"],
            )
            plt.ylabel(feature)
            plt.title(f"{feature} by diabetes")

        elif plot_type == "scatter":
            plt.figure()
            jitter = np.random.uniform(-0.05, 0.05, size=len(df))
            x = df[self.TARGET_COL] + jitter
            plt.scatter(x, df[feature], alpha=0.6)
            plt.xticks([0, 1], ["No diabetes", "Diabetes"])
            plt.ylabel(feature)
            plt.title(f"{feature} (jittered)")

        else:
            raise ValueError(f"Unsupported plot type for numeric: {plot_type}")

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    # --------------------------------------------------
    # Grouped bar for categorical (SNPs, binary)
    # --------------------------------------------------
    def _plot_grouped_bar(self, feature: str, save_path: str):
        df = self.df[[feature, self.TARGET_COL]].dropna()
        assert not df.empty, "No data for categorical plot"

        counts = (
            df.groupby([feature, self.TARGET_COL])
              .size()
              .unstack(fill_value=0)
        )

        counts.plot(kind="bar", width=0.8)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.legend(["No diabetes", "Diabetes"])
        plt.title(f"{feature} by diabetes")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # --------------------------------------------------
    # Encoded race / smoking
    # --------------------------------------------------
    def _plot_encoded_categorical(self, feature: str, save_path: str):
        df = self.df.copy()

        if feature == "race (encoded)":
            df["cat"] = self._encode_race(df)
            mapping = self.RACE_MAPPING
            xlabel = "Race"
        else:
            df["cat"] = self._encode_smoking(df)
            mapping = self.SMOKING_MAPPING
            xlabel = "Smoking status"

        df = df.dropna(subset=["cat", self.TARGET_COL])
        assert not df.empty, "No encoded categorical data"

        counts = (
            df.groupby(["cat", self.TARGET_COL])
              .size()
              .unstack(fill_value=0)
        )

        counts.plot(kind="bar", width=0.8)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.legend(["No diabetes", "Diabetes"])
        plt.title(f"{feature} by diabetes")

        legend_text = "\n".join(f"{k}: {v}" for k, v in mapping.items())
        plt.gcf().text(1.02, 0.5, legend_text, fontsize=9, va="center")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    # --------------------------------------------------
    # Encoding helpers
    # --------------------------------------------------
    def _encode_race(self, df):
        mapping = {
            "raceafricanamerican": 1,
            "raceasian": 2,
            "racecaucasian": 3,
            "racehispanic": 4,
            "raceother": 5,
        }
        encoded = pd.Series(index=df.index, dtype="Int64")
        for col, code in mapping.items():
            if col in df.columns:
                encoded[df[col] == 1] = code
        return encoded

    def _encode_smoking(self, df):
        mapping = {
            "smoking_no_info": 1,
            "smoking_current": 2,
            "smoking_ever": 3,
            "smoking_former": 4,
            "smoking_never": 5,
            "smoking_not_current": 6,
        }
        encoded = pd.Series(index=df.index, dtype="Int64")
        for col, code in mapping.items():
            if col in df.columns:
                encoded[df[col] == 1] = code
        return encoded

# --------- Standalone example ----------
if __name__ == "__main__":
    analyzer = RelationshipAnalyzer(
        db_path="data/database.db",
        table_name="analysis_master",
    )

    # Continuous numeric
    analyzer.analyze(
        feature="bmi",
        plot_type="hist",
        save_path="bmi_hist_by_diabetes.png",
    )

    # Binary clinical
    analyzer.analyze(
        feature="hypertension",
        plot_type="bar",
        save_path="hypertension_by_diabetes.png",
    )

    # Encoded categorical
    analyzer.analyze(
        feature="smoking (encoded)",
        plot_type="bar",
        save_path="smoking_by_diabetes.png",
    )

    # SNP example
    analyzer.analyze(
        feature="rs9581943",
        plot_type="bar",
        save_path="rs9581943_genotype_by_diabetes.png",
    )

    print("Step C standalone examples completed.")
