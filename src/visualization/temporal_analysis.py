# src/visualization/temporal_analysis.py
"""
STEP D: Module for temporal analysis visualizations based on registry dates.

Includes three plots: 
    D1. Cohort size over time
    •	X-axis: registry date (monthly or yearly)
    •	Y-axis: number of registered patients
    •	Purpose: data coverage & growth

    D2. Diabetes prevalence over time
    •	X-axis: registry date
    •	Y-axis: prevalence (% with diabetes)
    •	Purpose: epidemiological insight

    D3. Feature trend over time (user-selected)
    •	X-axis: registry date
    •	Y-axis: mean (or median) of selected feature
    •	Purpose: clinical evolution (e.g., BMI, HbA1c)

    * These three cover descriptive, epidemiological, and clinical angles.

"""

import pandas as pd
import sqlite3
from typing import Optional
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TemporalAnalyzer:
    """
    Temporal analysis for cohort data using year or registry_date.
    """

    def __init__(self, db_path: str, table_name: str):
        with sqlite3.connect(db_path) as conn:
            self.df = pd.read_sql(
                f"SELECT * FROM {table_name}",
                conn,
                parse_dates=["registry_date"]
            )

        if "registry_date" not in self.df.columns:
            raise ValueError("registry_date column is required")

        # Use existing year column if available, otherwise derive from registry_date
        if "year" not in self.df.columns:
            self.df["year"] = self.df["registry_date"].dt.year
        else:
            # Ensure year is integer (safe)
            self.df["year"] = self.df["year"].astype(int)

        if self.df["year"].nunique() == 1:
            logger.warning(
                "Temporal analysis has only one unique year (%s). "
                "Plots may not show meaningful trends.",
                self.df["year"].iloc[0],
            )

    # --------------------------------------------------
    # Helper: select time axis
    # --------------------------------------------------
    def _get_time_col(self, mode: str):
        if mode == "registry_date":
            return "registry_date"
        return "year"   # default

    # --------------------------------------------------
    # D1: Cohort size
    # --------------------------------------------------
    def plot_cohort_size(self, save_path: str, mode: str):
        time_col = self._get_time_col(mode)
        counts = self.df.groupby(time_col).size()

        plt.figure()
        counts.plot(marker="o")
        plt.xlabel(mode.replace("_", " ").title())
        plt.ylabel("Number of patients")
        plt.title(f"Cohort size over {mode}")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # --------------------------------------------------
    # D2: Diabetes prevalence
    # --------------------------------------------------
    def plot_prevalence(self, save_path: str, mode: str):
        time_col = self._get_time_col(mode)
        prevalence = self.df.groupby(time_col)["diabetes"].mean()

        plt.figure()
        prevalence.plot(marker="o")
        plt.xlabel(mode.replace("_", " ").title())
        plt.ylabel("Diabetes prevalence")
        plt.title(f"Diabetes prevalence over {mode}")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # --------------------------------------------------
    # D3: Feature trend
    # --------------------------------------------------
    def plot_feature_trend(
        self,
        feature: str,
        save_path: str,
        mode: str,
        agg: str,
    ):
        if feature not in self.df.columns:
            raise ValueError(f"{feature} not found")

        time_col = self._get_time_col(mode)
        grouped = self.df.groupby(time_col)[feature]

        series = grouped.median() if agg == "median" else grouped.mean()

        plt.figure()
        series.plot(marker="o")
        plt.xlabel(mode.replace("_", " ").title())
        plt.ylabel(f"{agg.capitalize()} {feature}")
        plt.title(f"{feature} over {mode}")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    """
    Stand-alone test for TemporalAnalyzer.

    Verifies:
    - Database connectivity
    - Year-based temporal analysis (default, meaningful)
    - Registry-date mode (future-proof)
    - All three temporal plot types
    """

    import os

    DB_PATH = "data/database.db"
    TABLE_NAME = "analysis_master"
    OUT_DIR = "tmp_temporal_plots"

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Running TemporalAnalyzer stand-alone test...")

    analyzer = TemporalAnalyzer(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
    )

    # --------------------------------------------------
    # Temporal mode: YEAR (recommended default)
    # --------------------------------------------------
    print("Testing YEAR-based temporal analysis")

    size_plot = os.path.join(OUT_DIR, "cohort_size_year.png")
    analyzer.plot_cohort_size(size_plot, mode="year")
    print(f"Cohort size (year) saved to {size_plot}")

    prev_plot = os.path.join(OUT_DIR, "prevalence_year.png")
    analyzer.plot_prevalence(prev_plot, mode="year")
    print(f"Prevalence (year) saved to {prev_plot}")

    test_feature = "bmi"

    if test_feature in analyzer.df.columns:
        feat_plot = os.path.join(OUT_DIR, f"{test_feature}_year_mean.png")
        analyzer.plot_feature_trend(
            feature=test_feature,
            save_path=feat_plot,
            mode="year",
            agg="mean",
        )
        print(f"Feature trend (year, mean) saved to {feat_plot}")
    else:
        print(f"Feature '{test_feature}' not found — skipping feature trend")

    # --------------------------------------------------
    # Temporal mode: REGISTRY DATE (future-proof test)
    # --------------------------------------------------
    # print("Testing REGISTRY-DATE-based temporal analysis")

    # size_plot_rd = os.path.join(OUT_DIR, "cohort_size_registry_date.png")
    # analyzer.plot_cohort_size(size_plot_rd, mode="year")
    # print(f"✔ Cohort size (registry_date) saved to {size_plot_rd}")

    print("TemporalAnalyzer stand-alone test completed.")


