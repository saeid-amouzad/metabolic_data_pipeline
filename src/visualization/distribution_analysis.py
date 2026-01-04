# src/visualization/distribution_analysis.py
""" 
STEP B: Feature distribution analysis using:
    - Histograms
    - Boxplots (outliter can e highlighted optionally)

"""

import sqlite3
import pandas as pd
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend (safe for Flask)
import matplotlib.pyplot as plt

from src.utils.validators import validate_file_exists
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

class DistributionAnalyzer:
    """
    Feature distribution analysis for:
    - Raw CSV
    - Integrated DB table
    - Feature-selected CSV

    Produces histograms and boxplots.
    """

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

        assert not self.df.empty, "Loaded dataset is empty"

    def _load_from_db(self, db_path: str, table: str) -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {table}", conn)

    def _feature_type(self, feature: str) -> str:
        if feature.startswith("rs"):
            return "Genotypic (SNP)"
        return "Phenotypic"

    def plot_distribution(
        self,
        feature: str,
        plot_type: str,
        save_path: str,
        dataset_label: str = "Unknown dataset",
        highlight_outliers: bool = False,
    ):

        assert feature in self.df.columns

        data = self.df[feature].dropna()
        assert not data.empty

        plt.figure()

        feature_type = self._feature_type(feature)

        if plot_type == "hist":
            bins = 3 if feature_type.startswith("Genotypic") else 30
            plt.hist(data, bins=bins)
            plt.ylabel("Frequency")

        # assert plot_type in {"hist", "box"}, "Unsupported plot type"
        elif plot_type == "box":
            plt.boxplot(data, vert=False, showfliers=not highlight_outliers)

            if highlight_outliers:
                logger.info(
                    "Plotting distribution: feature=%s plot=%s highlight_outliers=%s",
                    feature, plot_type, highlight_outliers
                )

                outliers = self._flag_outliers_iqr(data)
                plt.scatter(
                    data[outliers],
                    [1] * outliers.sum(),
                    color="red",
                    s=30,
                    label="Outliers",
                    zorder=3,
                )
                plt.legend()


        plt.xlabel(feature)
        plt.title(
            f"{feature_type} distribution — {feature}\nDataset: {dataset_label}"
        )

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def _flag_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """
        Identify outliers using the IQR rule (visual only).
        Returns a boolean mask.
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return (series < lower) | (series > upper)


# -------------------------------------------------
# Standalone example
# -------------------------------------------------
if __name__ == "__main__":
    analyzer = DistributionAnalyzer(
        csv_path="data/raw/cohort/pheno_dataset_2500.csv",
    )

    analyzer.plot_distribution(
        feature="bmi",
        plot_type="box",
        save_path="bmi_box_outliers.png",
        dataset_label="Raw phenotype",
        highlight_outliers=True,
    )

    print("Distribution analysis (with outliers) completed.")
