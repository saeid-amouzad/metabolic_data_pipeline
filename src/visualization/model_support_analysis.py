# src/visualization/model_support_analysis.py

"""
STEP E: Model support analysis visualizations.

Includes:
F1. Feature correlation heatmap
F2. Confusion matrix
F3. Metrics loader
F4. ROC curve
F5. Precision–Recall curve
F6. Statistical feature tests (t-test, chi-square)

| Plot                 | Data source           |
| ---------------------| --------------------- |
| Correlation heatmap  | features.csv          |
| Confusion matrix     | model artifacts       |
| ROC / PR             | model artifacts       |
| t-test p-values      | database (cohort)     |
| Chi-square p-values  | database (cohort)     |

"""

import json
from pathlib import Path

import sqlite3
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.multitest import multipletests

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # for Flask / server environments
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelSupportAnalyzer:
    """
    Provides feature selection and model support diagnostics.

    Includes:
    - Post-training model evaluation (ROC, PR, confusion matrix)
    - Feature correlation analysis (features.csv)
    - Cohort-level statistical screening (t-test, chi-square) using database data
    """

    def __init__(
        self,
        features_csv: str,
        artifacts_dir: str,
        target_col: str = "diabetes",
    ):
        self.features_csv = Path(features_csv)
        self.artifacts_dir = Path(artifacts_dir)
        self.target_col = target_col

        assert self.features_csv.exists(), "features.csv not found"
        assert self.artifacts_dir.exists(), "Artifacts directory not found"

        self.df = pd.read_csv(self.features_csv)
        assert target_col in self.df.columns, "Target column missing"

    # --- Statistical feature testing (Step E) ---
    ALPHA = 0.05
    DB_TARGET_COL = "diabetes"
    SNP_PREFIX = "rs"

    BINARY_COLS = {"gender", "hypertension", "heart_disease"}

    RACE_COLS = [
        "raceafricanamerican",
        "raceasian",
        "racecaucasian",
        "racehispanic",
        "raceother",
    ]

    SMOKING_COLS = [
        "smoking_no_info",
        "smoking_current",
        "smoking_ever",
        "smoking_former",
        "smoking_never",
        "smoking_not_current",
    ]

    # --------------------------------------------------
    # F1. Feature correlation heatmap
    # --------------------------------------------------
    def plot_correlation_heatmap(self, save_path: str):
        logger.info("Generating feature correlation heatmap")

        corr = self.df.drop(columns=[self.target_col]).corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"shrink": 0.75},
        )

        plt.title("Feature correlation heatmap")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # --------------------------------------------------
    # F2. Confusion matrix
    # --------------------------------------------------
    def plot_confusion_matrix(self, save_path: str):
        logger.info("Generating confusion matrix")

        y_true, y_pred = self._load_predictions()

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No diabetes", "Diabetes"],
        )

        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # --------------------------------------------------
    # F3. Metrics
    # --------------------------------------------------
    def load_metrics(self) -> dict:
        metadata_path = self.artifacts_dir / "model_metadata.json"
        assert metadata_path.exists(), "model_metadata.json not found"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info("Loaded model metrics")
        return metadata["metrics"]

    # --------------------------------------------------
    # F4. ROC curve
    # --------------------------------------------------
    def plot_roc_curve(self, save_path: str):
        y_true, _, y_prob = self._load_predictions(full=True)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # --------------------------------------------------
    # F5. Precision–Recall curve
    # --------------------------------------------------
    def plot_pr_curve(self, save_path: str):
        y_true, _, y_prob = self._load_predictions(full=True)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # --------------------------------------------------
    # Internal helper
    # --------------------------------------------------
    def _load_predictions(self, full=False):
        """
        Load predictions saved during training.
        """
        y_true = pd.read_csv(self.artifacts_dir / "y_true.csv").values.ravel()
        y_pred = pd.read_csv(self.artifacts_dir / "y_pred.csv").values.ravel()

        if full:
            y_prob = pd.read_csv(self.artifacts_dir / "y_prob.csv").values.ravel()
            return y_true, y_pred, y_prob

        return y_true, y_pred

    # --------------------------------------------------
    # Check if predictions exist
    # --------------------------------------------------  
    def has_predictions(self) -> bool:
        return (
            (self.artifacts_dir / "y_true.csv").exists()
            and (self.artifacts_dir / "y_pred.csv").exists()
            and (self.artifacts_dir / "y_prob.csv").exists()
        )


    # ======================================================
    # Statistical tests for feature support
    # ======================================================
    # helpers to load cohort data and perform statistical tests
    def _load_cohort_data(self, db_path: str, table_name: str) -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)

    @staticmethod
    def _chi_square(feature, target):
        ct = pd.crosstab(feature, target)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return np.nan
        _, p, _, _ = chi2_contingency(ct)
        return p

    def _encode_ohe_group(self, df, cols):
        encoded = pd.Series(index=df.index, dtype="Int64")
        for i, col in enumerate(cols):
            if col in df.columns:
                encoded[df[col] == 1] = i
        return encoded
    
    # main method to compute statistical tests
    def compute_statistical_feature_tests(
        self,
        db_path: str,
        table_name: str,
    ) -> pd.DataFrame:
        """
        Run t-test and chi-square tests using cohort data from database.
        """
        df = self._load_cohort_data(db_path, table_name)
        results = []

        ignore_cols = {"location", "patient_id", "registry_date"}

        feature_cols = [
            c for c in df.columns
            if c not in ignore_cols
            and c != self.DB_TARGET_COL
            and c not in self.RACE_COLS
            and c not in self.SMOKING_COLS
        ]

        for feature in feature_cols:
            sub = df[[feature, self.DB_TARGET_COL]].dropna()
            if sub.empty:
                continue

            x = sub[feature]
            y = sub[self.DB_TARGET_COL]

            # SNP → chi-square
            if feature.startswith(self.SNP_PREFIX):
                p = self._chi_square(x, y)
                test_group = "chi2"

            # Binary → chi-square
            elif feature in self.BINARY_COLS:
                p = self._chi_square(x, y)
                test_group = "chi2"

            # Continuous → t-test
            elif pd.api.types.is_numeric_dtype(x):
                x0, x1 = x[y == 0], x[y == 1]
                if len(x0) < 2 or len(x1) < 2:
                    continue
                _, p = ttest_ind(x0, x1, equal_var=False)
                test_group = "ttest"

            else:
                continue

            results.append({
                "feature": feature,
                "test_group": test_group,
                "p_value": p
            })

        # Race
        p = self._chi_square(
            self._encode_ohe_group(df, self.RACE_COLS),
            df[self.DB_TARGET_COL]
        )
        results.append({
            "feature": "race",
            "test_group": "chi2",
            "p_value": p
        })

        # Smoking
        p = self._chi_square(
            self._encode_ohe_group(df, self.SMOKING_COLS),
            df[self.DB_TARGET_COL]
        )
        results.append({
            "feature": "smoking",
            "test_group": "chi2",
            "p_value": p
        })

        res = pd.DataFrame(results)

        mask = res["p_value"].notna()
        rejected, p_fdr, _, _ = multipletests(
            res.loc[mask, "p_value"],
            alpha=self.ALPHA,
            method="fdr_bh"
        )

        res.loc[mask, "p_fdr"] = p_fdr
        res.loc[mask, "significant"] = rejected

        self.stat_results_ = res
        return res
    
    # Plotting p-value groups
    def _plot_pvalue_group(self, df, title: str, save_path: str):
        x = np.arange(len(df))
        width = 0.35

        plt.figure(figsize=(14, 5))
        plt.bar(x - width/2, df["p_value"], width, label="Raw p-value")
        plt.bar(x + width/2, df["p_fdr"], width, label="FDR-corrected p-value")

        plt.axhline(
            self.ALPHA,
            linestyle="--",
            linewidth=2,
            label=f"α = {self.ALPHA}"
        )

        plt.xticks(x, df["feature"], rotation=45, ha="right")
        plt.ylabel("p-value")
        plt.yscale("log")
        plt.title(title)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    # Plot methdods for t-test and chi2
    def plot_ttest_pvalues(
        self,
        db_path: str,
        table_name: str,
        save_path: str,
    ):
        res = self.compute_statistical_feature_tests(db_path, table_name)
        df = res[res["test_group"] == "ttest"]
        self._plot_pvalue_group(
            df,
            "t-test: continuous features",
            save_path
        )

    def plot_chi2_pvalues(
        self,
        db_path: str,
        table_name: str,
        save_path: str,
    ):
        res = self.compute_statistical_feature_tests(db_path, table_name)
        df = res[res["test_group"] == "chi2"]
        self._plot_pvalue_group(
            df,
            "Chi-square: categorical & SNP features",
            save_path
        )

# ======================================================
# Stand-alone test
# ======================================================
if __name__ == "__main__":
    from pathlib import Path

    OUT_DIR = Path("tmp_model_support")
    OUT_DIR.mkdir(exist_ok=True)

    analyzer = ModelSupportAnalyzer(
        features_csv="data/processed/features.csv",
        artifacts_dir="src/models/artifacts",
    )

    # -------------------------------
    # Feature-based diagnostics
    # -------------------------------
    analyzer.plot_correlation_heatmap(OUT_DIR / "correlation.png")
    print("Correlation heatmap saved.")

    analyzer.plot_confusion_matrix(OUT_DIR / "confusion_matrix.png")
    print("Confusion matrix saved.")

    analyzer.plot_roc_curve(OUT_DIR / "roc_curve.png")
    print("ROC curve saved.")

    analyzer.plot_pr_curve(OUT_DIR / "pr_curve.png")
    print("PR curve saved.")

    metrics = analyzer.load_metrics()
    print("Loaded metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # -------------------------------
    # Cohort-level statistical screening
    # (Step E – Feature Selection & Modeling Support)
    # -------------------------------
    DB_PATH = "data/database.db"
    TABLE_NAME = "analysis_master"

    analyzer.plot_ttest_pvalues(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        save_path=OUT_DIR / "ttest_feature_significance.png",
    )
    print("t-test p-value plot saved.")

    analyzer.plot_chi2_pvalues(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        save_path=OUT_DIR / "chi2_feature_significance.png",
    )
    print("Chi-square p-value plot saved.")
