# src/features/feature_pipeline.py
"""
Feature Pipeline (DB → Features → CSV)
Pipeline determines:
    - when and how things run
    - Orchestrates feature engineering + feature selection
    - Can be used for:
        - Cohort (training / analysis)
        - Single patient (prediction)

| Scenario               | Data source | Feature eng | Feature selection | Prediction | Feature storage            | Notes                           |
| ---------------------- | ------------| ------------| ------------------| ---------- | ---------------------------| ------------------------------- |
| Cohort + consent = YES | database    | ✅ Yes     |✅ Fit + Transform | ❌ No     | ✅ Always stored           | Used to create ML-ready dataset |
| Single + consent = YES | UI          | ✅ Yes     | ❌ Transform only | ✅ Yes    | ✅ Append after prediction | GDPR-compliant + prediction     |
| Single + consent = NO  | UI          | ✅ Yes     | ❌ Transform only | ✅ Yes    | ❌ Never stored            | Prediction-only                 |

"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_reduction import FeatureReducer
from src.features.feature_io import save_feature_matrix_csv

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeaturePipeline:
    """
    Orchestrates feature engineering + feature selection.

    Works for:
    - Cohort-level data (from SQLite)
    - Single-patient data (after preprocessing)
    """

    def __init__(
        self,
        db_path: str = "data/database.db",
        output_csv: str = "data/processed/features.csv",
        drop_cols: Optional[list] = None,
    ):
        self.db_path = db_path
        self.output_csv = output_csv
        self.drop_cols = drop_cols or ["location", "race"]

        self.engineer = FeatureEngineer()
        self.reducer = FeatureReducer()
        self.reducer.load_selected_features()


    # ------------------------------------------------------------------
    # Database access
    # ------------------------------------------------------------------
    def load_processed_data(self, table: str = "analysis_master") -> pd.DataFrame:
        LOGGER.info("Loading processed data from SQLite: %s", table)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)

        assert not df.empty, "Loaded dataset is empty"
        assert "diabetes" in df.columns

        return df

    # ------------------------------------------------------------------
    # Core feature pipeline
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Starting feature engineering")

        cols_to_drop = [
            c for c in df.columns
            if c == "location" or c.startswith("race:")
        ]

        df = df.drop(columns=cols_to_drop)

        df_eng = self.engineer.transform(df)

        phenotypic_cols = [
            "gender", "age", "hypertension", "heart_disease",
            "bmi", "hba1c_level", "blood_glucose_level",
            "smoking_ordinal",
            "bmi_group", "age_group", "metabolic_risk_score",
        ]

        snp_cols = [c for c in df_eng.columns if c.startswith("rs")]

        self.reducer.fit(
            df=df_eng,
            phenotypic_cols=phenotypic_cols,
            snp_cols=snp_cols,
            target_col="diabetes",
        )

        X = self.reducer.transform(df_eng)
        LOGGER.info("Final feature count: %d", X.shape[1])

        return X

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_features(self, X: pd.DataFrame) -> None:
        save_feature_matrix_csv(X, self.output_csv)
        LOGGER.info("Saved feature matrix to %s", self.output_csv)

    # ------------------------------------------------------------------
    # Single-patient workflow
    # ------------------------------------------------------------------
    def transform_single_patient(
        self,
        patient_df: pd.DataFrame,
        consent_to_store: bool = False,
    ) -> pd.DataFrame:

        """
        Applies feature engineering to a single patient.
        Uses already-fitted feature selection.
        """

        LOGGER.info("Processing single patient (inference-only)")

        cols_to_drop = [
            c for c in patient_df.columns
            if c == "location" or c.startswith("race:")
        ]

        patient_df = patient_df.drop(columns=cols_to_drop, errors="ignore")

        # Feature engineering (same as cohort)
        df_eng = self.engineer.transform(patient_df)

        # Just SELECT features — no reduction
        X_patient = df_eng[
            self.reducer.selected_phenotypic + self.reducer.selected_snps
        ]

        # ----------------------------------
        # Optional append (only for retrain)
        # ----------------------------------
        if consent_to_store:
            if "diabetes" not in patient_df.columns:
                raise AssertionError(
                    "Consent given, but target label 'diabetes' is missing."
                )

            X_with_target = X_patient.copy()
            X_with_target["diabetes"] = patient_df["diabetes"].iloc[0]

            self._append_to_feature_store(X_with_target)

        return X_patient

    
    def _append_to_feature_store(self, X_patient: pd.DataFrame):
        path = Path(self.output_csv)

        if path.exists():
            X_existing = pd.read_csv(path)
            assert list(X_existing.columns) == list(X_patient.columns), (
                "Feature schema mismatch when appending single patient data"
            )
            X_all = pd.concat([X_existing, X_patient], ignore_index=True)
        else:
            X_all = X_patient

        # Save updated feature store
        save_feature_matrix_csv(X_all, self.output_csv)

        LOGGER.info(
            "Feature store updated | total_rows=%d",
            len(X_all)
        )

        assert Path(self.output_csv).exists(), "Feature file was not saved"

        

# ----- stand alone example -----
if __name__ == "__main__":
    pipeline = FeaturePipeline(
        db_path="data/database.db",
        output_csv="data/processed/features.csv",
    )

    # ---- Cohort ----
    df = pipeline.load_processed_data()
    X = pipeline.fit_transform(df)

    X_with_target = X.copy()
    X_with_target["diabetes"] = df["diabetes"].values
    pipeline.save_features(X_with_target)

    print("Cohort feature pipeline is completed.")
    print("Final cohort feature count:", X.shape[1])
    print("Final cohort features:", X.columns.tolist(), "\n")

    # ======================================================
    # Single-patient test data (AFTER preprocessing) 
    #   simulates output AFTER preprocessing + integration
    # ======================================================

    single_patient_df = pd.DataFrame(
        [{
            # --------------------
            # Identifiers / metadata
            # --------------------
            "patient_id": "patient_123",
            "year": 2025,
            "registry_date": "2025-01-10",

            # --------------------
            # Demographics
            # --------------------
            "gender": 1,
            "age": 52,
            "location": "SE",

            # Race (should be DROPPED)
            "raceafricanamerican": 0,
            "raceasian": 1,
            "racecaucasian": 0,
            "racehispanic": 0,
            "raceother": 0,

            # --------------------
            # Clinical variables
            # --------------------
            "hypertension": 1,
            "heart_disease": 0,
            "bmi": 31.2,
            "hba1c_level": 6.8,
            "blood_glucose_level": 145,

            # Target (ignored for inference)
            "diabetes": 1,

            # --------------------
            # Smoking (one-hot, preprocessed)
            # --------------------
            "smoking_never": 0,
            "smoking_former": 1,
            "smoking_current": 0,
            "smoking_not_current": 0,
            "smoking_no_info": 0,
            "smoking_ever": 0,

            # --------------------
            # Genotype (subset example)
            # --------------------
            "rs61652270": 0,
            "rs60004782": 1,
            "rs9554188": 2,
            "rs9581927": 0,
            "rs9581929": 1,
            "rs7985481": 0,
            "rs9581931": 2,
            "rs4424773": 1,
            "rs9554193": 0,
            "rs7995917": 1,
            "rs7993114": 0,
            "rs11618581": 2,
            "rs9554197": 1,
            "rs11618036": 0,
            "rs11618832": 1,
            "rs11616678": 0,
            "rs11618052": 2,
            "rs9579127": 1,
            "rs7999100": 0,
            "rs8000004": 1,
            "rs9579128": 0,
            "rs2297316": 2,
            "rs9581943": 1,
        }]
    )


    # ---- Consent flag (UI-driven) ----
    consent_to_store = True  # change to True to append features

    X_patient = pipeline.transform_single_patient(
        patient_df=single_patient_df,
        consent_to_store=consent_to_store,
    )

    print("\n Single-patient feature extraction complete.")
    print("Single-patient feature shape:", X_patient.shape)
    print("Single-patient features:", X_patient.columns.tolist())
    print(X_patient)

