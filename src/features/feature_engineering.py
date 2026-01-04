"""
Feature engineering: Create meaningful features.
Engineering → what features mean
Can: 
    - Adds information
    - Increases feature count

Examples:
    BMI → BMI group
    Age → age group
    Metabolic risk score

"""

# src/features/engineering.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Responsible ONLY for creating new features.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # BMI group
        df["bmi_group"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, np.inf],
            labels=[0, 1, 2, 3],
            right=False
        ).astype("int8")

        # Age group
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 75, np.inf],
            labels=[0, 1, 2, 3, 4],
            right=False
        ).astype("int8")

        # Metabolic risk score
        df["metabolic_risk_score"] = (
            (df["bmi"] >= 30).astype(int) +
            (df["hypertension"] == 1).astype(int) +
            (df["heart_disease"] == 1).astype(int) +
            (df["hba1c_level"] >= 6.5).astype(int)
        ).astype("int")

        # --- Smoking: collapse one-hot into ordinal ---
        df["smoking_ordinal"] = np.select(
            [
                df["smoking_no_info"] == 1,
                df["smoking_never"] == 1,
                df["smoking_former"] == 1,
                df["smoking_ever"] == 1,
                df["smoking_not_current"] == 1,
                df["smoking_current"] == 1,
            ],
            [-1, 0, 1, 1, 2, 3],
            default=-1
        ).astype("int")

        # Drop original smoking columns to avoid redundancy
        smoking_cols = [
            "smoking_history_no_info",
            "smoking_history_current",
            "smoking_history_ever",
            "smoking_history_former",
            "smoking_history_never",
            "smoking_history_not_current",
        ]

        df = df.drop(columns=[c for c in smoking_cols if c in df.columns])
        
        return df
