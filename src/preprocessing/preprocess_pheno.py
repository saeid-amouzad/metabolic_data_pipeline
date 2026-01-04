"""
Load raw files
Validate schema & values
Clean and encode data
Output clean DataFrames
No responsibility for consent or storage

"""

# src/preprocessing/preprocess_pheno.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.preprocessing.base import BasePreprocessor

class PhenotypePreprocessor(BasePreprocessor):
    """
    QC and preprocessing for phenotypic data.
    Output is analysis-ready and SQL-safe.
    """
    # Drop location and race columns
    REQUIRED_COLS = {
        "patient_id", "year", "gender", "age",
        "hypertension", "heart_disease", "smoking_history",
        "bmi", "hba1c_level", "blood_glucose_level",
         "diabetes"
    }

    def load(self):
        self.df = pd.read_csv(self.input_path)
        self.df.columns = (
        self.df.columns
        .str.lower()               # lowercase
        .str.strip()               # remove leading/trailing spaces
        .str.replace(" ", "_")     # replace spaces with underscores
        .str.replace(":", "", regex=False)  # remove ':' characters
    )

    def validate(self):
        assert self.REQUIRED_COLS.issubset(self.df.columns), \
            "Missing required phenotypic columns"

    def process(self):
        df = self.df.copy()

        df = df.drop_duplicates()

        # Missing values
        df["gender"] = df["gender"].fillna("unknown")
        df["smoking_history"] = df["smoking_history"].fillna("unknown")

        for col in [
            "year", "age", "bmi", "hypertension",
            "heart_disease", "hba1c_level",
            "blood_glucose_level"
        ]:
            df[col] = df[col].fillna(df[col].median())

        # Type casting
        df["gender"] = LabelEncoder().fit_transform(df["gender"])

        # Enforce canonical smoking categories
            # Canonical categories observed in the FULL cohort
        SMOKING_CATEGORIES = [
            "never",
            "former",
            "current",
            "not current",
            "No Info",
            "ever"
        ]
        df["smoking_history"] = (
            df["smoking_history"]
            .astype("category")
            .cat.set_categories(SMOKING_CATEGORIES)
        )

        df = pd.get_dummies(
            df,
            columns=["smoking_history"],
            prefix="smoking",
            dtype="int"
        )

        df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)  # spaces → _
        .str.replace(":", "", regex=False)     # remove ':' characters
    )

        df["patient_id"] = df["patient_id"].astype(int)
        df["year"] = df["year"].astype(int)
        df["age"] = df["age"].astype(int)
        df["hypertension"] = df["hypertension"].astype(int)
        df["heart_disease"] = df["heart_disease"].astype(int)
        df["blood_glucose_level"] = df["blood_glucose_level"].astype(int)
        df["diabetes"] = df["diabetes"].astype(int)
        df["bmi"] = df["bmi"].astype("float32")
        df["hba1c_level"] = df["hba1c_level"].astype("float32")

        # # ---Compute quartiles and IQR (optional) - e,g, for BMI
        # # --- Cap outliers based on 1st and 99th percentiles
        # # -- Values smaller than lower are replaced by lower
        # # -- Values larger than upper are replaced by upper
        # # -- Values in between stay unchanged

        # # Define capping thresholds
        # bmi_cap_lower = df["bmi"].quantile(0.01)
        # bmi_cap_upper = df["bmi"].quantile(0.99)

        # # Apply capping
        # df["bmi"] = df["bmi"].clip(lower=bmi_cap_lower, upper=bmi_cap_upper)

        # Clinical QC
        assert df["age"].between(0, 120).all(), "Invalid age values detected"
        assert df["bmi"].between(10, 82).all(), "Invalid BMI values detected"
        assert df["hba1c_level"].between(3, 20).all(), "Invalid HbA1c values detected"
        assert df["blood_glucose_level"].between(40, 600).all(), "Invalid glucose values detected"

        #
        EXPECTED_COLUMNS = [
            "patient_id", "year","gender", "age", "location",
            "raceafricanamerican", "raceasian", "racecaucasian",
            "racehispanic", "raceother", "hypertension", "heart_disease",
            "bmi", "hba1c_level", "blood_glucose_level", "diabetes",
            "smoking_history_no_info", "smoking_history_current",
            "smoking_history_ever", "smoking_history_former",
            "smoking_history_never", "smoking_history_not_current"]

        # Schema validation: ensuring the presence of all required variables
        # assert set(EXPECTED_COLUMNS).issubset(df.columns), "Phenotypic feature mismatch"

        # # Final feature count check
        # EXPECTED_FEATURE_COUNT = 22
        # assert df.shape[1] == EXPECTED_FEATURE_COUNT, (
        #     f"Feature mismatch: expected {EXPECTED_FEATURE_COUNT}, "
        #     f"got {df.shape[1]}")

        self.df = df

    def output(self):
        return self.df
