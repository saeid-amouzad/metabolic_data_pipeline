"""
Load raw files
Validate schema & values
Clean and encode data
Output clean DataFrames
No responsibility for consent or storage

"""

# src/preprocessing/preprocess_geno.py

import pandas as pd
from src.preprocessing.base import BasePreprocessor


class GenotypePreprocessor(BasePreprocessor):
    """
    QC and preprocessing for SNP dosage data.
    """

    def load(self):
        self.df = pd.read_csv(self.input_path)

    def validate(self):
        assert "patient_id" in self.df.columns, \
            "patient_id missing in genotypic data"

    def process(self):
        df = self.df.copy()

        patient_id = df["patient_id"]
        df = df.drop(columns=["patient_id"])

        df = df.apply(pd.to_numeric, errors="coerce")

        # Dosage validation
        invalid = ~df.isin([0, 1, 2]) & ~df.isna()
        assert not invalid.any().any(), "Invalid genotype dosage"

        # Missingness filtering
        df = df.loc[:, df.isna().mean() <= 0.10]

        # MAF filtering
        maf = df.mean(skipna=True) / 2
        df = df.loc[:, maf >= 0.01]

        # Imputation
        df = df.fillna(df.mean())

        df.insert(0, "patient_id", patient_id)

        self.df = df

    def output(self):
        return self.df
