"""
End-to-end preprocessing + integration pipeline.

Orchestrates preprocessing
Enforces policy rules:
    cohort must have stored raw data when consented
    single-patient can run without storage
Passes registry_date forward
Consent to store data is required for any DB write.
Policy matrix:

| Case           | is_single_patient | consent_to_store | Result                     |
| -------------- | ----------------- | ---------------- | -------------------------- |
| Cohort         | False             | ❌ False          | **BLOCK (raise error)**    |
| Cohort         | False             | ✅ True           | Store + integrate          |
| Single patient | True              | ❌ False          | Process only (no DB write) |
| Single patient | True              | ✅ True           | Store + integrate          |


"""

# src/preprocessing/preprocessing_pipeline.py

from src.preprocessing.preprocess_pheno import PhenotypePreprocessor
from src.preprocessing.preprocess_geno import GenotypePreprocessor
from src.preprocessing.integrator import DataIntegrator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PreprocessingPipeline:
    """
    End-to-end preprocessing + integration pipeline.
    """

    def __init__(self, db_path):
        self.db_path = db_path

    def run(
        self,
        pheno_path,
        geno_path,
        *,
        is_single_patient,
        registry_date=None,
        consent_to_store=False
    ):
        """
        End-to-end preprocessing with consent enforcement.
        Parameters

        pheno_path : str | None
            Path to phenotypic raw data
        geno_path : str | None
            Path to genotypic raw data
        is_single_patient : bool
            True if data belongs to one patient
        registry_date : str | None
            Date provided by user (YYYY-MM-DD)

        consent_to_store : bool
            True if consent to store data is given
        """

        # -----------------------------
        # POLICY ENFORCEMENT
        # -----------------------------
        if not is_single_patient and not consent_to_store:
            raise AssertionError(
                "Cohort preprocessing is not allowed without consent to store data."
            )

        if not is_single_patient:
            assert pheno_path is not None, \
                "Cohort preprocessing requires phenotypic raw data"

            assert geno_path is not None, \
                "Cohort preprocessing requires genotypic raw data"

        logger.info(
            "Starting preprocessing | single_patient=%s | date=%s",
            is_single_patient, registry_date
        )

        # -----------------------------
        # Phenotypic preprocessing
        # -----------------------------
        pheno_proc = PhenotypePreprocessor(pheno_path)
        pheno_proc.load()
        pheno_proc.validate()
        pheno_proc.process()
        pheno_df = pheno_proc.output()

        # -----------------------------
        # Genotypic preprocessing
        # -----------------------------
        geno_proc = GenotypePreprocessor(geno_path)
        geno_proc.load()
        geno_proc.validate()
        geno_proc.process()
        geno_df = geno_proc.output()

        # -----------------------------
        # SINGLE PATIENT — NO CONSENT
        # -----------------------------
        if is_single_patient and not consent_to_store:
            logger.info(
                "Single-patient preprocessing completed (no storage, prediction-only)"
            )
            # ✅ always return a dataframe for next step
            final_df = pheno_df.merge(geno_df, on="patient_id", how="inner")

            #assert len(final_df) == len(pheno_df), "Genotype mismatch after merge"
            assert not final_df.empty, "No matching genotype found for patient"

            return {
                "final_df": final_df,
                "stored": False
            }

        # -----------------------------
        # INTEGRATION & STORAGE
        # -----------------------------
        integrator = DataIntegrator(self.db_path)

        final_df = integrator.integrate(
            pheno_df,
            geno_df,
            registry_date=registry_date,
            is_single_patient=is_single_patient
        )

        logger.info(
            "Preprocessing completed | rows=%d | stored=True",
            len(final_df)
        )

        return {
            "final_df": final_df,
            "stored": True
        }

    
if __name__ == "__main__":
    pipeline = PreprocessingPipeline(
        db_path="data/database.db"
    )

    # SINGLE PATIENT preprocessing (prediction-only)
    result = pipeline.run(
        pheno_path="data/raw/single_patient/patient_2501/patient_2501_pheno.csv",
        geno_path="data/raw/single_patient/patient_2501/patient_2501_geno.csv",
        is_single_patient=True,
        consent_to_store=False,
        registry_date="2025-12-25"
    )

    print("\nSingle patient features shape:", result["final_df"].shape)
    print("Single patient consent status:", result["stored"])
    #print("Column names: ", list(result["final_df"].columns))
    print("Single patient preprocessing completed (no storage).\n")

    
    # SINGLE PATIENT preprocessing WITH consent
    result = pipeline.run(
        pheno_path="data/raw/single_patient/patient_2501/patient_2501_pheno.csv",
        geno_path="data/raw/single_patient/patient_2501/patient_2501_geno.csv",
        is_single_patient=True,
        consent_to_store=True,
        registry_date="2025-12-25"
    )

    final_df = result["final_df"]

    print("Integrated shape:", final_df.shape)
    print("Single patient data stored with consent.\n")

    # -------- COHORT preprocessing (large raw data) with consent -------
    result = pipeline.run(
        pheno_path="data/raw/cohort/pheno_dataset_2500.csv",
        geno_path="data/raw/cohort/geno_dataset_2500.csv",
        is_single_patient=False,
        consent_to_store=True,
        registry_date="2025-12-25"
    )

    final_df = result["final_df"]
    print("Final integrated shape:", final_df.shape, "\n")

    # COHORT preprocessing WITHOUT consent (EXPECTED TO FAIL)
    try:
        pipeline.run(
            pheno_path="data/raw/cohort/pheno_dataset_2500.csv",
            geno_path="data/raw/cohort/geno_dataset_2500.csv",
            is_single_patient=False,
            consent_to_store=False,
            registry_date="2025-12-25"
        )
        print("\nCohort preprocessing, NO CONSENT.")
    except AssertionError as e:
        print("\nCohort preprocessing blocked as expected.")
        print("\nReason:", str(e))
    

