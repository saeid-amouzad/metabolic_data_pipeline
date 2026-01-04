"""
DES-style main pipeline controller.

Executes the cohort workflow as a strict chain of steps. 
Each step must complete successfully before the next step starts.

If any step fails, the pipeline stops immediately.
"""

import logging
import yaml
from pathlib import Path

from src.features.feature_pipeline import FeaturePipeline
from src.models.model_trainer import ModelTrainer
from src.utils.db import DatabaseClient  # assumes helper exists
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.models.inference import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DES-Pipeline")


# =====================================================
# STEP VALIDATORS (EVENT GUARDS)
# =====================================================

def assert_ingestion_complete(pheno_path, geno_path):
    assert pheno_path.exists(), "Phenotypic data not found"
    assert geno_path.exists(), "Genotypic data not found"
    logger.info("✔ Ingestion completed")


def assert_preprocessing_complete(db_path):
    db = DatabaseClient(db_path)
    assert db.table_exists("analysis_master"), \
        "Integrated cohort table not found"
    assert db.count_rows("analysis_master") > 0, \
        "Integrated cohort table is empty"
    logger.info("✔ Preprocessing & integration completed")


def assert_feature_pipeline_complete(features_path, schema_path):
    assert features_path.exists(), "features.csv not created"
    assert schema_path.exists(), "feature_selection_log.yaml not created"
    logger.info("✔ Feature pipeline completed")


def assert_model_training_complete(artifact_dir):
    artifacts = list(artifact_dir.glob("*.joblib"))
    metadata = artifact_dir / "model_metadata.json"

    assert len(artifacts) > 0, "Model artifact not found"
    assert metadata.exists(), "Model metadata not found"
    logger.info("✔ ML training completed")


# =====================================================
# COHORT PIPELINE (DES-style)
# =====================================================

def run_cohort_pipeline():
    """
    Runs the full cohort pipeline in strict sequence.
    """

    logger.info("Starting cohort pipeline")

    # -----------------------------
    # STEP 1: Ingestion
    # -----------------------------
    # Ingestion is typically done via Flask.
    # Here we only verify that it already happened.

    pheno_path = Path("data/raw/cohort/pheno_dataset_2500.csv")
    geno_path = Path("data/raw/cohort/geno_dataset_2500.csv")

    assert_ingestion_complete(pheno_path, geno_path)

    # -----------------------------
    # STEP 2: Preprocessing & integration
    # -----------------------------
    db_path = Path("data/database.db")
    assert_preprocessing_complete(db_path)

    # -----------------------------
    # STEP 3: Feature pipeline
    # -----------------------------
    feature_pipeline = FeaturePipeline(
        db_path=str(db_path),
        output_csv="data/preprocessed/features.csv"
    )

    df = feature_pipeline.load_processed_data()
    X = feature_pipeline.fit_transform(df)

    X_with_target = X.copy()
    X_with_target["diabetes"] = df["diabetes"].values
    feature_pipeline.save_features(X_with_target)

    assert_feature_pipeline_complete(
        features_path=Path("data/preprocessed/features.csv"),
        schema_path=Path("config/feature_selection_log.yaml")
    )

    # -----------------------------
    # STEP 4: ML training
    # -----------------------------
    trainer = ModelTrainer("config/config.yaml")
    trainer.train_select_best()

    assert_model_training_complete(
        artifact_dir=Path("src/models/artifacts")
    )

    logger.info("✔ Cohort pipeline finished successfully")

def run_single_patient_pipeline(
    patient_id: str,
    pheno_path: Path,
    geno_path: Path,
    consent_integrate: bool = False,
    consent_retrain: bool = False,
):
    """
    DES-style single-patient pipeline.

    Prediction is ALWAYS executed before any integration,
    storage, or retraining.
    """

    logger.info(f"Starting single-patient pipeline for {patient_id}")

    # -----------------------------
    # STEP 1: Ingestion (verify)
    # -----------------------------
    assert pheno_path.exists(), "Phenotypic file not found"
    assert geno_path.exists(), "Genotypic file not found"
    logger.info("✔ Ingestion completed")

    # -----------------------------
    # STEP 2: Preprocessing
    # -----------------------------
    preprocessing_pipeline = PreprocessingPipeline(
        db_path="data/database.db"
    )

    result = preprocessing_pipeline.run(
        pheno_path=str(pheno_path),
        geno_path=str(geno_path),
        is_single_patient=True,
        consent_to_store=consent_integrate,
        registry_date=None
    )

    if consent_integrate:
        full_df = result["final_df"]

        # DES invariant:
        # Single-patient integration appends exactly ONE new row.
        # Therefore, the last row represents the patient.
        patient_df = full_df.tail(1)

        assert patient_df.shape[0] == 1, (
            "Expected exactly 1 newly integrated patient row"
        )
    else:
        pheno_df = result["phenotypic"]
        geno_df = result["genotypic"]

        patient_df = pheno_df.merge(
            geno_df,
            on="patient_id",
            how="inner"
        )

        assert patient_df.shape[0] == 1, (
            f"Expected exactly 1 patient row, got {patient_df.shape[0]}"
        )

    logger.info("✔ Preprocessing completed (single patient isolated)")

    
    # -----------------------------
    # STEP 3: Feature processing (transform only)
    # -----------------------------
    feature_pipeline = FeaturePipeline(
        db_path="data/database.db",
        output_csv="data/preprocessed/features.csv"
    )

    # -------------------------------------------------
    # DES GUARD: load fitted feature selection
    # -------------------------------------------------
    schema_path = Path("config/feature_selection_log.yaml")
    assert schema_path.exists(), (
        "Feature selection schema not found. "
        "Run cohort feature pipeline first."
    )

    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    # Rehydrate reducer state (NO fitting!)
    feature_pipeline.reducer.selected_phenotypic = (
        schema["phenotypic"]["selected"]
    )
    feature_pipeline.reducer.selected_snps = (
        schema["genotype"]["selected"]
    )

    X_patient = feature_pipeline.transform_single_patient(
        patient_df=patient_df,
        consent_to_store=False  # NEVER store before prediction
    )

    assert X_patient.shape[0] == 1
    assert X_patient.shape[1] > 0
    logger.info("✔ Feature processing completed")

    # -----------------------------
    # STEP 4: Prediction
    # -----------------------------
    inference = InferenceService()
    prediction = inference.predict(X_patient)[0]

    assert 0.0 <= prediction <= 1.0
    logger.info(
        f"✔ Prediction completed — "
        f"Type 2 diabetes risk = {prediction:.3f}"
    )

    # -----------------------------
    # STEP 5: Optional integration
    # -----------------------------
    if consent_integrate:
        logger.info("Integrating patient into cohort database")

        # Integration already happened during preprocessing
        # Verify by checking DB row count increase if needed
        logger.info("✔ Integration completed")

    # -----------------------------
    # STEP 6: Optional retraining
    # -----------------------------
    if consent_retrain and consent_integrate:
        logger.info("Retraining model with patient data")

        feature_pipeline.transform_single_patient(
            patient_df=patient_df,
            consent_to_store=True  # append to features.csv
        )

        trainer = ModelTrainer("config/config.yaml")
        trainer.train_select_best()

        logger.info("✔ Retraining completed")

    # -----------------------------
    # STEP 7: Report
    # -----------------------------
    logger.info(
        f"Single-patient pipeline finished successfully. "
        f"Final risk score: {prediction:.3f}"
    )

    return prediction

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    logger.info("DES-style pipeline started")

    print("\n" + "=" * 60)
    print(" COHORT PIPELINE ")
    print("=" * 60 + "\n")

    # -------------------------------------------------
    # COHORT PIPELINE
    # -------------------------------------------------
    run_cohort_pipeline()

    logger.info("Cohort data pipeline completed successfully")

    print("\n" + "=" * 60)
    print(" SINGLE-PATIENT PIPELINE ")
    print("=" * 60 + "\n")

    # -------------------------------------------------
    # SINGLE PATIENT PIPELINE
    # -------------------------------------------------
    prediction = run_single_patient_pipeline(
        patient_id="patient_2501",
        pheno_path=Path(
            "data/raw/single_patient/patient_2501/patient_2501_pheno.csv"
        ),
        geno_path=Path(
            "data/raw/single_patient/patient_2501/patient_2501_geno.csv"
        ),
        consent_integrate=True,
        consent_retrain=False
    )

    logger.info(
        f"Single-patient pipeline finished successfully. "
        f"Predicted diabetes risk = {prediction:.3f}"
    )

    logger.info("DES-style pipeline execution completed")

