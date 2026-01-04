# src/ingestion/ingest_data.py
"""
Decides where raw data lives
Stores upload metadata (including registry_date)
Enforces raw storage consent
Does no data transformation

"""

import os
import shutil

from src.utils.logger import setup_logger
from src.utils.db import get_connection
from src.utils.validators import (
    validate_file_exists,
    validate_data_type,
    validate_file_format,
    validate_tabular_not_empty,
    validate_image_file,
    validate_registry_date
)
from src.utils.schema_validator import assert_feature_compatibility

logger = setup_logger(__name__)


class IngestData:
    """
    Orchestrates ingestion of raw data from UI.
    """

    def __init__(self, db_path, raw_base_dir):
        self.db_path = db_path
        self.raw_base_dir = raw_base_dir

    def ingest(
        self,
        file_path,
        data_type,
        patient_id=None,
        registry_date=None,
        consent_raw_storage=False,
        source="ui"
    ):
        """
        Main ingestion entry point.

        - patient_id = None   → cohort ingestion
        - patient_id provided → single-patient ingestion
        """

        logger.info(
            "Ingestion started | file=%s | type=%s",
            file_path, data_type
        )

        # ---------- Validation ----------
        validate_file_exists(file_path)
        validate_data_type(data_type)
        validate_file_format(file_path, data_type)
        validate_registry_date(registry_date)

        ext = os.path.splitext(file_path)[1].lower()

        if ext in {".csv", ".xlsx", ".json"}:
            validate_tabular_not_empty(file_path)
            assert_feature_compatibility(file_path, data_type)

        else:
            validate_image_file(file_path)

        # ---------- Raw data storage ----------
        stored_path = None
        if consent_raw_storage:
            stored_path = self._store_raw_data(file_path, patient_id)

        # ---------- Metadata ----------
        self._store_metadata(
            patient_id=patient_id,
            filename=os.path.basename(file_path),
            data_type=data_type,
            registry_date=registry_date,
            consent=consent_raw_storage,
            is_single_patient=bool(patient_id),
            source=source
        )

        logger.info("Ingestion completed | file=%s", file_path)

        return {
            "status": "success",
            "stored_path": stored_path
        }

    def _store_raw_data(self, file_path, patient_id):
        """
        Stores raw data on disk.

        - Cohort        → data/raw/cohort/
        - Single patient → data/raw/single_patient/{patient_id}/
        """
        target_dir = (
            os.path.join(self.raw_base_dir, "single_patient", patient_id)
            if patient_id
            else os.path.join(self.raw_base_dir, "cohort")
        )

        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(
            target_dir,
            os.path.basename(file_path)
        )

        shutil.copy(file_path, target_path)

        logger.info("Raw file stored at %s", target_path)

        return target_path

    def _store_metadata(self, **kwargs):
        """
        Writes ingestion metadata to SQLite.
        """
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO datasets (
                patient_id,
                filename,
                data_type,
                source,
                registry_date,
                consent_raw_storage,
                is_single_patient
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            kwargs["patient_id"],
            kwargs["filename"],
            kwargs["data_type"],
            kwargs["source"],
            kwargs["registry_date"],
            kwargs["consent"],
            kwargs["is_single_patient"]
        ))

        conn.commit()
        conn.close()

# ===========================
# Stand alone test
# ===========================
if __name__ == "__main__":

    # python -m src.ingestion.ingest_data

    from src.utils.db import init_db

    DB_PATH = "data/database.db"
    RAW_DIR = "data/raw"

    init_db(DB_PATH)

    manager = IngestData(
        db_path=DB_PATH,
        raw_base_dir=RAW_DIR
    )

    # =========================
    # 1. Cohort PHENO CSV
    # =========================
    print("\n>>> Ingesting cohort phenotypic data <<<")

    result = manager.ingest(
        file_path="I:/technical_task/raw data import/pheno_dataset_2500.csv",
        data_type="phenotypic",
        patient_id=None,
        registry_date="2025-12-25",
        consent_raw_storage=True,
        source="manual_test"
    )

    print("Cohort phenotypic result:", result)

    # =========================
    # 1. Cohort GENOTYPIC CSV
    # =========================
    print("\n>>> Ingesting cohort genotypic data <<<")

    result = manager.ingest(
        file_path="I:/technical_task/raw data import/geno_dataset_2500.csv",
        data_type="genotypic",
        patient_id=None,
        registry_date="2025-12-25",
        consent_raw_storage=True,
        source="manual_test"
    )

    print("Cohort genotypic result:", result)

    # =========================
    # 2. Single-patient PHENO
    # =========================
    print("\n>>> Ingesting single-patient phenotypic data <<<")

    result = manager.ingest(
        file_path="I:/technical_task/raw data import/patient_2501_pheno.csv",
        data_type="phenotypic",
        patient_id="patient_2501",
        registry_date="2025-12-25",
        consent_raw_storage=True,
        source="manual_test"
    )

    print("Single patient phenotypic result:", result)

    # =========================
    # 3. Single-patient GENO
    # =========================
    print("\n>>> Ingesting single-patient genotypic data <<<")

    result = manager.ingest(
        file_path="I:/technical_task/raw data import/patient_2501_geno.csv",
        data_type="genotypic",
        patient_id="patient_2501",
        registry_date="2025-12-25",
        consent_raw_storage=True,
        source="manual_test"
    )

    print("Single patient genotypic result:", result)
