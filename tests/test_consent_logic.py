# src: tests/test_consent_logic.py
"""
Tests for consent logic in preprocessing pipeline.
----------
Run the tests from project root:
pytest tests/test_consent_logic.py

Expected outputs: 4 passed in 0.8s
----------
✔ GDPR consent is respected
✔ Single-patient prediction-only flow is safe
✔ Cohort processing cannot bypass consent
✔ Database is not polluted accidentally
✔ Future refactors cannot break this logic silently

"""

import sqlite3
import pandas as pd
import pytest
from pathlib import Path
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline


# -----------------------------
# Helpers: dummy input data
# -----------------------------
def create_dummy_pheno_csv(path: Path):

    # REQUIRED_COLS = {
    #     "patient_id", "year", "gender", "age",
    #     "hypertension", "heart_disease", "smoking_history",
    #     "bmi", "hba1c_level", "blood_glucose_level",
    #      "diabetes"
    # }

    df = pd.DataFrame({
        "patient_id": [1],
        "year": [2015],
        "age": [50],
        "bmi": [25.0],
        "gender": ["male"],
        "location": [1],
        "race": [1],
        "hypertension": [0],
        "heart_disease": [0],
        "hba1c_level": [6.0],
        "blood_glucose_level": [120],
        "smoking_history": ["never"],
        "diabetes": [0],
    })

    df.to_csv(path, index=False)


def create_dummy_geno_csv(path: Path):
    df = pd.DataFrame({
        "patient_id": [1],
        "rs61652270": [1],
        "rs60004782": [1],
        "rs9554188": [1],
        "rs9581927": [2],
        "rs9581929": [1],
        "rs7985481": [2],
        "rs9581931": [2],
        "rs4424773": [1],
        "rs9554193": [1],
        "rs7995917": [2],
        "rs7993114": [1],
        "rs11618581": [2],
        "rs9554197": [1],
        "rs11618036": [2],
        "rs11618832": [1],
        "rs11616678": [1],
        "rs11618052": [2],
        "rs9579127": [2],
        "rs7999100": [1],
        "rs8000004": [1],
        "rs9579128": [2],
        "rs2297316": [1],
        "rs9581943": [1]
    })
    df.to_csv(path, index=False)


# -----------------------------
# Tests
# -----------------------------
def test_single_patient_no_consent_not_stored(tmp_path):
    """
    Single patient WITHOUT consent:
    - preprocessing runs
    - NO SQLite tables are created
    """

    pheno = tmp_path / "pheno.csv"
    geno = tmp_path / "geno.csv"
    db_path = tmp_path / "test.db"

    create_dummy_pheno_csv(pheno)
    create_dummy_geno_csv(geno)

    pipeline = PreprocessingPipeline(db_path=str(db_path))

    result = pipeline.run(
        pheno_path=str(pheno),
        geno_path=str(geno),
        is_single_patient=True,
        consent_to_store=False,
        registry_date="2025-12-25",
    )

    assert result["stored"] is False
    assert "final_df" in result
    assert len(result["final_df"]) == 1
    #assert not result["final_df"].empty


    # DB should NOT exist or be empty
    assert not db_path.exists() or sqlite3.connect(db_path).execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall() == []


def test_single_patient_with_consent_is_stored(tmp_path):
    """
    Single patient WITH consent:
    - preprocessing runs
    - data IS stored in SQLite
    """

    pheno = tmp_path / "pheno.csv"
    geno = tmp_path / "geno.csv"
    db_path = tmp_path / "test.db"

    create_dummy_pheno_csv(pheno)
    create_dummy_geno_csv(geno)

    pipeline = PreprocessingPipeline(db_path=str(db_path))

    result = pipeline.run(
        pheno_path=str(pheno),
        geno_path=str(geno),
        is_single_patient=True,
        consent_to_store=True,
        registry_date="2025-12-25",
    )

    assert result["stored"] is True
    assert "final_df" in result

    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()

    table_names = {t[0] for t in tables}
    assert "analysis_master" in table_names
    conn.close()


def test_cohort_without_consent_is_blocked(tmp_path):
    """
    Cohort data WITHOUT consent:
    - preprocessing must NOT run
    """

    pheno = tmp_path / "pheno.csv"
    geno = tmp_path / "geno.csv"
    db_path = tmp_path / "test.db"

    create_dummy_pheno_csv(pheno)
    create_dummy_geno_csv(geno)

    pipeline = PreprocessingPipeline(db_path=str(db_path))

    with pytest.raises(AssertionError):
        pipeline.run(
            pheno_path=str(pheno),
            geno_path=str(geno),
            is_single_patient=False,
            consent_to_store=False,
            registry_date="2025-12-25",
        )


def test_cohort_with_consent_is_stored(tmp_path):
    """
    Cohort data WITH consent:
    - preprocessing runs
    - data IS stored
    """

    pheno = tmp_path / "pheno.csv"
    geno = tmp_path / "geno.csv"
    db_path = tmp_path / "test.db"

    create_dummy_pheno_csv(pheno)
    create_dummy_geno_csv(geno)

    pipeline = PreprocessingPipeline(db_path=str(db_path))

    result = pipeline.run(
        pheno_path=str(pheno),
        geno_path=str(geno),
        is_single_patient=False,
        consent_to_store=True,
        registry_date="2025-12-25",
    )

    assert result["stored"] is True

    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM analysis_master;"
    ).fetchone()[0]

    assert count == 1
    conn.close()
