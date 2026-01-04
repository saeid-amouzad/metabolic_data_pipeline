"""
Integration + SQLite storage
Combines processed phenotypic and genotypic data
Writes processed data.

It does three things:
    1. Adds registry_date to processed phenotypic data
    2. Stores each processed data type in SQLite:
        phenotypes_clean
        genotypes_clean
    3. Creates a merged dataset:
        analysis_master

| Scenario                    | phenotypes_clean | genotypes_clean | analysis_master |
| --------------------------- | ---------------- | --------------- | --------------- |
| Cohort (censent to rewrite) | REPLACE          | REPLACE         | REPLACE         |
| Single patient (consent)    | APPEND           | APPEND          | REBUILD         |

"""

# src/preprocessing/integrator.py

import sqlite3
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataIntegrator:
    """
    Integrates processed phenotypic and genotypic data
    into SQLite with correct append / replace semantics.
    """

    def __init__(self, db_path):
        self.db_path = db_path

    def integrate(
        self,
        pheno_df: pd.DataFrame,
        geno_df: pd.DataFrame,
        *,
        registry_date: str,
        is_single_patient: bool,
        table_name: str = "analysis_master"
    ) -> pd.DataFrame:

        conn = sqlite3.connect(self.db_path)

        # --------------------------------------------------
        # Add registry_date
        # --------------------------------------------------
        pheno_df = pheno_df.copy()
        pheno_df["registry_date"] = registry_date

        # --------------------------------------------------
        # Data integrity checks (ADD HERE)
        # --------------------------------------------------
        assert "patient_id" in pheno_df.columns, "patient_id missing in pheno_df"
        assert "patient_id" in geno_df.columns, "patient_id missing in geno_df"

        assert pheno_df["patient_id"].is_unique, (
            "Duplicate patient_id found in pheno_df"
        )
        assert geno_df["patient_id"].is_unique, (
            "Duplicate patient_id found in geno_df"
        )

        # --------------------------------------------------
        # Decide storage mode
        # --------------------------------------------------
        if is_single_patient:
            pheno_mode = "append"
            geno_mode = "append"
        else:
            pheno_mode = "replace"
            geno_mode = "replace"

        # --------------------------------------------------
        # Store individual tables
        # --------------------------------------------------
        pheno_df.to_sql(
            "phenotypes_clean",
            conn,
            if_exists=pheno_mode,
            index=False
        )

        geno_df.to_sql(
            "genotypes_clean",
            conn,
            if_exists=geno_mode,
            index=False
        )

        # --------------------------------------------------
        # Append (for a single patient) or Rebuild merged table (for cohort)
        # --------------------------------------------------
        if is_single_patient:
            # Join ONLY the new patient and append
            single_row_df = pd.merge(
                pheno_df,
                geno_df,
                on="patient_id",
                how="inner"
            )

            single_row_df.to_sql(
                table_name,
                conn,
                if_exists="append",
                index=False
            )

        else:
                # Full cohort rebuild
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            conn.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT
                p.*,
                g.rs61652270,
                g.rs60004782,
                g.rs9554188,
                g.rs9581927,
                g.rs9581929,
                g.rs7985481,
                g.rs9581931,
                g.rs4424773,
                g.rs9554193,
                g.rs7995917,
                g.rs7993114,
                g.rs11618581,
                g.rs9554197,
                g.rs11618036,
                g.rs11618832,
                g.rs11616678,
                g.rs11618052,
                g.rs9579127,
                g.rs7999100,
                g.rs8000004,
                g.rs9579128,
                g.rs2297316,
                g.rs9581943
            FROM phenotypes_clean p
            INNER JOIN genotypes_clean g
                ON p.patient_id = g.patient_id
            """)

        conn.commit()

        final_df = pd.read_sql(
            f"SELECT * FROM {table_name}",
            conn
        )

        conn.close()

        logger.info(
            "Integrated data stored | rows=%d | single_patient=%s",
            len(final_df), is_single_patient
        )

        return final_df
