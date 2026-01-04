# src/utils/db.py

"""
Low-level helpers:
    - get_connection
    - init_db
    Used to:
        *initialize metadata tables
        *store upload metadata (GDPR-safe)

High-level DatabaseClient:
    Used for:
        *DES-style guards
        *checking if a step has completed
        *ensuring pipeline order

"""
import sqlite3
from pathlib import Path


# =====================================================
# Low-level helpers
# =====================================================

def get_connection(db_path):
    """
    Create and return a SQLite database connection.
    """
    return sqlite3.connect(db_path)


def init_db(db_path):
    """
    Initialize database tables if they do not exist.

    This table stores METADATA ONLY (not raw data content),
    which is GDPR-safe and audit-friendly.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Optional: present for single-patient ingestion
            patient_id TEXT,

            -- Original uploaded filename
            filename TEXT NOT NULL,

            -- phenotypic | genotypic | image
            data_type TEXT NOT NULL,

            -- UI / API / batch / external registry
            source TEXT,

            -- Registry / visit / acquisition date (YYYY-MM-DD)
            registry_date DATE,

            -- Whether user consented to storing raw data
            consent_raw_storage BOOLEAN,

            -- Distinguish cohort vs single-patient upload
            is_single_patient BOOLEAN,

            -- Optional: row count for tabular data
            rows INTEGER,

            -- Automatic audit timestamp
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# =====================================================
# High-level database client (DES support)
# =====================================================

class DatabaseClient:
    """
    Lightweight SQLite helper for pipeline validation
    and DES-style step checks.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # -------------------------------------------------
    # DES-style guards
    # -------------------------------------------------

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the SQLite database.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )

        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def count_rows(self, table_name: str) -> int:
        """
        Count number of rows in a table.
        """
        if not self.table_exists(table_name):
            return 0

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        conn.close()
        return count
