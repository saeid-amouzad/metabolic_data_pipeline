# src/utils/validators.py

"""
What it validates:
    - File existence
    - File type vs extension
    - Tabular data not empty
    - Image files not empty
    - Registry date format

"""


import os
import pandas as pd

# -------------------------------------------------
# Supported formats per data type
# -------------------------------------------------
SUPPORTED_FORMATS = {
    "phenotypic": {".csv", ".xlsx", ".json"},
    "genotypic": {".csv", ".xlsx", ".json", ".vcf", ".bed", ".bim", ".fam"},
    "image": {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dcm"}
}

ALLOWED_DATA_TYPES = set(SUPPORTED_FORMATS.keys())


# -------------------------------------------------
# Core validators
# -------------------------------------------------
def validate_file_exists(file_path):
    """
    Ensure file exists on disk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Uploaded file does not exist")


def validate_data_type(data_type):
    """
    Ensure data type is supported by the system.
    """
    if data_type not in ALLOWED_DATA_TYPES:
        raise ValueError(
            f"Invalid data type '{data_type}'. "
            f"Allowed types: {list(ALLOWED_DATA_TYPES)}"
        )


def validate_file_format(file_path, data_type):
    """
    Validate file extension against allowed formats
    for the given data type.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_FORMATS[data_type]:
        raise ValueError(
            f"Unsupported file format '{ext}' for data type '{data_type}'. "
            f"Supported formats: {SUPPORTED_FORMATS[data_type]}"
        )


# -------------------------------------------------
# Content-level validation
# -------------------------------------------------
def validate_tabular_not_empty(file_path):
    """
    Validate that a tabular file (CSV/XLSX/JSON) is not empty.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".xlsx":
        df = pd.read_excel(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    else:
        # Non-tabular formats should not reach here
        return

    if df.empty:
        raise ValueError("Uploaded tabular file is empty")


def validate_image_file(file_path):
    """
    Basic validation for image/signal files.
    Ensures file size > 0.
    """
    if os.path.getsize(file_path) == 0:
        raise ValueError("Uploaded image file is empty")

from datetime import datetime


def validate_registry_date(date_str):
    """
    Validate registry / acquisition date.

    Expected format: YYYY-MM-DD
    Allows None (optional field).
    """
    if date_str is None:
        return

    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            "registry_date must be in YYYY-MM-DD format "
            "(e.g., 2025-12-25)"
        )
