# tests/test_validators.py
"""  
    Input validation utilities (date format, file existence, data type)
    for ingestion entry point
    Test validators in src/utils/validators.py
    pytest tests/
"""

import pytest
import tempfile
import os

from src.utils.validators import (
    validate_registry_date,
    validate_file_exists,
    validate_data_type
)

def test_validate_registry_date_valid():
    validate_registry_date("2025-12-25")

def test_validate_registry_date_invalid():
    with pytest.raises(ValueError):
        validate_registry_date("25-12-2025")

def test_validate_file_exists():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name

    validate_file_exists(path)
    os.remove(path)

def test_validate_data_type_invalid():
    with pytest.raises(ValueError):
        validate_data_type("audio")
