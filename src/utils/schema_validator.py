# src/utils/schema_validator.py

import pandas as pd

# Expected feature sets derived from your ACTUAL datasets
EXPECTED_FEATURES = {
    "phenotypic": {
        "patient_id",
        "year",
        "gender",
        "age",
        "location",
        "race:africanamerican",
        "race:asian",
        "race:caucasian",
        "race:hispanic",
        "race:other",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "hba1c_level",
        "blood_glucose_level",
        "diabetes"   # allowed but handled later as target
    },

    "genotypic": {
        "patient_id",
        "rs61652270",
        "rs60004782",
        "rs9554188",
        "rs9581927",
        "rs9581929",
        "rs7985481",
        "rs9581931",
        "rs4424773",
        "rs9554193",
        "rs7995917",
        "rs7993114",
        "rs11618581",
        "rs9554197",
        "rs11618036",
        "rs11618832",
        "rs11616678",
        "rs11618052",
        "rs9579127",
        "rs7999100",
        "rs8000004",
        "rs9579128",
        "rs2297316",
        "rs9581943"
    }
}


def assert_feature_compatibility(file_path, data_type):
    """
    Ensures that single-patient data has the SAME schema
    as the cohort raw data.

    To prevent corruption of the database.
    """
    df = pd.read_csv(file_path)

    expected = EXPECTED_FEATURES[data_type]
    missing = expected - set(df.columns)

    assert not missing, (
        f"Feature mismatch for {data_type}. "
        f"Missing columns: {missing}"
    )
