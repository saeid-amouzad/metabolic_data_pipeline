# src/utils/schema_utils.py

def align_to_schema(df, expected_columns):
    """
    Ensure DataFrame has exactly the expected columns.
        Missing columns are added with 0.
        Extra columns raise an error.
    """
    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)

    if extra:
        raise ValueError(f"Unexpected columns: {extra}")

    for col in missing:
        df[col] = 0

    return df[expected_columns]
