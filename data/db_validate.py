# data/db_validate.py
# Script to validate the contents of the analysis_master table
# run: python data/db_validate.py
# python -m data.db_validate

import sqlite3
import pandas as pd

conn = sqlite3.connect("data/database.db")

df = pd.read_sql(
    "SELECT * FROM analysis_master",
    conn
)

conn.close()

df.to_csv(
    "data/processed/analysis_master.csv",
    index=False
)

print("Saved analysis_master.csv")
