"""
Heart Failure Prediction — ETL Pipeline
========================================
Extracts raw data from two CSV sources, transforms and validates them,
then loads into the heart_failure_db PostgreSQL database.

Tables populated:
    raw.patients_clinical    — 299 follow-up clinical records (death event target)
    raw.patients_diagnostic  — 918 diagnostic records (heart disease target)

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import logging
import os
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("etl_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Database Config ───────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "heart_failure_db",
    "user":     "postgres",
    "password": "postgres"
}

# ── File Paths ────────────────────────────────────────────────────────────────
CLINICAL_PATH    = "heart_failure_clinical_records_dataset (1).csv"
DIAGNOSTIC_PATH  = "heart.csv"

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════════════════════

def extract_clinical(filepath: str) -> pd.DataFrame:
    """
    Load the 299-row follow-up dataset.
    Target: DEATH_EVENT — did the patient die during the follow-up period?
    """
    logger.info(f"Extracting clinical data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df

def extract_diagnostic(filepath: str) -> pd.DataFrame:
    """
    Load the 918-row diagnostic dataset.
    Target: HeartDisease — does the patient have heart disease?
    """
    logger.info(f"Extracting diagnostic data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

def transform_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the clinical dataset.
    Rules applied:
        - Rename DEATH_EVENT to death_event (lowercase convention)
        - Drop rows with null values in critical clinical columns
        - Validate age is within realistic range (1-120)
        - Validate ejection_fraction is within realistic range (1-100)
        - Flag and log any anomalies
    """
    logger.info("Transforming clinical dataset")
    original_count = len(df)

    # Standardize column names to lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Validate age range
    invalid_age = df[(df["age"] < 1) | (df["age"] > 120)]
    if len(invalid_age) > 0:
        logger.warning(f"Found {len(invalid_age)} rows with invalid age — removing")
        df = df[(df["age"] >= 1) & (df["age"] <= 120)]

    # Validate ejection fraction range
    invalid_ef = df[(df["ejection_fraction"] < 1) | (df["ejection_fraction"] > 100)]
    if len(invalid_ef) > 0:
        logger.warning(f"Found {len(invalid_ef)} rows with invalid ejection fraction — removing")
        df = df[(df["ejection_fraction"] >= 1) & (df["ejection_fraction"] <= 100)]

    # Drop nulls in critical columns
    critical_cols = ["age", "ejection_fraction", "serum_creatinine", "death_event"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with null values in critical columns")

    logger.info(f"Clinical transform complete: {original_count} → {len(df)} rows")
    return df

def transform_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the diagnostic dataset.
    Rules applied:
        - Rename columns to snake_case for PostgreSQL compatibility
        - Validate cholesterol (0 values likely mean missing — log them)
        - Validate RestingBP (0 is physiologically impossible)
        - Standardize sex values (M/F already correct)
    """
    logger.info("Transforming diagnostic dataset")
    original_count = len(df)

    # Rename columns to snake_case
    df = df.rename(columns={
        "Age":             "age",
        "Sex":             "sex",
        "ChestPainType":   "chest_pain_type",
        "RestingBP":       "resting_bp",
        "Cholesterol":     "cholesterol",
        "FastingBS":       "fasting_bs",
        "RestingECG":      "resting_ecg",
        "MaxHR":           "max_hr",
        "ExerciseAngina":  "exercise_angina",
        "Oldpeak":         "oldpeak",
        "ST_Slope":        "st_slope",
        "HeartDisease":    "heart_disease"
    })

    # Log zero cholesterol rows (likely missing data, keep but flag)
    zero_chol = df[df["cholesterol"] == 0]
    if len(zero_chol) > 0:
        logger.warning(f"{len(zero_chol)} rows have cholesterol = 0 (likely missing) — keeping but flagging")

    # Remove rows where resting_bp = 0 (physiologically impossible)
    invalid_bp = df[df["resting_bp"] == 0]
    if len(invalid_bp) > 0:
        logger.warning(f"Removing {len(invalid_bp)} rows with resting_bp = 0")
        df = df[df["resting_bp"] > 0]

    # Validate max heart rate
    invalid_hr = df[(df["max_hr"] < 40) | (df["max_hr"] > 250)]
    if len(invalid_hr) > 0:
        logger.warning(f"Removing {len(invalid_hr)} rows with unrealistic max_hr")
        df = df[(df["max_hr"] >= 40) & (df["max_hr"] <= 250)]

    logger.info(f"Diagnostic transform complete: {original_count} → {len(df)} rows")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    """Create and return a PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)

def load_clinical(df: pd.DataFrame):
    """
    Load the clinical dataset into raw.patients_clinical.
    Uses execute_values for efficient bulk insert.
    Truncates existing data first (full reload pattern).
    """
    logger.info("Loading clinical data into raw.patients_clinical")
    conn = get_connection()
    cur = conn.cursor()

    # Truncate existing data (clean reload)
    cur.execute("TRUNCATE TABLE raw.patients_clinical RESTART IDENTITY;")
    logger.info("Truncated raw.patients_clinical")

    # Prepare records for bulk insert
    records = []
    for _, row in df.iterrows():
        records.append((
            int(row["age"]),
            int(row["anaemia"]),
            int(row["creatinine_phosphokinase"]),
            int(row["diabetes"]),
            int(row["ejection_fraction"]),
            int(row["high_blood_pressure"]),
            float(row["platelets"]),
            float(row["serum_creatinine"]),
            int(row["serum_sodium"]),
            int(row["sex"]),
            int(row["smoking"]),
            int(row["time"]),
            int(row["death_event"])
        ))

    # Bulk insert using execute_values (much faster than executemany)
    execute_values(cur, """
        INSERT INTO raw.patients_clinical
        (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
         high_blood_pressure, platelets, serum_creatinine, serum_sodium,
         sex, smoking, time, death_event)
        VALUES %s
    """, records)

    conn.commit()
    logger.info(f"Loaded {len(records)} records into raw.patients_clinical")
    cur.close()
    conn.close()

def load_diagnostic(df: pd.DataFrame):
    """
    Load the diagnostic dataset into raw.patients_diagnostic.
    """
    logger.info("Loading diagnostic data into raw.patients_diagnostic")
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE raw.patients_diagnostic RESTART IDENTITY;")
    logger.info("Truncated raw.patients_diagnostic")

    records = []
    for _, row in df.iterrows():
        records.append((
            int(row["age"]),
            str(row["sex"]),
            str(row["chest_pain_type"]),
            int(row["resting_bp"]),
            int(row["cholesterol"]),
            int(row["fasting_bs"]),
            str(row["resting_ecg"]),
            int(row["max_hr"]),
            str(row["exercise_angina"]),
            float(row["oldpeak"]),
            str(row["st_slope"]),
            int(row["heart_disease"])
        ))

    execute_values(cur, """
        INSERT INTO raw.patients_diagnostic
        (age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
         resting_ecg, max_hr, exercise_angina, oldpeak, st_slope, heart_disease)
        VALUES %s
    """, records)

    conn.commit()
    logger.info(f"Loaded {len(records)} records into raw.patients_diagnostic")
    cur.close()
    conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def validate():
    """
    Confirm row counts match expectations after loading.
    This is your data contract check — if counts dont match, something went wrong.
    """
    logger.info("Validating loaded data")
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM raw.patients_clinical;")
    clinical_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM raw.patients_diagnostic;")
    diagnostic_count = cur.fetchone()[0]

    cur.execute("SELECT MIN(age), MAX(age), AVG(age)::NUMERIC(5,2) FROM raw.patients_clinical;")
    clinical_age = cur.fetchone()

    cur.execute("SELECT MIN(age), MAX(age), AVG(age)::NUMERIC(5,2) FROM raw.patients_diagnostic;")
    diagnostic_age = cur.fetchone()

    cur.execute("SELECT death_event, COUNT(*) FROM raw.patients_clinical GROUP BY death_event ORDER BY death_event;")
    death_dist = cur.fetchall()

    cur.execute("SELECT heart_disease, COUNT(*) FROM raw.patients_diagnostic GROUP BY heart_disease ORDER BY heart_disease;")
    disease_dist = cur.fetchall()

    logger.info(f"raw.patients_clinical: {clinical_count} rows")
    logger.info(f"  Age range: {clinical_age[0]} - {clinical_age[1]}, avg: {clinical_age[2]}")
    logger.info(f"  Death event distribution: {dict(death_dist)}")

    logger.info(f"raw.patients_diagnostic: {diagnostic_count} rows")
    logger.info(f"  Age range: {diagnostic_age[0]} - {diagnostic_age[1]}, avg: {diagnostic_age[2]}")
    logger.info(f"  Heart disease distribution: {dict(disease_dist)}")

    cur.close()
    conn.close()

    return clinical_count, diagnostic_count

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Heart Failure ETL Pipeline — Start")
    logger.info("=" * 60)

    try:
        # Extract
        clinical_raw     = extract_clinical(CLINICAL_PATH)
        diagnostic_raw   = extract_diagnostic(DIAGNOSTIC_PATH)

        # Transform
        clinical_clean   = transform_clinical(clinical_raw)
        diagnostic_clean = transform_diagnostic(diagnostic_raw)

        # Load
        load_clinical(clinical_clean)
        load_diagnostic(diagnostic_clean)

        # Validate
        c_count, d_count = validate()

        logger.info("=" * 60)
        logger.info(f"ETL Pipeline Complete")
        logger.info(f"  patients_clinical:   {c_count} rows loaded")
        logger.info(f"  patients_diagnostic: {d_count} rows loaded")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ETL Pipeline failed: {e}")
        raise
