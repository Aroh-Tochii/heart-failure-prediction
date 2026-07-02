"""
Heart Failure Prediction — Drift Monitoring
=============================================
Monitors three things:
    1. Clinical data drift  — has patient population changed vs baseline?
    2. Diagnostic data drift — has diagnostic profile changed vs baseline?
    3. Prediction drift      — are risk scores shifting over time?

Generates three separate HTML reports saved to reports/drift/

Run this weekly or monthly in production to catch:
    - Changes in patient demographics
    - Shifts in clinical measurement distributions
    - Model score drift indicating retraining is needed

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import psycopg2
import os
import logging
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("drift_monitor.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "heart_failure_db",
    "user":     "postgres",
    "password": "postgres"
}

OUTPUT_DIR        = "reports/drift"
REFERENCE_DIR     = "reports/reference"
TIMESTAMP         = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)

# ── Load data from PostgreSQL ─────────────────────────────────────────────────
def load_clinical() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT age, ejection_fraction, serum_creatinine, serum_sodium,
               anaemia, high_blood_pressure, time, death_event
        FROM raw.patients_clinical
    """, conn)
    conn.close()
    logger.info(f"Clinical data loaded: {len(df)} rows")
    return df

def load_diagnostic() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT age, resting_bp, cholesterol, max_hr,
               oldpeak, fasting_bs, heart_disease
        FROM raw.patients_diagnostic
    """, conn)
    conn.close()
    logger.info(f"Diagnostic data loaded: {len(df)} rows")
    return df

def load_predictions() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql("""
            SELECT age, prediction, probability, risk_level,
                   EXTRACT(HOUR FROM predicted_at) AS prediction_hour
            FROM predictions.risk_predictions
        """, conn)
        conn.close()
        logger.info(f"Predictions loaded: {len(df)} rows")
        return df
    except Exception as e:
        conn.close()
        logger.warning(f"Could not load predictions: {e}")
        return pd.DataFrame()

# ── Reference snapshot management ─────────────────────────────────────────────
def load_or_create_reference(df: pd.DataFrame, name: str) -> pd.DataFrame:
    path = f"{REFERENCE_DIR}/{name}_reference.csv"
    if os.path.exists(path):
        logger.info(f"Loading existing reference for {name} from {path}")
        return pd.read_csv(path)
    else:
        logger.info(f"No reference found for {name} — creating baseline snapshot")
        df.to_csv(path, index=False)
        logger.info(f"Reference saved: {path}")
        return df

# ── Run drift report ───────────────────────────────────────────────────────────
def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                     report_name: str, title: str) -> str:

    logger.info(f"Running drift analysis: {title}")

    # Align columns
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    reference_df = reference_df[common_cols]
    current_df   = current_df[common_cols]

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ])

    result = report.run(reference_data=reference_df, current_data=current_df)

    report_path = f"{OUTPUT_DIR}/{report_name}_{TIMESTAMP}.html"
    result.save_html(report_path)
    logger.info(f"Report saved: {report_path}")

    return report_path

# ── Summary logger ─────────────────────────────────────────────────────────────
def log_summary(clinical_df, diagnostic_df, predictions_df):
    """Log key statistics to give a quick health check without opening reports."""
    logger.info("\n" + "="*60)
    logger.info("DRIFT MONITORING SUMMARY")
    logger.info("="*60)

    # Clinical summary
    logger.info("\nClinical Dataset:")
    logger.info(f"  Patients:           {len(clinical_df)}")
    logger.info(f"  Avg age:            {clinical_df['age'].mean():.1f}")
    logger.info(f"  Avg EF:             {clinical_df['ejection_fraction'].mean():.1f}%")
    logger.info(f"  Avg creatinine:     {clinical_df['serum_creatinine'].mean():.2f}")
    logger.info(f"  Death rate:         {clinical_df['death_event'].mean()*100:.1f}%")

    # Diagnostic summary
    logger.info("\nDiagnostic Dataset:")
    logger.info(f"  Patients:           {len(diagnostic_df)}")
    logger.info(f"  Avg age:            {diagnostic_df['age'].mean():.1f}")
    logger.info(f"  Avg max HR:         {diagnostic_df['max_hr'].mean():.1f}")
    logger.info(f"  Disease rate:       {diagnostic_df['heart_disease'].mean()*100:.1f}%")

    # Prediction summary
    if len(predictions_df) > 0:
        logger.info("\nPrediction Statistics:")
        logger.info(f"  Total predictions:  {len(predictions_df)}")
        logger.info(f"  Avg probability:    {predictions_df['probability'].mean():.3f}")
        risk_dist = predictions_df['risk_level'].value_counts()
        for level, count in risk_dist.items():
            logger.info(f"  {level} risk:         {count} ({count/len(predictions_df)*100:.1f}%)")
    else:
        logger.info("\nPredictions: No predictions logged yet")

    logger.info("="*60)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Heart Failure Drift Monitoring — Start")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info("="*60)

    try:
        # Load current data
        clinical_df    = load_clinical()
        diagnostic_df  = load_diagnostic()
        predictions_df = load_predictions()

        # Load or create reference baselines
        clinical_ref    = load_or_create_reference(clinical_df,   "clinical")
        diagnostic_ref  = load_or_create_reference(diagnostic_df, "diagnostic")

        # Run drift reports
        report1 = run_drift_report(
            clinical_ref, clinical_df,
            "clinical_drift",
            "Clinical Patient Data Drift"
        )

        report2 = run_drift_report(
            diagnostic_ref, diagnostic_df,
            "diagnostic_drift",
            "Diagnostic Patient Data Drift"
        )

        # Log summary
        log_summary(clinical_df, diagnostic_df, predictions_df)

        logger.info("\n=== Drift Monitoring Complete ===")
        logger.info(f"Reports saved to: {OUTPUT_DIR}/")
        logger.info(f"  Clinical drift:    {report1}")
        logger.info(f"  Diagnostic drift:  {report2}")
        logger.info("\nOpen these HTML files in your browser to view full reports.")

    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")
        raise
