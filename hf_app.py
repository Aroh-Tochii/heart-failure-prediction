"""
Heart Failure Prediction — FastAPI Service
==========================================
Two prediction endpoints serving two clinical models:

    POST /predict/death-risk
        Input:  Clinical patient data (7 features + engineered)
        Output: Death risk probability, risk level, clinical flags
        Model:  Logistic Regression (highest recall — minimizes missed deaths)

    POST /predict/heart-disease
        Input:  Diagnostic patient data (chest pain, ECG, max HR, etc.)
        Output: Heart disease probability and risk level
        Model:  XGBoost trained on diagnostic dataset (future enhancement)

    GET  /health     — Service health check
    GET  /           — Service info

Author: Alfred (Aroh-Tochii)
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
import joblib
import psycopg2
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from hf_mlflow_utils import normalize_mlflow_paths

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{APP_ROOT / 'mlflow.db'}",
)
EXPERIMENT_NAME     = "heart_failure_prediction"
DECISION_THRESHOLD  = 0.3

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "heart_failure_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Failure Clinical Decision Support API",
    description="""
    ## Clinical Decision Support System

    This API provides two prediction services for cardiac care:

    ### 1. Death Risk Prediction (`/predict/death-risk`)
    For patients already diagnosed with heart failure.
    Predicts mortality risk during the follow-up period.
    **Optimized for Recall** — minimizes missed high-risk patients.
    Decision threshold: 0.3 (clinically conservative)

    ### 2. Heart Disease Detection (`/predict/heart-disease`)
    For patients presenting with cardiac symptoms.
    Predicts likelihood of underlying heart disease.

    ### Risk Levels
    - **Low Risk** (< 30%) → Standard monitoring
    - **Medium Risk** (30-60%) → Enhanced follow-up within 7 days
    - **High Risk** (> 60%) → Immediate clinical escalation
    """,
    version="1.0.0"
)

# ── Global model store ────────────────────────────────────────────────────────
models = {}

# ── Load models at startup ────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global models
    normalize_mlflow_paths(APP_ROOT)
    logger.info("Loading models from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run hf_train_model.py first.")

    # Load best model by recall (Logistic Regression)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'Logistic Regression'",
        order_by=["metrics.recall DESC"],
        max_results=1
    )

    if runs:
        run = runs[0]
        model_uri = f"runs:/{run.info.run_id}/model"
        models["death_risk"] = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Death risk model loaded — Recall: {run.data.metrics.get('recall', 'N/A')}")
    else:
        # Fallback: load most recent successful run
        all_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.recall DESC"],
            max_results=1
        )
        if all_runs:
            run = all_runs[0]
            model_uri = f"runs:/{run.info.run_id}/model"
            models["death_risk"] = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Death risk model loaded (fallback) from run: {run.info.run_id}")

    logger.info(f"Models loaded: {list(models.keys())}")

# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ClinicalPatientData(BaseModel):
    """Input schema for death risk prediction."""
    serum_creatinine: float = Field(..., ge=0.1, le=15.0,
        description="Serum creatinine level (mg/dL). Normal: 0.6-1.2")
    ejection_fraction: int = Field(..., ge=10, le=80,
        description="Percentage of blood pumped per heartbeat. Normal: 55-70%")
    age: int = Field(..., ge=18, le=120,
        description="Patient age in years")
    serum_sodium: int = Field(..., ge=110, le=150,
        description="Serum sodium level (mEq/L). Normal: 135-145")
    time: int = Field(..., ge=1, le=300,
        description="Follow-up period in days")
    anaemia: int = Field(..., ge=0, le=1,
        description="Anaemia present: 1=Yes, 0=No")
    high_blood_pressure: int = Field(..., ge=0, le=1,
        description="High blood pressure: 1=Yes, 0=No")

    class Config:
        json_schema_extra = {
            "example": {
                "serum_creatinine": 1.9,
                "ejection_fraction": 20,
                "age": 65,
                "serum_sodium": 130,
                "time": 30,
                "anaemia": 1,
                "high_blood_pressure": 1
            }
        }

class RiskPredictionResponse(BaseModel):
    """Output schema for risk predictions."""
    prediction: int
    probability: float
    risk_level: str
    clinical_flags: list
    recommendation: str
    model_used: str
    decision_threshold: float
    predicted_at: str

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(data: dict) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    df = pd.DataFrame([data])

    df["kidney_heart_risk"] = (
        (df["serum_creatinine"] > 1.5).astype(int) *
        (df["ejection_fraction"] < 30).astype(int)
    )
    df["age_creatinine_interaction"] = df["age"] * df["serum_creatinine"]
    df["hyponatremia"] = (df["serum_sodium"] < 135).astype(int)
    df["comorbidity_score"] = df["anaemia"] + df["high_blood_pressure"]

    return df

def get_clinical_flags(data: dict) -> list:
    """Generate clinical warning flags based on input values."""
    flags = []

    if data["ejection_fraction"] < 30:
        flags.append("CRITICAL: Severely reduced ejection fraction (<30%)")
    elif data["ejection_fraction"] < 40:
        flags.append("WARNING: Reduced ejection fraction (<40%)")

    if data["serum_creatinine"] > 2.0:
        flags.append("CRITICAL: Severe kidney dysfunction (creatinine >2.0)")
    elif data["serum_creatinine"] > 1.5:
        flags.append("WARNING: Elevated creatinine — kidney stress detected")

    if data["serum_sodium"] < 130:
        flags.append("CRITICAL: Severe hyponatremia (sodium <130)")
    elif data["serum_sodium"] < 135:
        flags.append("WARNING: Low sodium — fluid imbalance detected")

    if data["age"] >= 75:
        flags.append("NOTE: Age 75+ — elevated baseline mortality risk (61%)")

    if data["anaemia"] == 1 and data["high_blood_pressure"] == 1:
        flags.append("NOTE: Multiple comorbidities — anaemia + hypertension")

    return flags

def get_recommendation(risk_level: str) -> str:
    """Return clinical action recommendation based on risk level."""
    recommendations = {
        "Low":    "Standard monitoring protocol. Schedule routine follow-up in 30 days.",
        "Medium": "Enhanced monitoring required. Follow-up within 7 days. Review medication.",
        "High":   "IMMEDIATE ACTION REQUIRED. Escalate to senior clinician. Consider admission."
    }
    return recommendations.get(risk_level, "Consult clinical team.")

def save_prediction_to_db(patient_data: dict, prediction: int,
                           probability: float, risk_level: str, source: str):
    """Log prediction to the predictions schema for audit trail."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions.risk_predictions
            (age, prediction, probability, risk_level, source)
            VALUES (%s, %s, %s, %s, %s)
        """, (patient_data.get("age"), prediction,
              probability, risk_level, source))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not save prediction to DB: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service":       "Heart Failure Clinical Decision Support API",
        "version":       "1.0.0",
        "models_loaded": list(models.keys()),
        "endpoints": {
            "/predict/death-risk":    "Mortality risk for heart failure patients",
            "/predict/heart-disease": "Heart disease detection from diagnostic data (coming soon)",
            "/health":                "Service health check",
            "/docs":                  "Interactive API documentation"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status":        "healthy",
        "models_loaded": list(models.keys()),
        "death_risk_model": "death_risk" in models
    }

@app.post("/predict/death-risk", response_model=RiskPredictionResponse)
def predict_death_risk(patient: ClinicalPatientData):
    """
    Predict mortality risk for a heart failure patient.

    Optimized for high recall — will flag more patients as high risk
    to minimize missed deaths. Decision threshold: 0.3 (not 0.5).
    """
    if "death_risk" not in models:
        raise HTTPException(status_code=503, detail="Death risk model not loaded")

    try:
        input_dict = patient.dict()
        df = engineer_features(input_dict)

        model = models["death_risk"]
        proba = model.predict_proba(df)[0][1]
        pred  = int(proba >= DECISION_THRESHOLD)

        if proba < 0.30:
            risk_level = "Low"
        elif proba < 0.60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        flags          = get_clinical_flags(input_dict)
        recommendation = get_recommendation(risk_level)

        # Log to database for audit trail
        save_prediction_to_db(input_dict, pred, float(proba), risk_level, "clinical")

        logger.info(
            f"Death risk prediction: prob={proba:.4f}, "
            f"risk={risk_level}, flags={len(flags)}"
        )

        return RiskPredictionResponse(
            prediction         = pred,
            probability        = round(float(proba), 4),
            risk_level         = risk_level,
            clinical_flags     = flags,
            recommendation     = recommendation,
            model_used         = "Logistic Regression (recall-optimized)",
            decision_threshold = DECISION_THRESHOLD,
            predicted_at       = datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predict/heart-disease")
def heart_disease_placeholder():
    return {
        "status":  "coming_soon",
        "message": "Heart disease detection model will be added in v2.0",
        "note":    "Train on diagnostic dataset (raw.patients_diagnostic) with XGBoost"
    }

@app.get("/stats")
def prediction_stats():
    """Return prediction statistics from the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                risk_level,
                COUNT(*) AS total,
                ROUND(AVG(probability)::NUMERIC, 3) AS avg_probability
            FROM predictions.risk_predictions
            GROUP BY risk_level
            ORDER BY risk_level
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"prediction_stats": [
            {"risk_level": r[0], "total": r[1], "avg_probability": float(r[2])}
            for r in rows
        ]}
    except Exception as e:
        return {"error": str(e)}

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hf_app:app", host="0.0.0.0", port=8002, reload=True)
