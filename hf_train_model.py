"""
Heart Failure Prediction — Feature Engineering & Model Training
===============================================================
Trains three models on the clinical dataset:
    1. XGBoost (primary — best tabular performance)
    2. Logistic Regression (clinical explainability)
    3. Random Forest (baseline comparison)

Optimization target: RECALL for death class (death_event = 1)
Reason: False negatives (missed deaths) are clinically far more
        costly than false positives (unnecessary monitoring).

Decision threshold: 0.3 (lower than default 0.5 to maximize recall)

All experiments tracked in MLflow. Best model registered for API use.

Author: Alfred (Aroh-Tochii)
"""

import os
import pandas as pd
import numpy as np
import psycopg2
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from hf_mlflow_utils import normalize_mlflow_paths

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import xgboost as xgb

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("train_model.log"), logging.StreamHandler()]
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

PROJECT_ROOT = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI  = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME      = "heart_failure_prediction"
DECISION_THRESHOLD   = 0.3    # Lower threshold to maximize recall (catch more deaths)
RANDOM_STATE         = 42

# ── Selected features (based on EDA and clinical reasoning) ──────────────────
SELECTED_FEATURES = [
    "serum_creatinine",    # Kidney function — strongest clinical separator
    "ejection_fraction",   # Heart pump efficiency — core cardiac metric
    "age",                 # Age 75+ showed 61% death rate
    "serum_sodium",        # Fluid balance indicator
    "time",                # Follow-up duration — statistical powerhouse
    "anaemia",             # Meaningful prevalence difference in non-survivors
    "high_blood_pressure"  # Elevated in non-survivor group
]

TARGET = "death_event"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA FROM POSTGRESQL
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Pull clinical data from PostgreSQL."""
    logger.info("Loading clinical data from PostgreSQL")
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql(f"""
        SELECT {', '.join(SELECTED_FEATURES)}, {TARGET}
        FROM raw.patients_clinical
    """, conn)
    conn.close()
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create clinically meaningful composite features.
    These capture interaction effects that individual features miss.
    """
    logger.info("Engineering features")
    df = df.copy()

    # Kidney-heart interaction: high creatinine + low ejection fraction
    # Both failing together is more dangerous than either alone
    df["kidney_heart_risk"] = (
        (df["serum_creatinine"] > 1.5).astype(int) *
        (df["ejection_fraction"] < 30).astype(int)
    )

    # Age-adjusted kidney risk: elderly patients tolerate kidney stress less
    df["age_creatinine_interaction"] = df["age"] * df["serum_creatinine"]

    # Hyponatremia flag: serum sodium below 135 is clinically significant
    df["hyponatremia"] = (df["serum_sodium"] < 135).astype(int)

    # Composite comorbidity score: sum of binary risk factors
    df["comorbidity_score"] = df["anaemia"] + df["high_blood_pressure"]

    logger.info(f"Feature engineering complete: {len(df.columns)} total features")
    logger.info(f"  kidney_heart_risk positive: {df['kidney_heart_risk'].sum()} patients")
    logger.info(f"  hyponatremia positive: {df['hyponatremia'].sum()} patients")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, threshold=DECISION_THRESHOLD, model_name=""):
    """
    Evaluate model using clinical threshold (0.3 not 0.5).
    Primary metric: Recall for death class.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "precision":         round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":          round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_test, y_proba), 4),
        "decision_threshold": threshold
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"\n{'='*50}")
    logger.info(f"{model_name} — Performance at threshold {threshold}")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"  True Negatives  (Correct Survived): {tn}")
    logger.info(f"  False Positives (Over-flagged):      {fp}")
    logger.info(f"  False Negatives (MISSED DEATHS):     {fn}  ← minimize this")
    logger.info(f"  True Positives  (Correct Deaths):    {tp}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Survived', 'Died'])}")

    return metrics, y_proba

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN WITH MLFLOW TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def train_and_track(name, model, X_train, X_test, y_train, y_test,
                    params, log_fn=mlflow.sklearn.log_model):
    """Train one model and log everything to MLflow."""

    with mlflow.start_run(run_name=name):
        logger.info(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        metrics, y_proba = evaluate_model(model, X_test, y_test, model_name=name)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_params({"selected_features": str(SELECTED_FEATURES),
                           "decision_threshold": DECISION_THRESHOLD})
        mlflow.log_metrics(metrics)
        log_fn(model, name="model")

        # Log confusion matrix as artifact
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        with open(f"confusion_matrix_{name.lower().replace(' ', '_')}.txt", "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Threshold: {DECISION_THRESHOLD}\n\n")
            f.write(str(cm))
            f.write(f"\n\nFalse Negatives (missed deaths): {cm[1][0]}")
        mlflow.log_artifact(f"confusion_matrix_{name.lower().replace(' ', '_')}.txt")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"  MLflow run ID: {run_id}")

    return metrics, run_id

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Heart Failure Model Training Pipeline — Start")
    logger.info("=" * 60)

    # ── Load and prepare data ─────────────────────────────────────────────
    df = load_data()
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    logger.info(f"\nFeatures used for training: {list(X.columns)}")
    logger.info(f"Class distribution: {dict(y.value_counts())}")
    logger.info(f"Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.2f}:1")

    # Stratified split — preserves class ratio in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── MLflow setup ──────────────────────────────────────────────────────
    normalize_mlflow_paths(PROJECT_ROOT)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=str(PROJECT_ROOT / "mlruns" / "1"),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}

    # ── Model 1: XGBoost ──────────────────────────────────────────────────
    class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    xgb_params = {
        "n_estimators":      200,
        "max_depth":         4,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "scale_pos_weight":  class_ratio,  # Handle class imbalance
        "eval_metric":       "logloss",
        "random_state":      RANDOM_STATE
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)

    # Create a wrapper for XGBoost that works with our evaluate function
    class XGBWrapper:
        def __init__(self, model):
            self.model = model
        def fit(self, X, y):
            self.model.fit(X, y)
        def predict_proba(self, X):
            return self.model.predict_proba(X)

    metrics_xgb, run_xgb = train_and_track(
        "XGBoost", xgb_model,
        X_train, X_test, y_train, y_test,
        xgb_params, log_fn=mlflow.xgboost.log_model
    )
    results["XGBoost"] = {"metrics": metrics_xgb, "run_id": run_xgb, "model": xgb_model}

    # ── Model 2: Logistic Regression ──────────────────────────────────────
    lr_params = {
        "C":            0.1,
        "max_iter":     1000,
        "class_weight": "balanced",  # Handles imbalance automatically
        "random_state": RANDOM_STATE,
        "solver":       "lbfgs"
    }

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**lr_params)),
    ])
    metrics_lr, run_lr = train_and_track(
        "Logistic Regression", lr_pipeline,
        X_train, X_test, y_train, y_test,
        lr_params
    )
    results["Logistic Regression"] = {"metrics": metrics_lr, "run_id": run_lr}

    # ── Model 3: Random Forest ────────────────────────────────────────────
    rf_params = {
        "n_estimators":  200,
        "max_depth":     6,
        "class_weight":  "balanced",
        "random_state":  RANDOM_STATE
    }
    rf_model = RandomForestClassifier(**rf_params)
    metrics_rf, run_rf = train_and_track(
        "Random Forest", rf_model,
        X_train, X_test, y_train, y_test,
        rf_params
    )
    results["Random Forest"] = {"metrics": metrics_rf, "run_id": run_rf, "model": rf_model}

    # ── Compare and select best model ─────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON — Primary metric: RECALL (death class)")
    logger.info("=" * 60)

    comparison = []
    for name, result in results.items():
        m = result["metrics"]
        comparison.append({
            "Model":     name,
            "Recall":    m["recall"],
            "Precision": m["precision"],
            "ROC-AUC":   m["roc_auc"],
            "F1":        m["f1_score"],
            "Accuracy":  m["accuracy"]
        })
        logger.info(f"\n{name}:")
        logger.info(f"  Recall:    {m['recall']} ← PRIMARY METRIC")
        logger.info(f"  ROC-AUC:   {m['roc_auc']}")
        logger.info(f"  Precision: {m['precision']}")
        logger.info(f"  F1:        {m['f1_score']}")

    # Select best by recall
    best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["recall"])
    best_run_id = results[best_name]["run_id"]

    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_name}")
    logger.info(f"  Recall: {results[best_name]['metrics']['recall']}")
    logger.info(f"  Run ID: {best_run_id}")
    logger.info(f"  Decision threshold: {DECISION_THRESHOLD}")
    logger.info(f"{'='*60}")

    # Feature importance for XGBoost
    if hasattr(xgb_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature":    X.columns,
            "importance": xgb_model.feature_importances_
        }).sort_values("importance", ascending=False)

        logger.info("\nXGBoost Feature Importance (descending):")
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        importance_df.to_csv("feature_importance.csv", index=False)

    logger.info("\n=== Training Pipeline Complete ===")
    logger.info(f"View results: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
