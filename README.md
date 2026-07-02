# Heart Failure Clinical Decision Support System

A production-grade machine learning system that predicts patient mortality risk and supports clinical triage decisions for heart failure patients — built as a fully containerized, monitored, and CI/CD-automated system.

## Overview

This system transforms raw clinical data into a live decision support service used by three audiences: hospital risk teams, clinicians, and insurance assessors. It covers the complete ML lifecycle across two clinical datasets — from structured database design and SQL analysis through model training, API deployment, drift monitoring, and automated testing.

## Architecture

```
Two Clinical Datasets (299 + 917 records)
              ↓
ETL Pipeline → PostgreSQL (3 schemas: raw, analytics, predictions)
              ↓
SQL Analysis (8 clinical queries)
              ↓
EDA → 8 Clinical Charts
              ↓
Feature Engineering (4 composite clinical features)
              ↓
3 Models (XGBoost, Logistic Regression, Random Forest) → MLflow
              ↓
FastAPI Clinical Decision Support Service
              ↓
Docker (multi-container: API + PostgreSQL)
              ↓
Evidently AI Drift Monitoring (2 datasets monitored)
              ↓
GitHub Actions CI/CD (5 jobs including clinical logic validation)
```

## Tech Stack

| Layer | Tool |
|---|---|
| Data storage | PostgreSQL (3 schemas) |
| Data processing | Python, Pandas |
| Models | XGBoost, Logistic Regression, Random Forest |
| Experiment tracking | MLflow |
| API serving | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Drift monitoring | Evidently AI |
| CI/CD | GitHub Actions |

## Project Structure

```
.
├── hf_etl_pipeline.py        # Dual-dataset ETL into PostgreSQL schemas
├── hf_train_model.py         # 3-model training with MLflow tracking
├── hf_eda.py                 # 8 clinical visualization charts
├── hf_app.py                 # FastAPI clinical decision support service
├── hf_monitor_drift.py       # Dual-dataset drift monitoring
├── hf_Dockerfile             # Container image for API service
├── hf_docker_compose.yml     # Orchestrates API + PostgreSQL containers
├── hf_requirements.txt       # Python dependencies
├── reports/
│   ├── figures/              # 8 EDA clinical charts
│   ├── drift/                # Timestamped drift reports
│   └── reference/            # Frozen baseline snapshots
└── .github/workflows/
    └── hf_ci_cd.yml          # 5-job automated pipeline
```

## Datasets

| Dataset | Records | Target | Source |
|---|---|---|---|
| Clinical follow-up | 299 | `death_event` — mortality during follow-up | Chicco & Jurman (2020) |
| Diagnostic records | 917 | `heart_disease` — disease presence at diagnosis | UCI Heart Disease (combined) |

## Database Design

Three schemas deliberately separate data by purpose:

```sql
raw.patients_clinical      -- 299 follow-up records (untouched source data)
raw.patients_diagnostic    -- 917 diagnostic records (untouched source data)
predictions.risk_predictions -- Every API prediction logged with timestamp
```

## Model Performance

Primary optimization target: **Recall** (minimizing missed deaths)

Clinical reasoning: A false negative (missed death) carries significantly higher cost than a false positive (unnecessary monitoring). The system is tuned accordingly.

| Model | Recall | ROC-AUC | Missed Deaths | Decision Threshold |
|---|---|---|---|---|
| Logistic Regression | **0.895** | 0.882 | **2** | 0.3 |
| Random Forest | 0.842 | 0.900 | 3 | 0.3 |
| XGBoost | 0.684 | 0.858 | 6 | 0.3 |

**Production model: Logistic Regression** — selected for highest recall (fewest missed deaths).

## Feature Engineering

4 composite features engineered from clinical domain knowledge:

| Feature | Logic | Clinical Rationale |
|---|---|---|
| `kidney_heart_risk` | creatinine > 1.5 AND ejection_fraction < 30 | Combined organ failure is more dangerous than either alone |
| `age_creatinine_interaction` | age × creatinine | Elderly patients tolerate kidney stress less |
| `hyponatremia` | serum_sodium < 135 | Clinically significant fluid imbalance threshold |
| `comorbidity_score` | anaemia + high_blood_pressure | Cumulative binary risk factor count |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict/death-risk` | POST | Mortality risk prediction with clinical flags |
| `/predict/heart-disease` | GET | Heart disease detection (v2.0) |
| `/health` | GET | Service health check |
| `/stats` | GET | Aggregate prediction statistics from database |
| `/docs` | GET | Interactive API documentation |

### Example Request

```bash
curl -X POST http://localhost:8002/predict/death-risk \
  -H "Content-Type: application/json" \
  -d '{
    "serum_creatinine": 1.9,
    "ejection_fraction": 20,
    "age": 65,
    "serum_sodium": 130,
    "time": 30,
    "anaemia": 1,
    "high_blood_pressure": 1
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.9691,
  "risk_level": "High",
  "clinical_flags": [
    "CRITICAL: Severely reduced ejection fraction (<30%)",
    "WARNING: Elevated creatinine — kidney stress detected",
    "WARNING: Low sodium — fluid imbalance detected",
    "NOTE: Multiple comorbidities — anaemia + hypertension"
  ],
  "recommendation": "IMMEDIATE ACTION REQUIRED. Escalate to senior clinician. Consider admission.",
  "model_used": "Logistic Regression (recall-optimized)",
  "decision_threshold": 0.3,
  "predicted_at": "2026-07-02T09:04:53.113103"
}
```

## Running Locally

**Start the full stack:**
```bash
docker compose -f hf_docker_compose.yml up -d
```

**Stop the full stack:**
```bash
docker compose -f hf_docker_compose.yml down
```

**Run ETL pipeline:**
```bash
python3 hf_etl_pipeline.py
```

**Train models:**
```bash
python3 hf_train_model.py
```

**Generate EDA charts:**
```bash
python3 hf_eda.py
```

**Run drift monitoring:**
```bash
python3 hf_monitor_drift.py
```

**View MLflow experiments:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

## Key Clinical Findings (from SQL and EDA)

- Patients who died had significantly lower ejection fraction (33.5% vs 40.3%) and higher serum creatinine (1.84 vs 1.18) than survivors
- Patients aged 75+ had a 61% mortality rate — more than double the under-50 group (23%)
- Asymptomatic male patients had an 82.9% heart disease rate — the highest of any demographic group
- The SQL-derived triage rule (EF < 30 AND creatinine > 1.5) correctly identified patients with 72.7% death rate

## Why This Project

This system demonstrates the full production ML lifecycle for a high-stakes clinical domain: structured multi-schema database design, SQL-driven analytical reasoning, recall-optimized model selection under class imbalance, clinical explainability through feature flags, containerized deployment, ongoing drift monitoring, and automated quality gates — every layer a production ML engineer is responsible for.
