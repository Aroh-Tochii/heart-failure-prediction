"""In-memory clinical data from CSV — used for cloud deploy without PostgreSQL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CLINICAL_CSV = PROJECT_ROOT / "heart_failure_clinical_records_dataset (1).csv"

_df: pd.DataFrame | None = None


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    if "death_event" not in df.columns:
        for col in list(df.columns):
            if col.upper() == "death_event":
                df = df.rename(columns={col: "death_event"})
                break

    df["kidney_heart_risk"] = (
        (df["serum_creatinine"] > 1.5) & (df["ejection_fraction"] < 30)
    )
    df["hyponatremia"] = df["serum_sodium"] < 135
    df["age_creatinine_interaction"] = df["age"] * df["serum_creatinine"]
    df["comorbidity_score"] = df["anaemia"] + df["high_blood_pressure"]
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    return df


def get_dataframe() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = _engineer(pd.read_csv(CLINICAL_CSV))
    return _df.copy()


def get_patient_row(patient_id: int) -> pd.Series | None:
    df = get_dataframe()
    match = df[df["id"] == int(patient_id)]
    if match.empty:
        return None
    return match.iloc[0]


def insert_patient(patient_data: dict) -> int:
    global _df
    df = get_dataframe()
    new_id = int(df["id"].max()) + 1
    row = dict(patient_data)
    row["id"] = new_id
    row["kidney_heart_risk"] = bool(
        row.get("serum_creatinine", 0) > 1.5
        and row.get("ejection_fraction", 100) < 30
    )
    row["hyponatremia"] = bool(row.get("serum_sodium", 140) < 135)
    row["age_creatinine_interaction"] = row.get("age", 0) * row.get("serum_creatinine", 0)
    row["comorbidity_score"] = row.get("anaemia", 0) + row.get("high_blood_pressure", 0)
    _df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return new_id
