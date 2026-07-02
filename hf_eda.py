"""
Heart Failure Prediction — Exploratory Data Analysis
======================================================
Clinical EDA producing charts designed for three audiences:
    1. Hospital risk teams — mortality patterns and risk factors
    2. Clinicians — feature distributions and clinical thresholds
    3. Data scientists — correlations, class balance, feature relationships

All charts saved to the /reports/figures/ folder.

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import psycopg2
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "heart_failure_db",
    "user":     "postgres",
    "password": "postgres"
}

OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette — clinical, professional ────────────────────────────────────
SURVIVED_COLOR = "#2196F3"   # Blue — survived
DIED_COLOR     = "#F44336"   # Red — died
NEUTRAL_COLOR  = "#607D8B"   # Slate
ACCENT_COLOR   = "#009688"   # Teal

# ── Load data from PostgreSQL ─────────────────────────────────────────────────
def load_data():
    """Pull both datasets from PostgreSQL into DataFrames."""
    conn = psycopg2.connect(**DB_CONFIG)

    clinical = pd.read_sql("""
        SELECT age, anaemia, creatinine_phosphokinase, diabetes,
               ejection_fraction, high_blood_pressure, platelets,
               serum_creatinine, serum_sodium, sex, smoking, time, death_event
        FROM raw.patients_clinical
    """, conn)

    diagnostic = pd.read_sql("""
        SELECT age, sex, chest_pain_type, resting_bp, cholesterol,
               fasting_bs, resting_ecg, max_hr, exercise_angina,
               oldpeak, st_slope, heart_disease
        FROM raw.patients_diagnostic
    """, conn)

    conn.close()
    print(f"Clinical dataset loaded: {len(clinical)} rows")
    print(f"Diagnostic dataset loaded: {len(diagnostic)} rows")
    return clinical, diagnostic

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Class Distribution (Survival vs Death)
# Purpose: Show stakeholders the baseline imbalance
# ══════════════════════════════════════════════════════════════════════════════
def plot_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Patient Outcome Distribution", fontsize=16, fontweight="bold", y=1.02)

    counts = df["death_event"].value_counts()
    labels = ["Survived", "Died"]
    colors = [SURVIVED_COLOR, DIED_COLOR]

    # Bar chart
    bars = axes[0].bar(labels, [counts[0], counts[1]], color=colors, width=0.5, edgecolor="white")
    axes[0].set_title("Count of Outcomes", fontsize=13)
    axes[0].set_ylabel("Number of Patients")
    for bar, count in zip(bars, [counts[0], counts[1]]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(count), ha="center", fontweight="bold", fontsize=12)
    axes[0].set_ylim(0, 240)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Pie chart
    axes[1].pie([counts[0], counts[1]], labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 12})
    axes[1].set_title("Proportion of Outcomes", fontsize=13)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Age Distribution by Outcome
# Purpose: Show mortality increases with age
# ══════════════════════════════════════════════════════════════════════════════
def plot_age_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Age Distribution by Patient Outcome", fontsize=16, fontweight="bold")

    survived = df[df["death_event"] == 0]["age"]
    died     = df[df["death_event"] == 1]["age"]

    # Histogram
    axes[0].hist(survived, bins=20, alpha=0.7, color=SURVIVED_COLOR, label=f"Survived (n={len(survived)})")
    axes[0].hist(died,     bins=20, alpha=0.7, color=DIED_COLOR,     label=f"Died (n={len(died)})")
    axes[0].axvline(survived.mean(), color=SURVIVED_COLOR, linestyle="--", linewidth=2,
                   label=f"Survived avg: {survived.mean():.1f}")
    axes[0].axvline(died.mean(), color=DIED_COLOR, linestyle="--", linewidth=2,
                   label=f"Died avg: {died.mean():.1f}")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Number of Patients")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Box plot
    data_to_plot = [survived, died]
    bp = axes[1].boxplot(data_to_plot, tick_labels=["Survived", "Died"],
                         patch_artist=True, notch=False,
                         medianprops={"color": "white", "linewidth": 2})
    bp["boxes"][0].set_facecolor(SURVIVED_COLOR)
    bp["boxes"][1].set_facecolor(DIED_COLOR)
    axes[1].set_title("Age Spread by Outcome")
    axes[1].set_ylabel("Age (years)")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_age_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Key Clinical Markers by Outcome
# Purpose: Show which markers separate survivors from non-survivors
# ══════════════════════════════════════════════════════════════════════════════
def plot_clinical_markers(df):
    markers = [
        ("ejection_fraction", "Ejection Fraction (%)", "Lower = worse cardiac function"),
        ("serum_creatinine",  "Serum Creatinine (mg/dL)", "Higher = worse kidney function"),
        ("serum_sodium",      "Serum Sodium (mEq/L)", "Lower = worse fluid balance"),
        ("creatinine_phosphokinase", "CPK (mcg/L)", "Enzyme released when heart muscle damaged"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Key Clinical Markers: Survivors vs Non-Survivors",
                 fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i, (col, label, note) in enumerate(markers):
        survived = df[df["death_event"] == 0][col]
        died     = df[df["death_event"] == 1][col]

        bp = axes[i].boxplot([survived, died], tick_labels=["Survived", "Died"],
                             patch_artist=True,
                             medianprops={"color": "white", "linewidth": 2},
                             flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
        bp["boxes"][0].set_facecolor(SURVIVED_COLOR)
        bp["boxes"][1].set_facecolor(DIED_COLOR)
        axes[i].set_title(label, fontsize=12, fontweight="bold")
        axes[i].set_ylabel(label)
        axes[i].text(0.5, -0.12, note, transform=axes[i].transAxes,
                    ha="center", fontsize=9, color="gray", style="italic")
        axes[i].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_clinical_markers.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Death Rate by Age Group (Bar chart for stakeholders)
# Purpose: Clear, simple, actionable for hospital risk teams
# ══════════════════════════════════════════════════════════════════════════════
def plot_death_rate_by_age(df):
    df = df.copy()
    df["age_group"] = pd.cut(df["age"],
                             bins=[0, 49, 64, 74, 120],
                             labels=["Under 50", "50-64", "65-74", "75+"])

    summary = df.groupby("age_group", observed=True).agg(
        total=("death_event", "count"),
        deaths=("death_event", "sum")
    ).reset_index()
    summary["death_rate"] = (summary["deaths"] / summary["total"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [SURVIVED_COLOR if r < 40 else DIED_COLOR for r in summary["death_rate"]]
    bars = ax.bar(summary["age_group"], summary["death_rate"],
                  color=colors, width=0.5, edgecolor="white")

    for bar, rate, total in zip(bars, summary["death_rate"], summary["total"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{rate}%\n(n={total})", ha="center", fontsize=11, fontweight="bold")

    ax.axhline(y=40, color="orange", linestyle="--", linewidth=1.5, label="40% threshold")
    ax.set_title("Mortality Rate by Age Group", fontsize=15, fontweight="bold")
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Death Rate (%)", fontsize=12)
    ax.set_ylim(0, 75)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    survived_patch = mpatches.Patch(color=SURVIVED_COLOR, label="Below 40% threshold")
    died_patch     = mpatches.Patch(color=DIED_COLOR,     label="Above 40% threshold")
    ax.legend(handles=[survived_patch, died_patch], fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_death_rate_by_age.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Correlation Heatmap
# Purpose: Show data scientists which features correlate with death
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df):
    numeric_cols = ["age", "ejection_fraction", "serum_creatinine",
                    "serum_sodium", "creatinine_phosphokinase",
                    "platelets", "time", "death_event"]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.5,
                annot_kws={"size": 10})
    ax.set_title("Feature Correlation Matrix\n(Focus on death_event row)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Risk Factor Prevalence (Categorical features)
# Purpose: Show which binary risk factors are most common in deaths
# ══════════════════════════════════════════════════════════════════════════════
def plot_risk_factor_prevalence(df):
    risk_factors = ["anaemia", "diabetes", "high_blood_pressure", "smoking"]
    labels       = ["Anaemia", "Diabetes", "High Blood\nPressure", "Smoking"]

    survived = df[df["death_event"] == 0]
    died     = df[df["death_event"] == 1]

    survived_rates = [survived[f].mean() * 100 for f in risk_factors]
    died_rates     = [died[f].mean() * 100     for f in risk_factors]

    x = np.arange(len(risk_factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, survived_rates, width, label="Survived",
                   color=SURVIVED_COLOR, alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, died_rates,     width, label="Died",
                   color=DIED_COLOR,     alpha=0.85, edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{bar.get_height():.1f}%", ha="center", fontsize=9)

    ax.set_title("Risk Factor Prevalence: Survivors vs Non-Survivors",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Prevalence (%)", fontsize=11)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_risk_factor_prevalence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Diagnostic Dataset: Heart Disease by Chest Pain Type and Sex
# Purpose: The most clinically important finding from the diagnostic data
# ══════════════════════════════════════════════════════════════════════════════
def plot_chest_pain_analysis(df):
    summary = df.groupby(["chest_pain_type", "sex"]).agg(
        total=("heart_disease", "count"),
        disease=("heart_disease", "sum")
    ).reset_index()
    summary["rate"] = (summary["disease"] / summary["total"] * 100).round(1)

    pain_order = ["ASY", "NAP", "ATA", "TA"]
    summary["chest_pain_type"] = pd.Categorical(summary["chest_pain_type"],
                                                  categories=pain_order, ordered=True)
    summary = summary.sort_values("chest_pain_type")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pain_order))
    width = 0.35

    male   = summary[summary["sex"] == "M"]
    female = summary[summary["sex"] == "F"]

    bars_m = ax.bar(x - width/2, male["rate"],   width, label="Male",
                    color="#1565C0", alpha=0.85, edgecolor="white")
    bars_f = ax.bar(x + width/2, female["rate"], width, label="Female",
                    color="#AD1457", alpha=0.85, edgecolor="white")

    for bar in list(bars_m) + list(bars_f):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{bar.get_height():.1f}%", ha="center", fontsize=9)

    ax.set_title("Heart Disease Rate by Chest Pain Type and Sex\n"
                 "ASY = Asymptomatic  |  NAP = Non-Anginal Pain  |  "
                 "ATA = Atypical Angina  |  TA = Typical Angina",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pain_order, fontsize=12)
    ax.set_ylabel("Heart Disease Rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/07_chest_pain_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Ejection Fraction vs Serum Creatinine Scatter (Risk zones)
# Purpose: Visualize the risk triage zones we built in SQL
# ══════════════════════════════════════════════════════════════════════════════
def plot_risk_zones(df):
    fig, ax = plt.subplots(figsize=(12, 7))

    survived = df[df["death_event"] == 0]
    died     = df[df["death_event"] == 1]

    ax.scatter(survived["ejection_fraction"], survived["serum_creatinine"],
               c=SURVIVED_COLOR, alpha=0.6, s=60, label="Survived", zorder=3)
    ax.scatter(died["ejection_fraction"], died["serum_creatinine"],
               c=DIED_COLOR, alpha=0.6, s=60, label="Died", zorder=3)

    # Risk zone boundaries from our SQL CTE
    ax.axvline(x=30, color="orange", linestyle="--", linewidth=1.5,
               label="EF = 30 (High Risk threshold)")
    ax.axhline(y=1.5, color="purple", linestyle="--", linewidth=1.5,
               label="Creatinine = 1.5 (High Risk threshold)")

    # Shade high risk zone
    ax.axvspan(0, 30, alpha=0.05, color="red")
    ax.fill_betweenx([1.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 1.5 else 6],
                     0, 30, alpha=0.05, color="red")

    ax.set_title("Risk Zone Map: Ejection Fraction vs Serum Creatinine\n"
                 "Bottom-left quadrant = High Risk (EF < 30 AND Creatinine > 1.5)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Ejection Fraction (%)", fontsize=12)
    ax.set_ylabel("Serum Creatinine (mg/dL)", fontsize=12)
    ax.set_ylim(0, 6)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/08_risk_zone_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Heart Failure EDA — Generating Clinical Charts")
    print("=" * 60)

    clinical, diagnostic = load_data()

    print("\nGenerating charts...")
    plot_class_distribution(clinical)
    plot_age_distribution(clinical)
    plot_clinical_markers(clinical)
    plot_death_rate_by_age(clinical)
    plot_correlation_heatmap(clinical)
    plot_risk_factor_prevalence(clinical)
    plot_chest_pain_analysis(diagnostic)
    plot_risk_zones(clinical)

    print("\n" + "=" * 60)
    print(f"EDA Complete — 8 charts saved to {OUTPUT_DIR}/")
    print("=" * 60)
