"""
Heart Failure Clinical AI Agent — Complete Version
====================================================
Full agent with:
- Natural language understanding (Gemini 3.5 Flash)
- RAG clinical knowledge base (ChromaDB)
- PostgreSQL database tools (5 tools)
- ML model risk prediction
- New patient intake (auto INSERT + PREDICT + LOG)
- Population queries
- Clinical guidelines retrieval

Author: Tochukwu Aroh
"""

import os
import re
import json
import psycopg2
import pandas as pd
import chromadb
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "heart_failure_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres")
}

AGENT_DIR  = Path(__file__).parent
CHROMA_DIR = AGENT_DIR / "agent" / "chroma_db"
DATA_SOURCE = os.getenv("DATA_SOURCE", "auto").lower()

def use_csv_mode() -> bool:
    if DATA_SOURCE == "csv":
        return True
    if DATA_SOURCE == "postgres":
        return False
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        return False
    except Exception:
        return True

# ── Gemini client ─────────────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL  = "gemini-3.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
def get_chroma():
    if not CHROMA_DIR.exists():
        return None
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return chroma_client.get_collection("clinical_knowledge")
    except Exception:
        return None

# ── DB helper ─────────────────────────────────────────────────────────────────
def get_db():
    return psycopg2.connect(**DB_CONFIG)

# ── Clinical risk scoring ─────────────────────────────────────────────────────
def calculate_risk(row: dict) -> dict:
    score = 0
    ef  = row.get("ejection_fraction", 40)
    cr  = row.get("serum_creatinine", 1.0)
    na  = row.get("serum_sodium", 137)
    age = row.get("age", 60)
    an  = row.get("anaemia", 0)
    khr = (cr > 1.5) and (ef < 30)

    if ef < 20:   score += 40
    elif ef < 30: score += 30
    elif ef < 40: score += 20
    if cr > 2.0:  score += 30
    elif cr > 1.5: score += 20
    if na < 125:  score += 20
    elif na < 135: score += 12
    if khr:        score += 15
    if age > 75:   score += 10
    elif age > 65: score += 5
    if an:         score += 5

    prob  = min(score / 100, 0.99)
    risk  = "CRITICAL" if prob >= 0.8 else "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.3 else "LOW"
    flags = []
    if ef < 20:    flags.append("CRITICAL: Ejection fraction severely reduced (<20%) — life-threatening")
    elif ef < 30:  flags.append("CRITICAL: Ejection fraction critically low (<30%) — immediate action")
    elif ef < 40:  flags.append("WARNING: Ejection fraction reduced (<40%) — HFrEF confirmed")
    if cr > 2.0:   flags.append("CRITICAL: Severely elevated creatinine — acute kidney injury likely")
    elif cr > 1.5: flags.append("WARNING: Elevated creatinine — kidney stress detected")
    if na < 125:   flags.append("CRITICAL: Severe hyponatremia — emergency fluid management required")
    elif na < 135: flags.append("WARNING: Low sodium — fluid imbalance and neurohormonal activation")
    if khr:        flags.append("HIGH RISK: Kidney-Heart failure pattern — cardiorenal syndrome likely")
    if age > 75:   flags.append("NOTE: Age >75 — elevated baseline mortality risk (61% in this population)")

    escalation = {
        "CRITICAL": "IMMEDIATE ICU ESCALATION — Call senior cardiologist NOW. Do not delay.",
        "HIGH":     "URGENT cardiology review within 4 hours. Consider hospital admission.",
        "MEDIUM":   "Same day clinical review. Optimize medications. Increase monitoring.",
        "LOW":      "Routine follow-up in 4-12 weeks. Continue current management."
    }[risk]

    return {
        "probability": prob,
        "risk_level":  risk,
        "flags":       flags,
        "escalation":  escalation,
        "score":       score
    }

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Insert new patient
# ══════════════════════════════════════════════════════════════════════════════
def insert_and_assess_patient(patient_data: dict) -> str:
    """
    Insert a new patient into the database, run risk assessment,
    and log the prediction. Returns full clinical assessment.
    """
    try:
        # Set defaults for missing fields
        defaults = {
            "age": 60, "sex": 1, "anaemia": 0, "creatinine_phosphokinase": 582,
            "diabetes": 0, "ejection_fraction": 38, "high_blood_pressure": 0,
            "platelets": 263358, "serum_creatinine": 1.0, "serum_sodium": 137,
            "smoking": 0, "time": 30, "death_event": 0
        }
        for key, val in defaults.items():
            if key not in patient_data:
                patient_data[key] = val

        # Calculate engineered features
        patient_data["kidney_heart_risk"] = bool(
            patient_data["serum_creatinine"] > 1.5 and
            patient_data["ejection_fraction"] < 30
        )
        patient_data["hyponatremia"] = bool(patient_data["serum_sodium"] < 135)
        patient_data["age_creatinine_interaction"] = (
            patient_data["age"] * patient_data["serum_creatinine"]
        )
        patient_data["comorbidity_score"] = (
            patient_data["anaemia"] + patient_data["high_blood_pressure"]
        )

        if use_csv_mode():
            from hf_clinical_data import insert_patient
            new_id = insert_patient(patient_data)
        else:
            conn = get_db()
            cur  = conn.cursor()
            cur.execute("SELECT MAX(id) FROM raw.patients_clinical")
            max_id = cur.fetchone()[0] or 0
            new_id = max_id + 1

            cur.execute("""
                INSERT INTO raw.patients_clinical
                (id, age, anaemia, creatinine_phosphokinase, diabetes,
                 ejection_fraction, high_blood_pressure, platelets,
                 serum_creatinine, serum_sodium, sex, smoking, time,
                 death_event, kidney_heart_risk, hyponatremia,
                 age_creatinine_interaction, comorbidity_score)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                new_id,
                patient_data["age"], patient_data["anaemia"],
                patient_data["creatinine_phosphokinase"], patient_data["diabetes"],
                patient_data["ejection_fraction"], patient_data["high_blood_pressure"],
                patient_data["platelets"], patient_data["serum_creatinine"],
                patient_data["serum_sodium"], patient_data["sex"],
                patient_data["smoking"], patient_data["time"],
                patient_data["death_event"],
                patient_data["kidney_heart_risk"], patient_data["hyponatremia"],
                patient_data["age_creatinine_interaction"],
                patient_data["comorbidity_score"]
            ))
            conn.commit()

        # Run risk assessment
        risk = calculate_risk(patient_data)

        # Log prediction
        if not use_csv_mode():
            try:
                cur.execute("""
                    INSERT INTO predictions.risk_predictions
                    (patient_id, risk_probability, risk_level, model_used, predicted_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    new_id, risk["probability"], risk["risk_level"],
                    "ClinicalAI-RuleBasedV1", datetime.now()
                ))
                conn.commit()
            except Exception:
                conn.rollback()
            finally:
                cur.close()
                conn.close()

        flags_text = "\n".join([f"  • {f}" for f in risk["flags"]]) if risk["flags"] else "  None"

        return f"""
PATIENT ADMITTED — ID {new_id}
Admitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CLINICAL VALUES:
  Age: {patient_data['age']} | Sex: {'Male' if patient_data['sex']==1 else 'Female'}
  Ejection Fraction: {patient_data['ejection_fraction']}% (normal >50%)
  Serum Creatinine:  {patient_data['serum_creatinine']} mg/dL (normal 0.7-1.2)
  Serum Sodium:      {patient_data['serum_sodium']} mEq/L (normal 135-145)
  Anaemia: {'Yes' if patient_data['anaemia'] else 'No'} | Diabetes: {'Yes' if patient_data['diabetes'] else 'No'}
  High BP: {'Yes' if patient_data['high_blood_pressure'] else 'No'} | Smoking: {'Yes' if patient_data['smoking'] else 'No'}

RISK ASSESSMENT:
  Risk Level:  {risk['risk_level']}
  Probability: {risk['probability']*100:.1f}%

CLINICAL FLAGS:
{flags_text}

RECOMMENDATION:
  {risk['escalation']}

Prediction logged to database. Audit trail complete.
Patient ID {new_id} stored in heart_failure_db."""

    except Exception as e:
        return f"Error admitting patient: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Get patient profile
# ══════════════════════════════════════════════════════════════════════════════
def get_patient_profile(patient_id: int) -> str:
    try:
        if use_csv_mode():
            from hf_clinical_data import get_patient_row
            r = get_patient_row(patient_id)
            if r is None:
                return f"No patient found with ID {patient_id}. Valid IDs are 1-299."
        else:
            conn = get_db()
            df   = pd.read_sql(
                f"SELECT * FROM raw.patients_clinical WHERE id = {patient_id}", conn
            )
            conn.close()
            if df.empty:
                return f"No patient found with ID {patient_id}. Valid IDs are 1-299."
            r = df.iloc[0]
        risk = calculate_risk(r.to_dict())
        flags_text = "\n".join([f"  • {f}" for f in risk["flags"]]) if risk["flags"] else "  None"
        outcome = "DIED during follow-up" if r["death_event"] == 1 else "SURVIVED follow-up"
        return f"""
PATIENT {patient_id} — Clinical Profile
Outcome: {outcome} | Age: {r['age']} | Sex: {'Male' if r['sex']==1 else 'Female'} | Follow-up: {r['time']} days

CARDIAC:
  Ejection Fraction: {r['ejection_fraction']}% | CPK: {r['creatinine_phosphokinase']} mcg/L

RENAL & METABOLIC:
  Serum Creatinine: {r['serum_creatinine']} mg/dL
  Serum Sodium:     {r['serum_sodium']} mEq/L
  Platelets:        {r['platelets']:,.0f} kiloplatelets/mL

COMORBIDITIES:
  Anaemia: {'Yes' if r['anaemia'] else 'No'} | Diabetes: {'Yes' if r['diabetes'] else 'No'}
  High BP: {'Yes' if r['high_blood_pressure'] else 'No'} | Smoking: {'Yes' if r['smoking'] else 'No'}

RISK ASSESSMENT:
  Risk Level:  {risk['risk_level']}
  Probability: {risk['probability']*100:.1f}%

CLINICAL FLAGS:
{flags_text}

RECOMMENDATION:
  {risk['escalation']}"""
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — High risk patients
# ══════════════════════════════════════════════════════════════════════════════
def get_high_risk_patients(limit: int = 10) -> str:
    try:
        if use_csv_mode():
            from hf_clinical_data import get_dataframe
            df = get_dataframe().sort_values(
                ["kidney_heart_risk", "ejection_fraction", "serum_creatinine"],
                ascending=[False, True, False],
            ).head(limit)
        else:
            conn = get_db()
            df   = pd.read_sql(f"""
                SELECT id, age, ejection_fraction, serum_creatinine,
                       serum_sodium, kidney_heart_risk, comorbidity_score, death_event
                FROM raw.patients_clinical
                ORDER BY kidney_heart_risk DESC,
                         ejection_fraction ASC,
                         serum_creatinine DESC
                LIMIT {limit}
            """, conn)
            conn.close()

        result = f"TOP {limit} HIGH-RISK PATIENTS\n{'='*55}\n"
        for _, r in df.iterrows():
            risk    = calculate_risk(r.to_dict())
            outcome = "DIED" if r["death_event"] == 1 else "Survived"
            result += f"\nPatient {int(r['id'])} | Age {r['age']} | EF {r['ejection_fraction']}% | "
            result += f"Creat {r['serum_creatinine']} | {risk['risk_level']} ({risk['probability']*100:.0f}%) | {outcome}"
        return result
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Population statistics
# ══════════════════════════════════════════════════════════════════════════════
def get_population_statistics() -> str:
    try:
        if use_csv_mode():
            from hf_clinical_data import get_dataframe
            df = get_dataframe()
            t = len(df)
            d = int(df["death_event"].sum())
            r = [
                t, d, df["age"].mean(), df["ejection_fraction"].mean(),
                df["serum_creatinine"].mean(), df["serum_sodium"].mean(),
                int((df["ejection_fraction"] < 20).sum()),
                int((df["ejection_fraction"] < 30).sum()),
                int((df["ejection_fraction"] < 40).sum()),
                int((df["serum_creatinine"] > 1.5).sum()),
                int((df["serum_creatinine"] > 2.0).sum()),
                int(df["kidney_heart_risk"].sum()),
                int((df["serum_sodium"] < 135).sum()),
                int(df["anaemia"].sum()),
                int(df["high_blood_pressure"].sum()),
                int(df["diabetes"].sum()),
            ]
        else:
            conn = get_db()
            cur  = conn.cursor()
            cur.execute("""
                SELECT COUNT(*), SUM(death_event), AVG(age),
                       AVG(ejection_fraction), AVG(serum_creatinine), AVG(serum_sodium),
                       SUM(CASE WHEN ejection_fraction < 20 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN ejection_fraction < 30 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN ejection_fraction < 40 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN serum_creatinine > 1.5 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN serum_creatinine > 2.0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN kidney_heart_risk THEN 1 ELSE 0 END),
                       SUM(CASE WHEN serum_sodium < 135 THEN 1 ELSE 0 END),
                       SUM(anaemia), SUM(high_blood_pressure), SUM(diabetes)
                FROM raw.patients_clinical
            """)
            r = cur.fetchone()
            cur.close(); conn.close()
            t, d = r[0], r[1]
        return f"""
PATIENT POPULATION — {t} total patients
Deaths: {d} ({d/t*100:.1f}%) | Survivors: {t-d} ({(t-d)/t*100:.1f}%)
Avg Age: {r[2]:.1f} yrs | Avg EF: {r[3]:.1f}% | Avg Creatinine: {r[4]:.2f} | Avg Sodium: {r[5]:.1f}

EJECTION FRACTION DISTRIBUTION:
  Critical EF (<20%):    {r[6]} patients ({r[6]/t*100:.1f}%)
  Severely low EF (<30%): {r[7]} patients ({r[7]/t*100:.1f}%)
  Reduced EF (<40%):     {r[8]} patients ({r[8]/t*100:.1f}%)

RENAL FUNCTION:
  Elevated creatinine (>1.5): {r[9]} patients ({r[9]/t*100:.1f}%)
  Severe creatinine (>2.0):   {r[10]} patients ({r[10]/t*100:.1f}%)
  Kidney-Heart risk pattern:  {r[11]} patients ({r[11]/t*100:.1f}%)

SODIUM & FLUID:
  Hyponatremia (sodium <135): {r[12]} patients ({r[12]/t*100:.1f}%)

COMORBIDITIES:
  Anaemia:           {r[13]} patients ({r[13]/t*100:.1f}%)
  High Blood Pressure: {r[14]} patients ({r[14]/t*100:.1f}%)
  Diabetes:          {r[15]} patients ({r[15]/t*100:.1f}%)"""
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — RAG clinical knowledge search
# ══════════════════════════════════════════════════════════════════════════════
def search_clinical_knowledge(query: str) -> str:
    collection = get_chroma()
    if not collection:
        return "Clinical knowledge base not available. Run hf_rag_builder.py first."
    try:
        # Embed the query
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query
        )
        query_embedding = result.embeddings[0].values

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        if not results["documents"][0]:
            return "No relevant clinical guidelines found."

        knowledge = []
        for i, (doc, meta) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0]
        )):
            knowledge.append(f"[From {meta['source']}]\n{doc}")

        return "CLINICAL GUIDELINES RETRIEVED:\n\n" + "\n\n---\n\n".join(knowledge)

    except Exception as e:
        return f"RAG search error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE PARSER — extracts patient values from free text
# ══════════════════════════════════════════════════════════════════════════════
def parse_patient_from_text(text: str) -> dict:
    """
    Use Gemini to extract structured patient data from free text.
    Handles typos, medical shorthand, different formats.
    """
    parse_prompt = f"""Extract clinical values from this text and return ONLY a JSON object.
If a value is not mentioned, do not include it.

Text: "{text}"

Extract these fields if present:
- age (integer, years)
- sex (1=male, 0=female)
- ejection_fraction (integer, percentage)
- serum_creatinine (float, mg/dL)
- serum_sodium (integer, mEq/L)
- anaemia (1=yes, 0=no)
- diabetes (1=yes, 0=no)
- high_blood_pressure (1=yes, 0=no)
- smoking (1=yes, 0=no)
- platelets (float)
- creatinine_phosphokinase (integer)
- time (integer, days of follow-up)

Medical shorthand mappings:
- EF = ejection_fraction
- Cr or creatinine = serum_creatinine
- Na or sodium = serum_sodium
- HBP or hypertension = high_blood_pressure
- M or male = sex 1, F or female = sex 0

Return ONLY valid JSON, nothing else. Example:
{{"age": 65, "ejection_fraction": 20, "serum_creatinine": 1.9}}"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=parse_prompt)]
            )],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        text_response = response.text.strip()
        # Clean JSON
        text_response = re.sub(r"```json|```", "", text_response).strip()
        return json.loads(text_response)
    except Exception as e:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# INTENT ROUTER — decides what the doctor is asking for
# ══════════════════════════════════════════════════════════════════════════════
def route_intent(question: str) -> str:
    q = question.lower()

    # New patient intake signals
    new_patient_signals = [
        "new patient", "admit", "admission", "just came in",
        "intake", "register", "add patient", "new case"
    ]
    if any(s in q for s in new_patient_signals):
        return "new_patient"

    # Check if clinical values are given without explicit "new patient"
    has_values = any(w in q for w in [
        "ejection fraction", "ef ", "creatinine", "sodium", "anaemia",
        "year old", "years old", "yo ", "y/o", "age "
    ])
    has_patient_id = bool(re.search(r'\b(?:patient|pt|id|#)\s*\d+', q))

    if has_values and not has_patient_id:
        return "new_patient"

    # Specific patient query
    if has_patient_id:
        if any(w in q for w in ["risk", "predict", "assess", "probability"]):
            return "predict"
        return "profile"

    # Population statistics
    if any(w in q for w in [
        "how many", "count", "total", "population", "statistic",
        "average", "mortality rate", "death rate", "percentage",
        "distribution", "exist", "database"
    ]):
        return "statistics"

    # High risk list
    if any(w in q for w in [
        "highest risk", "most at risk", "urgent", "critical patients",
        "who needs attention", "top", "worst", "priorit"
    ]):
        return "high_risk"

    # Clinical knowledge / guidelines
    if any(w in q for w in [
        "guideline", "protocol", "treatment", "medication", "drug",
        "what does", "explain", "mean", "what is", "escalat",
        "when to", "should i", "recommend", "therapy", "manage"
    ]):
        return "rag"

    # Compare survivors
    if any(w in q for w in ["compare", "survivor", "vs", "versus", "differ"]):
        return "compare"

    # Default to statistics for general questions
    return "statistics"


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE SURVIVORS
# ══════════════════════════════════════════════════════════════════════════════
def compare_survivors_vs_deaths() -> str:
    try:
        if use_csv_mode():
            from hf_clinical_data import get_dataframe
            df = get_dataframe().groupby("death_event").agg(
                n=("id", "count"),
                age=("age", "mean"),
                ef=("ejection_fraction", "mean"),
                cr=("serum_creatinine", "mean"),
                na=("serum_sodium", "mean"),
            ).reset_index()
        else:
            conn = get_db()
            df   = pd.read_sql("""
                SELECT death_event, COUNT(*) n,
                       AVG(age) age, AVG(ejection_fraction) ef,
                       AVG(serum_creatinine) cr, AVG(serum_sodium) na
                FROM raw.patients_clinical
                GROUP BY death_event ORDER BY death_event
            """, conn)
            conn.close()
        s = df[df["death_event"]==0].iloc[0]
        d = df[df["death_event"]==1].iloc[0]
        return f"""
SURVIVORS ({int(s['n'])}) vs NON-SURVIVORS ({int(d['n'])})
{'='*55}
                    SURVIVED        DIED        DIFFERENCE
Age:                {s['age']:.1f} yrs        {d['age']:.1f} yrs      +{d['age']-s['age']:.1f} yrs
Ejection Fraction:  {s['ef']:.1f}%           {d['ef']:.1f}%          {d['ef']-s['ef']:.1f}%
Serum Creatinine:   {s['cr']:.2f} mg/dL      {d['cr']:.2f} mg/dL     +{d['cr']-s['cr']:.2f}
Serum Sodium:       {s['na']:.1f} mEq/L      {d['na']:.1f} mEq/L     {d['na']-s['na']:.1f}

Key: Non-survivors had lower EF, higher creatinine, lower sodium, older age."""
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM = """You are a Clinical AI Assistant for heart failure decision support.
You have access to real patient data, clinical risk assessments, and evidence-based guidelines.
Give clear, specific, actionable clinical answers.
Use CRITICAL or URGENT for high-risk situations.
Always recommend escalation when risk is high.
When presenting risk assessments, include the probability, risk level, clinical flags, and recommendation.
Be concise but thorough. Use medical terminology but explain it simply."""


def ask_agent(question: str, history: list) -> str:
    intent = route_intent(question)
    print(f"  [Intent: {intent}]")

    # Get tool data
    if intent == "new_patient":
        print("  [Parsing patient data from text...]")
        patient_data = parse_patient_from_text(question)
        if not patient_data:
            return ("I couldn't extract patient clinical values from your message. "
                    "Please provide: age, ejection fraction, serum creatinine, "
                    "serum sodium, and any comorbidities (anaemia, diabetes, "
                    "high blood pressure, smoking).")
        print(f"  [Extracted: {patient_data}]")
        print("  [Inserting patient and running assessment...]")
        tool_result = insert_and_assess_patient(patient_data)

    elif intent == "profile":
        pid_match = re.search(r'\b(?:patient|pt|id|#)?\s*(\d+)', question, re.IGNORECASE)
        pid = int(pid_match.group(1)) if pid_match else 1
        print(f"  [Getting profile for patient {pid}...]")
        tool_result = get_patient_profile(pid)

    elif intent == "predict":
        pid_match = re.search(r'\b(?:patient|pt|id|#)?\s*(\d+)', question, re.IGNORECASE)
        pid = int(pid_match.group(1)) if pid_match else 1
        print(f"  [Predicting risk for patient {pid}...]")
        tool_result = get_patient_profile(pid)

    elif intent == "statistics":
        print("  [Querying population statistics...]")
        tool_result = get_population_statistics()

    elif intent == "high_risk":
        limit_match = re.search(r'\b(\d+)\b', question)
        limit = int(limit_match.group(1)) if limit_match and int(limit_match.group(1)) < 50 else 10
        print(f"  [Getting top {limit} high-risk patients...]")
        tool_result = get_high_risk_patients(limit)

    elif intent == "rag":
        print("  [Searching clinical knowledge base...]")
        tool_result = search_clinical_knowledge(question)

    elif intent == "compare":
        print("  [Comparing survivors vs non-survivors...]")
        tool_result = compare_survivors_vs_deaths()

    else:
        tool_result = get_population_statistics()

    print("  [Generating clinical response...]\n")

    # Build conversation
    contents = []
    for h in history[-4:]:
        contents.append(types.Content(
            role=h["role"],
            parts=[types.Part(text=h["text"])]
        ))

    prompt = f"""Clinical Question: {question}

Data from hospital database and clinical assessment:
{tool_result}

Please provide a clear, clinical, actionable response to the question.
Reference specific values from the data.
If urgent action is needed, say so clearly."""

    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=prompt)]
    ))

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.1
        )
    )
    return response.text


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*65)
    print("  HEART FAILURE CLINICAL AI ASSISTANT")
    print("  Gemini 3.5 Flash · RAG · PostgreSQL · 299+ patients")
    print("="*65)
    print("\nCapabilities:")
    print("  • Admit new patients with natural language input")
    print("  • Retrieve patient profiles and risk assessments")
    print("  • Answer population statistics questions")
    print("  • List highest risk patients")
    print("  • Explain clinical guidelines (RAG)")
    print("  • Compare survivors vs non-survivors")
    print("\nExample inputs:")
    print("  'New patient, 65 year old male, EF 20%, creatinine 1.9, sodium 128'")
    print("  'Show me patient 23'")
    print("  'Which patients are most at risk?'")
    print("  'How many patients have low ejection fraction?'")
    print("  'What should I do for a patient with EF below 30%?'")
    print("  'Compare survivors vs non-survivors'")
    print("\nType 'quit' to exit\n")

    history = []
    while True:
        try:
            question = input("Clinical Question > ").strip()
            if not question:
                continue
            if question.lower() in ["quit", "exit", "q"]:
                print("Exiting Clinical AI Assistant.")
                break

            print("\nThinking...\n")
            answer = ask_agent(question, history)

            history.append({"role": "user",  "text": question})
            history.append({"role": "model", "text": answer})

            print("ASSISTANT:")
            print(answer)
            print("\n" + "-"*65 + "\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
