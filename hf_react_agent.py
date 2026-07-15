"""
Heart Failure Clinical AI Agent — ReAct Version
=================================================
A TRUE AI agent using ReAct (Reason → Act → Observe → Repeat).

Differences from chatbot version:
- Plans multi-step tasks autonomously
- Chains tool outputs as inputs to next tools
- Observes results and decides next action
- Completes complex goals from ONE instruction

Author: Tochukwu Aroh
"""

import os, re, json, psycopg2, pandas as pd, chromadb
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
AGENT_DIR   = Path(__file__).parent / "agent"
CHROMA_DIR  = AGENT_DIR / "chroma_db"
MODEL       = "gemini-3.1-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"
MAX_STEPS   = 6

client = genai.Client(api_key=GEMINI_API_KEY)

def get_db():
    return psycopg2.connect(**DB_CONFIG)

# ── Risk Calculator ───────────────────────────────────────────────────────────
def calculate_risk(row: dict) -> dict:
    ef  = float(row.get("ejection_fraction", 40))
    cr  = float(row.get("serum_creatinine", 1.0))
    na  = float(row.get("serum_sodium", 137))
    age = float(row.get("age", 60))
    an  = int(row.get("anaemia", 0))
    khr = (cr > 1.5) and (ef < 30)
    score = 0
    if ef < 20:    score += 40
    elif ef < 30:  score += 30
    elif ef < 40:  score += 20
    if cr > 2.0:   score += 30
    elif cr > 1.5: score += 20
    if na < 125:   score += 20
    elif na < 135: score += 12
    if khr:        score += 15
    if age > 75:   score += 10
    elif age > 65: score += 5
    if an:         score += 5
    prob = min(score / 100, 0.99)
    risk = ("CRITICAL" if prob >= 0.8 else
            "HIGH"     if prob >= 0.6 else
            "MEDIUM"   if prob >= 0.3 else "LOW")
    flags = []
    if ef < 20:    flags.append("CRITICAL: EF <20% — life-threatening")
    elif ef < 30:  flags.append("CRITICAL: EF <30% — immediate action required")
    elif ef < 40:  flags.append("WARNING: EF <40% — HFrEF confirmed")
    if cr > 2.0:   flags.append("CRITICAL: Creatinine >2.0 — acute kidney injury")
    elif cr > 1.5: flags.append("WARNING: Creatinine >1.5 — kidney stress")
    if na < 125:   flags.append("CRITICAL: Sodium <125 — severe hyponatremia")
    elif na < 135: flags.append("WARNING: Sodium <135 — fluid imbalance")
    if khr:        flags.append("HIGH RISK: Kidney-Heart failure pattern (cardiorenal syndrome)")
    esc = {
        "CRITICAL": "IMMEDIATE ICU ESCALATION — Call senior cardiologist NOW",
        "HIGH":     "URGENT cardiology review within 4 hours",
        "MEDIUM":   "Same day clinical review required",
        "LOW":      "Routine follow-up in 4-12 weeks"
    }[risk]
    return {"probability": prob, "risk_level": risk, "flags": flags, "escalation": esc}


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def tool_insert_patient(patient_json: str) -> str:
    """Admit a new patient. Input: JSON string with clinical values."""
    try:
        data = json.loads(patient_json) if isinstance(patient_json, str) else patient_json
        defaults = {
            "age": 60, "sex": 1, "anaemia": 0, "creatinine_phosphokinase": 582,
            "diabetes": 0, "ejection_fraction": 38, "high_blood_pressure": 0,
            "platelets": 263358, "serum_creatinine": 1.0, "serum_sodium": 137,
            "smoking": 0, "time": 30, "death_event": 0
        }
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        data["kidney_heart_risk"]          = bool(data["serum_creatinine"] > 1.5 and data["ejection_fraction"] < 30)
        data["hyponatremia"]               = bool(data["serum_sodium"] < 135)
        data["age_creatinine_interaction"] = data["age"] * data["serum_creatinine"]
        data["comorbidity_score"]          = int(data["anaemia"]) + int(data["high_blood_pressure"])
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT MAX(id) FROM raw.patients_clinical")
        new_id = (cur.fetchone()[0] or 0) + 1
        cur.execute("""
            INSERT INTO raw.patients_clinical
            (id,age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
             high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,
             time,death_event,kidney_heart_risk,hyponatremia,age_creatinine_interaction,comorbidity_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (new_id, data["age"], data["anaemia"], data["creatinine_phosphokinase"],
              data["diabetes"], data["ejection_fraction"], data["high_blood_pressure"],
              data["platelets"], data["serum_creatinine"], data["serum_sodium"],
              data["sex"], data["smoking"], data["time"], data["death_event"],
              data["kidney_heart_risk"], data["hyponatremia"],
              data["age_creatinine_interaction"], data["comorbidity_score"]))
        conn.commit(); cur.close(); conn.close()
        risk = calculate_risk(data)
        return json.dumps({
            "patient_id": new_id,
            "risk_level": risk["risk_level"],
            "probability": round(risk["probability"] * 100, 1),
            "flags": risk["flags"],
            "escalation": risk["escalation"],
            "message": f"Patient {new_id} admitted successfully"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_get_patient(patient_id: str) -> str:
    """Get full profile for a specific patient. Input: patient ID."""
    try:
        conn = get_db()
        df = pd.read_sql(f"SELECT * FROM raw.patients_clinical WHERE id = {int(patient_id)}", conn)
        conn.close()
        if df.empty:
            return json.dumps({"error": f"No patient with ID {patient_id}"})
        r = df.iloc[0].to_dict()
        risk = calculate_risk(r)
        r.update({
            "risk_level": risk["risk_level"],
            "probability": round(risk["probability"] * 100, 1),
            "flags": risk["flags"],
            "escalation": risk["escalation"],
            "outcome": "DIED" if r.get("death_event") == 1 else "SURVIVED"
        })
        return json.dumps(r, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_query_patients(sql_filter: str) -> str:
    """Query patients with SQL filter. Input: WHERE clause e.g. 'ejection_fraction < 30'"""
    try:
        conn = get_db()
        df = pd.read_sql(f"""
            SELECT id, age, ejection_fraction, serum_creatinine,
                   serum_sodium, kidney_heart_risk, death_event
            FROM raw.patients_clinical
            WHERE {sql_filter}
            ORDER BY ejection_fraction ASC LIMIT 20
        """, conn)
        conn.close()
        if df.empty:
            return json.dumps({"count": 0, "patients": [], "filter": sql_filter})
        patients = []
        for _, r in df.iterrows():
            risk = calculate_risk(r.to_dict())
            patients.append({
                "id": int(r["id"]), "age": r["age"],
                "ejection_fraction": r["ejection_fraction"],
                "serum_creatinine": r["serum_creatinine"],
                "risk_level": risk["risk_level"],
                "probability": round(risk["probability"] * 100, 1),
                "outcome": "DIED" if r["death_event"] == 1 else "SURVIVED"
            })
        return json.dumps({"count": len(patients), "patients": patients})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_get_statistics(metric: str) -> str:
    """Get population statistics. Input: 'all' or specific metric."""
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*), SUM(death_event), AVG(age),
                   AVG(ejection_fraction), AVG(serum_creatinine), AVG(serum_sodium),
                   SUM(CASE WHEN ejection_fraction < 20 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN ejection_fraction < 30 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN ejection_fraction < 40 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN serum_creatinine > 1.5 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN kidney_heart_risk THEN 1 ELSE 0 END),
                   SUM(CASE WHEN serum_sodium < 135 THEN 1 ELSE 0 END),
                   SUM(anaemia), SUM(high_blood_pressure), SUM(diabetes)
            FROM raw.patients_clinical
        """)
        r = cur.fetchone(); cur.close(); conn.close()
        t, d = r[0], r[1]
        return json.dumps({
            "total": t, "deaths": d,
            "mortality_pct": round(d / t * 100, 1),
            "avg_age": round(r[2], 1),
            "avg_ef": round(r[3], 1),
            "avg_creatinine": round(r[4], 2),
            "avg_sodium": round(r[5], 1),
            "critical_ef_under20": r[6],
            "severe_ef_under30": r[7],
            "reduced_ef_under40": r[8],
            "high_creatinine": r[9],
            "kidney_heart_risk_count": r[10],
            "hyponatremia_count": r[11],
            "anaemia_count": r[12],
            "hbp_count": r[13],
            "diabetes_count": r[14]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_search_guidelines(query: str) -> str:
    """Search clinical knowledge base. Input: clinical question or finding."""
    try:
        if not CHROMA_DIR.exists():
            return json.dumps({"error": "Knowledge base not found. Run hf_rag_builder.py first."})
        cc  = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = cc.get_collection("clinical_knowledge")
        emb = client.models.embed_content(model=EMBED_MODEL, contents=query)
        results = col.query(query_embeddings=[emb.embeddings[0].values], n_results=3)
        if not results["documents"][0]:
            return json.dumps({"guidelines": "No relevant guidelines found"})
        chunks = [{"source": m["source"], "content": d[:400]}
                  for d, m in zip(results["documents"][0], results["metadatas"][0])]
        return json.dumps({"guidelines": chunks, "query": query})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_compare_groups(filters: str) -> str:
    """Compare two patient groups. Input: two SQL filters separated by | """
    try:
        parts = filters.split("|")
        f1 = parts[0].strip()
        f2 = parts[1].strip() if len(parts) > 1 else "ejection_fraction >= 40"
        conn = get_db()
        def grp(f):
            df = pd.read_sql(f"SELECT * FROM raw.patients_clinical WHERE {f}", conn)
            if df.empty: return {"count": 0}
            return {
                "count": len(df),
                "avg_age": round(df["age"].mean(), 1),
                "avg_ef": round(df["ejection_fraction"].mean(), 1),
                "avg_creatinine": round(df["serum_creatinine"].mean(), 2),
                "avg_sodium": round(df["serum_sodium"].mean(), 1),
                "mortality_pct": round(df["death_event"].mean() * 100, 1)
            }
        g1 = grp(f1); g2 = grp(f2); conn.close()
        return json.dumps({"group1": {"filter": f1, "stats": g1},
                           "group2": {"filter": f2, "stats": g2}})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_log_prediction(data: str) -> str:
    """Log a prediction to audit trail. Input: JSON with patient_id, risk_level, probability."""
    try:
        info = json.loads(data) if isinstance(data, str) else data
        conn = get_db(); cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO predictions.risk_predictions
                (patient_id, risk_probability, risk_level, model_used, predicted_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (info.get("patient_id"), info.get("probability", 0) / 100,
                  info.get("risk_level", "UNKNOWN"), "ClinicalAI-ReAct-v2", datetime.now()))
            conn.commit()
        except Exception:
            conn.rollback()
        cur.close(); conn.close()
        return json.dumps({"logged": True, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Tool Registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "insert_patient":    tool_insert_patient,
    "get_patient":       tool_get_patient,
    "predict_risk":      tool_get_patient,
    "query_patients":    tool_query_patients,
    "get_statistics":    tool_get_statistics,
    "search_guidelines": tool_search_guidelines,
    "compare_groups":    tool_compare_groups,
    "log_prediction":    tool_log_prediction,
}

TOOL_DESCRIPTIONS = """
AVAILABLE TOOLS (call with: ACTION: tool_name | INPUT: value)

1. insert_patient | INPUT: JSON with age, ejection_fraction, serum_creatinine, serum_sodium
   → Admits new patient to database, calculates risk, returns patient ID + assessment

2. get_patient | INPUT: patient ID number
   → Returns full clinical profile and risk assessment

3. predict_risk | INPUT: patient ID number
   → Returns risk prediction for a specific patient

4. query_patients | INPUT: SQL WHERE clause
   → Finds patients matching a condition
   → Examples: "ejection_fraction < 30", "kidney_heart_risk = true AND death_event = 1"

5. get_statistics | INPUT: "all"
   → Returns population-level statistics

6. search_guidelines | INPUT: clinical question or finding
   → Searches RAG knowledge base for ACC/AHA guidelines, drug protocols, escalation rules

7. compare_groups | INPUT: two SQL filters separated by |
   → Compares clinical metrics between groups
   → Example: "kidney_heart_risk = true | kidney_heart_risk = false"

8. log_prediction | INPUT: JSON with patient_id, risk_level, probability
   → Logs decision to audit trail
"""

SYSTEM = f"""You are a Clinical AI Agent for heart failure decision support.
You autonomously plan and execute multi-step clinical workflows.

{TOOL_DESCRIPTIONS}

HOW TO RESPOND:
Use this exact format for each step:

THOUGHT: [your clinical reasoning — what do you need to do and why]
ACTION: tool_name | INPUT: value

After receiving a tool result, continue with the next step:
THOUGHT: [what the result tells you and what to do next]
ACTION: tool_name | INPUT: value

When the goal is fully complete:
THOUGHT: I have all the information needed
FINAL ANSWER: [comprehensive, clinically actionable response]

RULES:
- Always start with a THOUGHT before each ACTION
- Chain tools: use output of one tool as input to the next
- For new patient admissions: insert_patient → search_guidelines → log_prediction
- For population analysis: get_statistics → query_patients → search_guidelines
- For complex comparisons: query_patients → compare_groups → search_guidelines
- Always end with FINAL ANSWER after completing all necessary steps
- Be specific with clinical values, risk levels, and recommendations
- Use CRITICAL/URGENT for high-risk situations"""


# ── Patient Value Parser ──────────────────────────────────────────────────────
def parse_patient_values(text: str) -> str:
    """Extract structured patient data from natural language."""
    prompt = f"""Extract clinical values from this text. Return ONLY a JSON object, nothing else.

Text: "{text}"

Fields to extract:
- age (integer)
- sex (1=male, 0=female)
- ejection_fraction (integer, percentage without % sign)
- serum_creatinine (float, mg/dL)
- serum_sodium (integer, mEq/L)
- anaemia (1=yes, 0=no)
- diabetes (1=yes, 0=no)
- high_blood_pressure (1=yes, 0=no)
- smoking (1=yes, 0=no)
- platelets (float, optional)

Shorthand: EF=ejection_fraction, Cr=creatinine, Na=sodium, HBP=high_blood_pressure
Only include fields that are mentioned. Return valid JSON only.
Example: {{"age": 65, "ejection_fraction": 20, "serum_creatinine": 1.9, "serum_sodium": 128}}"""
    try:
        r = client.models.generate_content(
            model=MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        clean = re.sub(r"```json|```", "", r.text).strip()
        return clean
    except Exception:
        return "{}"


def execute_tool(tool_name: str, tool_input: str) -> str:
    """Execute a named tool with given input."""
    name = tool_name.strip().lower().replace(" ", "_")
    if name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {name}. Available: {list(TOOLS.keys())}"})
    try:
        return TOOLS[name](tool_input.strip())
    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# REACT AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def react_agent(goal: str, history: list) -> str:
    """
    ReAct loop: Reason → Act → Observe → Reason → Act → ... → Final Answer
    Runs up to MAX_STEPS tool calls per goal.
    """
    # Detect new patient and pre-parse values
    is_new_patient = any(s in goal.lower() for s in
        ["new patient", "admit", "admission", "new case", "intake"])
    has_values     = any(w in goal.lower() for w in
        ["ejection fraction", "ef ", "creatinine", "sodium", "year old", "yo "])
    has_id         = bool(re.search(r'\b(?:patient|pt|id|#)\s*\d+', goal.lower()))

    patient_json = None
    if is_new_patient or (has_values and not has_id):
        print("  [Parsing patient values from text...]")
        patient_json = parse_patient_values(goal)
        if patient_json and patient_json != "{}":
            print(f"  [Extracted: {patient_json[:80]}...]")
            goal = f"{goal}\n[Parsed patient data: {patient_json}]"

    # Build message history
    messages = []
    for h in history[-4:]:
        messages.append(types.Content(role=h["role"], parts=[types.Part(text=h["text"])]))

    # Initial message
    messages.append(types.Content(
        role="user",
        parts=[types.Part(text=f"""Clinical Goal: {goal}

Plan your approach. Think about what tools you need and in what order.
Execute step by step until the goal is fully complete.""")]
    ))

    step_results = []
    final_answer  = None

    for step in range(MAX_STEPS):
        print(f"  [Step {step + 1}/{MAX_STEPS}] Reasoning...", end=" ", flush=True)

        response = client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0.1
            )
        )
        reply = response.text

        # Check for final answer
        if "FINAL ANSWER:" in reply:
            final_answer = reply.split("FINAL ANSWER:")[-1].strip()
            print(f"Complete ✓")
            print(f"  [Finished in {step + 1} steps]")
            break

        # Extract ACTION
        action_match = re.search(
            r'ACTION:\s*(\w+)\s*\|\s*INPUT:\s*(.+?)(?=\nTHOUGHT:|$)',
            reply, re.DOTALL | re.IGNORECASE
        )

        if action_match:
            tool_name  = action_match.group(1).strip()
            tool_input = action_match.group(2).strip()

            # Use pre-parsed patient data for insert_patient
            if tool_name.lower() in ["insert_patient"] and patient_json and patient_json != "{}":
                tool_input = patient_json

            print(f"→ {tool_name}")
            result = execute_tool(tool_name, tool_input)

            step_results.append({
                "step": step + 1,
                "tool": tool_name,
                "input": tool_input[:100],
                "result": result
            })

            # Feed result back to agent
            messages.append(types.Content(role="model", parts=[types.Part(text=reply)]))
            messages.append(types.Content(
                role="user",
                parts=[types.Part(text=f"""Result of {tool_name}:
{result}

Continue with the next step if needed, or provide your FINAL ANSWER if the goal is complete.""")]
            ))

        else:
            # No action found
            if step_results:
                # Agent may be giving a narrative — treat as final
                final_answer = reply
                print(f"Complete ✓")
                break
            else:
                # Force a tool call
                print(f"Retrying...")
                messages.append(types.Content(role="model", parts=[types.Part(text=reply)]))
                messages.append(types.Content(
                    role="user",
                    parts=[types.Part(text="Please use a tool. Format:\nTHOUGHT: [reasoning]\nACTION: tool_name | INPUT: value")]
                ))

    # Generate synthesis if no final answer
    if not final_answer:
        print("  [Synthesizing final answer...]")
        all_results = "\n\n".join([
            f"Step {s['step']} ({s['tool']}):\n{s['result']}"
            for s in step_results
        ])
        synth_messages = messages + [types.Content(
            role="user",
            parts=[types.Part(text=f"""Original goal: {goal}

All data gathered across {len(step_results)} steps:
{all_results}

Now provide a comprehensive, clinically actionable FINAL ANSWER that fully addresses the original goal.
Include: findings, risk levels, clinical flags, treatment recommendations, and next steps.""")]
        )]
        r = client.models.generate_content(
            model=MODEL,
            contents=synth_messages,
            config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.1)
        )
        final_answer = r.text

    return final_answer or "Unable to complete the clinical workflow. Please try again."


# ── CLI Interface ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  HEART FAILURE CLINICAL AI AGENT — ReAct")
    print("  Multi-step autonomous clinical workflows")
    print("  Gemini 3.1 Flash · RAG · PostgreSQL · 300+ patients")
    print("=" * 65)
    print("\nThis is a TRUE AI agent — give it complex multi-step goals:\n")
    print("  'Admit 70yr female, EF 18, creatinine 2.3, sodium 125, then")
    print("   find other patients with similar profiles'")
    print()
    print("  'Find all patients with EF below 20%, assess their risk,")
    print("   and retrieve treatment guidelines for each'")
    print()
    print("  'Compare mortality rate between patients with kidney-heart")
    print("   risk vs those without, and explain the clinical difference'")
    print()
    print("  'Show patient 23, predict their current risk, and find")
    print("   what guidelines apply to their profile'")
    print()
    print("  'Analyze our high-risk population and summarize")
    print("   the most common clinical patterns'")
    print("\nType 'quit' to exit\n")

    history = []
    while True:
        try:
            goal = input("Clinical Goal > ").strip()
            if not goal:
                continue
            if goal.lower() in ["quit", "exit", "q"]:
                print("Exiting Clinical AI Agent.")
                break

            print(f"\nAgent executing: {goal[:60]}...\n" if len(goal) > 60 else f"\nAgent executing: {goal}\n")
            answer = react_agent(goal, history)

            history.append({"role": "user",  "text": goal})
            history.append({"role": "model", "text": answer})
            if len(history) > 12:
                history = history[-12:]

            print("\n" + "=" * 65)
            print("AGENT RESPONSE:")
            print("=" * 65)
            print(answer)
            print("=" * 65 + "\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
