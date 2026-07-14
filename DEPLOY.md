# Deploy the Clinical AI Agent (Public Demo)

Host the Heart Failure Clinical AI Assistant so anyone can open a link and chat with it.

## What gets deployed

- **Web UI** — `frontend/index.html` (chat interface)
- **API** — `hf_agent_api.py` (FastAPI)
- **Agent** — Gemini 3.5 Flash + clinical tools + RAG
- **Data** — 299 patients from CSV (no PostgreSQL required in cloud)

## Option A — Render.com (recommended, free tier)

### 1. Push code to GitHub

Make sure these are committed:

- `hf_agent_Dockerfile`
- `hf_agent_api.py`, `hf_clinical_agent.py`, `hf_clinical_data.py`
- `frontend/`, `agent/` (knowledge + chroma_db)
- `heart_failure_clinical_records_dataset (1).csv`
- `render.yaml`

Do **not** commit `.env` (already gitignored).

### 2. Create Render account

1. Go to [https://render.com](https://render.com)
2. Sign up with GitHub
3. Click **New +** → **Blueprint**
4. Connect repo: `Aroh-Tochii/heart-failure-prediction`
5. Render reads `render.yaml` automatically

### 3. Add your Gemini API key

In the Render dashboard → **heart-failure-agent** → **Environment**:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | Your key from Google AI Studio |
| `DATA_SOURCE` | `csv` |
| `PORT` | `10000` |

Click **Save Changes** — Render redeploys.

### 4. Get your live URL

After deploy (~5 min), your app is at:

```
https://heart-failure-agent.onrender.com
```

(Test the exact URL in your Render dashboard.)

### 5. Add to GitHub profile

1. GitHub → **Settings** → **Profile**
2. Under **Website**, paste your Render URL
3. Or add to README:

```markdown
## Live Demo
Try the Clinical AI Assistant: https://your-app.onrender.com
```

---

## Option B — Run locally (for testing)

```bash
cd "/mnt/c/Users/UserAdmin/Desktop/heart failure prediction"
export DATA_SOURCE=csv
uvicorn hf_agent_api:app --host 0.0.0.0 --port 8004
```

Open: [http://localhost:8004](http://localhost:8004)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `429 RESOURCE_EXHAUSTED` | Gemini quota — create new API key in new Google Cloud project |
| Blank stats panel | Normal on cold start — ask a question in chat |
| Render sleeps after 15 min idle | Free tier — first visit takes ~30s to wake up |
| WSL has no internet | Deploy on Render instead — runs in cloud, not WSL |

---

## Security

- Never commit `.env` or API keys to GitHub
- Rotate keys if exposed in chat or screenshots
- Demo uses read-only CSV data; new patients are in-memory only (reset on redeploy)
