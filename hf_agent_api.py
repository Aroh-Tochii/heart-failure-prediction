"""
Heart Failure Clinical AI Agent — FastAPI Wrapper
===================================================
Wraps the clinical agent in a REST API so the HTML
frontend can call it via HTTP requests.

Endpoints:
  POST /chat          — send a clinical question, get agent response
  GET  /health        — health check
  GET  /stats         — quick population statistics
  GET  /patients      — list of high risk patients

Run:
  uvicorn hf_agent_api:app --host 0.0.0.0 --port 8004 --reload

Author: Tochukwu Aroh
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Add agent directory to path
AGENT_DIR = Path(__file__).parent / "agent"
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(Path(__file__).parent))

# Import agent functions
from hf_clinical_agent import (
    ask_agent,
    get_population_statistics,
    get_high_risk_patients,
    get_patient_profile,
    route_intent
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Failure Clinical AI Agent",
    description="AI-powered clinical decision support for heart failure patients",
    version="1.0.0"
)

# Allow frontend to call API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend HTML
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── Request/Response models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = []

class ChatResponse(BaseModel):
    response: str
    intent:   str
    status:   str = "success"

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serve the HTML frontend."""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"message": "Heart Failure Clinical AI Agent API", "docs": "/docs"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    from hf_clinical_agent import use_csv_mode
    return {
        "status":  "healthy",
        "service": "Heart Failure Clinical AI Agent",
        "version": "1.0.0",
        "model":   "gemini-3.5-flash",
        "rag":     "ChromaDB — 4 clinical documents",
        "data":    "csv" if use_csv_mode() else "postgresql",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint. Accepts a clinical question and
    returns the agent's response.
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        intent   = route_intent(request.message)
        response = ask_agent(request.message, request.history or [])

        return ChatResponse(
            response=response,
            intent=intent,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Get population statistics for dashboard."""
    try:
        stats = get_population_statistics()
        return {"stats": stats, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/high-risk")
def high_risk(limit: int = 10):
    """Get highest risk patients."""
    try:
        patients = get_high_risk_patients(limit)
        return {"patients": patients, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patient/{patient_id}")
def get_patient(patient_id: int):
    """Get specific patient profile."""
    try:
        profile = get_patient_profile(patient_id)
        return {"profile": profile, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8004"))
    uvicorn.run(
        "hf_agent_api:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
