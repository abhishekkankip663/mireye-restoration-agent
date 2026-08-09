"""
Live API + minimal UI for the restoration-funding prioritization agent.

The reasoning lives entirely in agent.py -- this file only exposes it
over HTTP (POST /prioritize) and serves a thin frontend (frontend/) that
calls that same endpoint. The UI has no logic of its own: it collects
parcel coordinates, calls /prioritize, and renders the agent's own tool
trace and recommendation. All reasoning/deciding/acting happens in the
agent, not in this file or the page.
"""

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_prioritization_agent

app = FastAPI(title="Mireye Restoration-Funding Prioritization Agent")


class Candidate(BaseModel):
    name: str
    lat: float
    lng: float


class PrioritizeRequest(BaseModel):
    candidates: list[Candidate]
    goal: str | None = None
    model: str | None = None


@app.post("/prioritize")
def prioritize(body: PrioritizeRequest = Body(...)):
    try:
        candidates = [c.model_dump() for c in body.candidates]
        return run_prioritization_agent(candidates, body.goal, body.model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api")
def root():
    return {
        "name": "Mireye Restoration-Funding Prioritization Agent",
        "usage": "POST /prioritize with {candidates: [{name, lat, lng}], goal?: str}",
    }


frontend_dir = Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
