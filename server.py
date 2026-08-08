"""
Minimal live API for the restoration-funding prioritization agent.

This is intentionally thin -- the whole point of this project is the
agent (agent.py), not a web app around it. This just exposes it over
HTTP for anyone who wants to try it live instead of via the CLI.
"""

from fastapi import Body, FastAPI, HTTPException
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


@app.get("/")
def root():
    return {
        "name": "Mireye Restoration-Funding Prioritization Agent",
        "usage": "POST /prioritize with {candidates: [{name, lat, lng}], goal?: str}",
    }
