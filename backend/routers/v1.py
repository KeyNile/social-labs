import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from schemas import (
    QueryRequest, QueryResponse, PersonaSummary,
    SimulateRequest, SimulateResponse, AgentResponse,
    TrajectoryResponse, TimelineEntry,
)
from persona_retriever import retrieve
from agent_runner import run_individual, run_discussion
from narrative_parser import parse_narrative

router = APIRouter(prefix="/v1")
_api_key_header = APIKeyHeader(name="X-API-Key")


def _verify_key(key: str = Security(_api_key_header)) -> str:
    valid = os.getenv("API_KEYS", "dev-key").split(",")
    if key.strip() not in valid:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


@router.post("/query", response_model=QueryResponse)
async def query_personas(req: QueryRequest, request: Request, _=Depends(_verify_key)):
    filters = req.filters.model_dump(exclude_none=True) if req.filters else {}
    personas = retrieve(
        context=req.context,
        filters=filters,
        n=req.n,
        engine=request.app.state.engine,
        narrative_dir=request.app.state.narrative_dir,
    )
    return QueryResponse(personas=[
        PersonaSummary(id=p.id, match_score=p.match_score,
                       summary=p.summary, key_events=p.key_events)
        for p in personas
    ])


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest, request: Request, _=Depends(_verify_key)):
    narrative_dir = request.app.state.narrative_dir
    client = request.app.state.anthropic_client

    if req.mode == "individual":
        raw = run_individual(req.persona_ids, req.question, narrative_dir, client)
        return SimulateResponse(responses=[AgentResponse(**r) for r in raw])

    raw_rounds = run_discussion(
        req.persona_ids, req.question, req.rounds or 3, narrative_dir, client
    )
    return SimulateResponse(
        responses=[],
        rounds=[[AgentResponse(**r) for r in round_] for round_ in raw_rounds],
    )


@router.get("/personas/{persona_id}/trajectory", response_model=TrajectoryResponse)
async def get_trajectory(persona_id: str, request: Request, _=Depends(_verify_key)):
    narrative_dir = request.app.state.narrative_dir
    matches = [p for p in Path(narrative_dir).glob(f"{persona_id}_*.md")
               if "neutral" not in p.name]
    if not matches:
        raise HTTPException(status_code=404, detail="Persona not found")

    meta = parse_narrative(matches[0])["metadata"]
    years = sorted(
        set(meta.get("income", {}).keys()) | set(meta.get("hgc_ever", {}).keys())
    )
    timeline = [
        TimelineEntry(
            year=int(y),
            education=meta.get("hgc_ever", {}).get(y),
            income=meta.get("income", {}).get(y),
            events=_year_events(meta, y),
        )
        for y in years
    ]
    return TrajectoryResponse(
        id=persona_id,
        background={"sex": meta.get("sex"), "birth_year": meta.get("birth_year")},
        timeline=timeline,
    )


def _year_events(meta: dict, year: str) -> list[str]:
    events = []
    if m := meta.get("marital", {}).get(year):
        events.append(m)
    if c := meta.get("children", {}).get(year):
        events.append(f"children: {c}")
    return events
