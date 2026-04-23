from pydantic import BaseModel, Field
from typing import Optional


class FilterParams(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    education: Optional[str] = None


class QueryRequest(BaseModel):
    context: str
    filters: Optional[FilterParams] = None
    n: int = Field(default=5, ge=1, le=20)


class PersonaSummary(BaseModel):
    id: str
    match_score: float
    summary: str
    key_events: list[str]


class QueryResponse(BaseModel):
    personas: list[PersonaSummary]


class SimulateRequest(BaseModel):
    persona_ids: list[str]
    question: str
    mode: str = Field(default="individual", pattern="^(individual|discussion)$")
    rounds: Optional[int] = Field(default=3, ge=1, le=10)


class AgentResponse(BaseModel):
    id: str
    response: str
    round: Optional[int] = None


class SimulateResponse(BaseModel):
    responses: list[AgentResponse]
    rounds: Optional[list[list[AgentResponse]]] = None


class TimelineEntry(BaseModel):
    year: int
    education: Optional[str] = None
    income: Optional[float] = None
    events: list[str] = []


class TrajectoryResponse(BaseModel):
    id: str
    background: dict
    timeline: list[TimelineEntry]
