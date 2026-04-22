# Persona Agent API MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dual-interface MVP — FastAPI REST API (for B2B clients) + Next.js demo UI (user counseling mode + researcher simulation mode).

**Architecture:** Three FastAPI endpoints (`/v1/query`, `/v1/simulate`, `/v1/personas/{id}/trajectory`) backed by a 2-stage persona retrieval engine (embedding similarity → metadata filter) and a Claude-powered agent runner. Next.js frontend consumes the same API to power both user-facing counseling and researcher simulation views.

**Tech Stack:** FastAPI, sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`), numpy, anthropic SDK, KuzuDB, Next.js 14, Tailwind CSS, recharts.

---

## File Map

```
backend/
├── schemas.py                  CREATE — Pydantic request/response models
├── narrative_parser.py         CREATE — parse autobiography text + JSON from MD files
├── embeddings.py               CREATE — embedding engine (load/generate/search)
├── persona_retriever.py        CREATE — 2-stage retrieval (embedding + metadata filter)
├── agent_runner.py             CREATE — autobiography → Claude API response
├── routers/
│   ├── __init__.py             CREATE — empty
│   └── v1.py                   CREATE — /v1/* FastAPI routes
├── api_server.py               CREATE — FastAPI app entry point
├── tests/
│   ├── __init__.py             CREATE — empty
│   ├── test_narrative_parser.py CREATE
│   ├── test_embeddings.py      CREATE
│   ├── test_retriever.py       CREATE
│   ├── test_agent_runner.py    CREATE
│   └── test_api.py             CREATE

frontend/
├── app/page.tsx                MODIFY — dual-mode UI (user + researcher toggle)
└── components/
    ├── PersonaCard.tsx         CREATE
    ├── TrajectoryChart.tsx     CREATE
    └── SimulationPanel.tsx     CREATE
```

---

## Task 1: Install Dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add sentence-transformers to requirements**

Edit `backend/requirements.txt`, append:
```
sentence-transformers==3.4.1
```

- [ ] **Step 2: Install**

```bash
cd /Users/kaypark/social-labs/backend
source venv/bin/activate
pip install sentence-transformers==3.4.1
```

Expected: Successfully installed sentence-transformers and torch dependencies.

- [ ] **Step 3: Install frontend recharts**

```bash
cd /Users/kaypark/social-labs/frontend
npm install recharts
npm install --save-dev @types/recharts
```

Expected: `recharts` appears in `package.json` dependencies.

- [ ] **Step 4: Commit**

```bash
cd /Users/kaypark/social-labs
git add backend/requirements.txt frontend/package.json frontend/package-lock.json
git commit -m "deps: add sentence-transformers and recharts"
```

---

## Task 2: Pydantic Schemas

**Files:**
- Create: `backend/schemas.py`

- [ ] **Step 1: Write schemas**

Create `backend/schemas.py`:
```python
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
```

- [ ] **Step 2: Verify schemas import without error**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -c "from schemas import QueryRequest, SimulateRequest, TrajectoryResponse; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/kaypark/social-labs
git add backend/schemas.py
git commit -m "feat: add Pydantic schemas for persona API"
```

---

## Task 3: Narrative Parser

**Files:**
- Create: `backend/narrative_parser.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_narrative_parser.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/__init__.py` (empty file).

Create `backend/tests/test_narrative_parser.py`:
```python
from pathlib import Path
from narrative_parser import parse_narrative

SAMPLE = Path(__file__).parent.parent.parent / "output" / "narratives" / "ID_001_Female_Non-Black-Non-Hispanic_1981.md"


def test_parse_returns_persona_id():
    result = parse_narrative(SAMPLE)
    assert result["persona_id"] == "ID_001"


def test_parse_returns_autobiography_text():
    result = parse_narrative(SAMPLE)
    assert len(result["autobiography"]) > 100
    assert "```json" not in result["autobiography"]


def test_parse_returns_metadata_with_birth_year():
    result = parse_narrative(SAMPLE)
    assert result["metadata"]["birth_year"] == 1981


def test_parse_returns_income_data():
    result = parse_narrative(SAMPLE)
    assert isinstance(result["metadata"].get("income", {}), dict)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_narrative_parser.py -v
```

Expected: `ModuleNotFoundError: No module named 'narrative_parser'`

- [ ] **Step 3: Implement narrative_parser.py**

Create `backend/narrative_parser.py`:
```python
import re
import json
from pathlib import Path


def parse_narrative(md_path: Path) -> dict:
    """Parse an autobiography MD file.

    Returns:
        {
            "persona_id": "ID_001",
            "autobiography": "<story text>",
            "metadata": {birth_year, sex, income, hgc_ever, ...}
        }
    """
    content = md_path.read_text(encoding="utf-8")

    # Extract persona_id from filename stem: ID_001_Female_... -> "ID_001"
    parts = md_path.stem.split("_")
    persona_id = f"{parts[0]}_{parts[1]}"

    # Split at <details> block
    details_start = content.find("<details>")
    autobiography = content[:details_start].strip() if details_start != -1 else content.strip()

    # Extract JSON from ```json ... ``` block inside <details>
    json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    metadata = json.loads(json_match.group(1)) if json_match else {}

    return {"persona_id": persona_id, "autobiography": autobiography, "metadata": metadata}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_narrative_parser.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/kaypark/social-labs
git add backend/narrative_parser.py backend/tests/__init__.py backend/tests/test_narrative_parser.py
git commit -m "feat: add narrative_parser to extract autobiography text and metadata"
```

---

## Task 4: Embedding Engine

**Files:**
- Create: `backend/embeddings.py`
- Create: `backend/tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_embeddings.py`:
```python
import numpy as np
import pytest
import shutil
from pathlib import Path
from embeddings import EmbeddingEngine

NARRATIVE_DIR = str(Path(__file__).parent.parent.parent / "output" / "narratives")
CACHE_DIR = "/tmp/test_embed_cache"


@pytest.fixture(autouse=True)
def clean_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def test_engine_builds_matrix():
    engine = EmbeddingEngine(NARRATIVE_DIR, CACHE_DIR)
    assert engine.matrix.ndim == 2
    assert engine.matrix.shape[0] > 0  # at least 1 persona


def test_search_returns_top_k():
    engine = EmbeddingEngine(NARRATIVE_DIR, CACHE_DIR)
    results = engine.search("career change at 40", top_k=5)
    assert len(results) == 5
    assert all(isinstance(pid, str) for pid, _ in results)
    assert all(0.0 <= score <= 1.0 for _, score in results)


def test_search_scores_descending():
    engine = EmbeddingEngine(NARRATIVE_DIR, CACHE_DIR)
    results = engine.search("college education and income", top_k=10)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_cache_reused_on_second_init():
    engine1 = EmbeddingEngine(NARRATIVE_DIR, CACHE_DIR)
    cache_matrix = engine1.matrix.copy()
    engine2 = EmbeddingEngine(NARRATIVE_DIR, CACHE_DIR)
    assert np.allclose(engine1.matrix, engine2.matrix)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_embeddings.py -v
```

Expected: `ModuleNotFoundError: No module named 'embeddings'`

- [ ] **Step 3: Implement embeddings.py**

Create `backend/embeddings.py`:
```python
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from narrative_parser import parse_narrative

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingEngine:
    def __init__(self, narrative_dir: str, cache_dir: str):
        self.model = SentenceTransformer(MODEL_NAME)
        self.matrix, self.persona_ids = self._load_or_generate(narrative_dir, cache_dir)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.persona_ids[i], float(scores[i])) for i in top_indices]

    def _load_or_generate(self, narrative_dir: str, cache_dir: str):
        matrix_path = Path(cache_dir) / "embeddings.npy"
        ids_path = Path(cache_dir) / "persona_ids.json"

        if matrix_path.exists() and ids_path.exists():
            return np.load(matrix_path), json.loads(ids_path.read_text())

        return self._generate(narrative_dir, cache_dir)

    def _generate(self, narrative_dir: str, cache_dir: str):
        # Only process emotional (non-neutral) autobiographies
        paths = sorted(
            p for p in Path(narrative_dir).glob("ID_*.md")
            if "neutral" not in p.name
        )

        persona_ids, texts = [], []
        for path in paths:
            parsed = parse_narrative(path)
            persona_ids.append(parsed["persona_id"])
            texts.append(parsed["autobiography"])

        matrix = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / "embeddings.npy", matrix)
        (Path(cache_dir) / "persona_ids.json").write_text(json.dumps(persona_ids))

        return matrix, persona_ids
```

- [ ] **Step 4: Run tests (first run downloads model ~120MB)**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_embeddings.py -v -s
```

Expected: 4 PASSED (first run takes ~2min for model download)

- [ ] **Step 5: Commit**

```bash
cd /Users/kaypark/social-labs
git add backend/embeddings.py backend/tests/test_embeddings.py
git commit -m "feat: add embedding engine with multilingual sentence-transformers"
```

---

## Task 5: Persona Retriever

**Files:**
- Create: `backend/persona_retriever.py`
- Create: `backend/tests/test_retriever.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_retriever.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock
from persona_retriever import retrieve, PersonaSummary

NARRATIVE_DIR = str(Path(__file__).parent.parent.parent / "output" / "narratives")

def _mock_engine(candidates: list[tuple[str, float]]) -> MagicMock:
    engine = MagicMock()
    engine.search.return_value = candidates
    return engine


def test_retrieve_returns_up_to_n():
    engine = _mock_engine([
        ("ID_001", 0.9), ("ID_002", 0.8), ("ID_003", 0.7),
        ("ID_004", 0.6), ("ID_005", 0.5), ("ID_006", 0.4),
    ])
    results = retrieve(context="career change", filters={}, n=3,
                       engine=engine, narrative_dir=NARRATIVE_DIR)
    assert len(results) <= 3


def test_retrieve_returns_persona_summary_objects():
    engine = _mock_engine([("ID_001", 0.9)])
    results = retrieve(context="test", filters={}, n=5,
                       engine=engine, narrative_dir=NARRATIVE_DIR)
    assert all(isinstance(r, PersonaSummary) for r in results)


def test_age_max_filter_excludes_older_personas():
    # ID_001 born 1981 → age 45 in 2026, should be excluded for age_max=30
    engine = _mock_engine([("ID_001", 0.9)])
    results = retrieve(context="young professional", filters={"age_max": 30}, n=5,
                       engine=engine, narrative_dir=NARRATIVE_DIR)
    assert all(r.id != "ID_001" for r in results)


def test_match_score_preserved():
    engine = _mock_engine([("ID_001", 0.87)])
    results = retrieve(context="test", filters={}, n=5,
                       engine=engine, narrative_dir=NARRATIVE_DIR)
    if results:
        assert abs(results[0].match_score - 0.87) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'persona_retriever'`

- [ ] **Step 3: Implement persona_retriever.py**

Create `backend/persona_retriever.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path
from narrative_parser import parse_narrative

CURRENT_YEAR = 2026

EDUCATION_ORDER = [
    "less than hs", "hs diploma", "ged", "some college",
    "bachelor", "graduate", "professional"
]


@dataclass
class PersonaSummary:
    id: str
    match_score: float
    summary: str
    key_events: list[str] = field(default_factory=list)


def retrieve(
    context: str,
    filters: dict,
    n: int,
    engine,
    narrative_dir: str,
) -> list[PersonaSummary]:
    candidates = engine.search(context, top_k=20)
    results = []
    for persona_id, score in candidates:
        md_path = _find_md(persona_id, narrative_dir)
        if md_path is None:
            continue
        meta = parse_narrative(md_path)["metadata"]
        if not _passes_filter(meta, filters):
            continue
        results.append(PersonaSummary(
            id=persona_id,
            match_score=score,
            summary=_build_summary(meta),
            key_events=_key_events(meta),
        ))
        if len(results) >= n:
            break
    return results


def _find_md(persona_id: str, narrative_dir: str) -> Path | None:
    matches = [p for p in Path(narrative_dir).glob(f"{persona_id}_*.md")
               if "neutral" not in p.name]
    return matches[0] if matches else None


def _passes_filter(meta: dict, filters: dict) -> bool:
    if not filters:
        return True
    birth_year = meta.get("birth_year", 0)
    age = CURRENT_YEAR - birth_year
    if filters.get("age_min") and age < filters["age_min"]:
        return False
    if filters.get("age_max") and age > filters["age_max"]:
        return False
    if filters.get("education"):
        hgc = meta.get("hgc_ever", {})
        highest = list(hgc.values())[-1].lower() if hgc else ""
        if filters["education"].lower() not in highest:
            return False
    return True


def _build_summary(meta: dict) -> str:
    age = CURRENT_YEAR - meta.get("birth_year", 0)
    sex = meta.get("sex", "Unknown")
    hgc = meta.get("hgc_ever", {})
    education = list(hgc.values())[-1] if hgc else "Unknown"
    income = meta.get("income", {})
    max_income = max(income.values()) if income else 0
    return f"{age}세 {sex}, {education}, 최대소득 ${max_income:,}"


def _key_events(meta: dict) -> list[str]:
    events = []
    marital = meta.get("marital", {})
    for year, status in sorted(marital.items()):
        if "Married" in status and "Never" not in status:
            events.append(f"{year} 결혼/이혼")
    hgc = meta.get("hgc_ever", {})
    for year, edu in sorted(hgc.items()):
        if "Bachelor" in edu or "Graduate" in edu:
            events.append(f"{year} {edu}")
    income = meta.get("income", {})
    sorted_years = sorted(income.keys())
    for i in range(1, len(sorted_years)):
        prev, curr = income[sorted_years[i - 1]], income[sorted_years[i]]
        if prev > 0 and curr < prev * 0.5:
            events.append(f"{sorted_years[i]} 소득 급락")
    return events[:5]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_retriever.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/kaypark/social-labs
git add backend/persona_retriever.py backend/tests/test_retriever.py
git commit -m "feat: add 2-stage persona retriever (embedding + metadata filter)"
```

---

## Task 6: Agent Runner

**Files:**
- Create: `backend/agent_runner.py`
- Create: `backend/tests/test_agent_runner.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_agent_runner.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch
from agent_runner import run_individual, run_discussion

NARRATIVE_DIR = str(Path(__file__).parent.parent.parent / "output" / "narratives")


def _mock_client(response_text: str):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = msg
    return client


def test_individual_returns_one_response_per_persona():
    client = _mock_client("This was my experience...")
    results = run_individual(["ID_001", "ID_002"], "biggest challenge?",
                             NARRATIVE_DIR, client)
    assert len(results) == 2
    assert all("id" in r and "response" in r for r in results)


def test_individual_uses_autobiography_in_system_prompt():
    client = _mock_client("test response")
    run_individual(["ID_001"], "test question", NARRATIVE_DIR, client)
    call_kwargs = client.messages.create.call_args[1]
    assert "My Story" in call_kwargs["system"] or "Autobiography" in call_kwargs["system"]


def test_discussion_returns_all_rounds():
    client = _mock_client("round response")
    results = run_discussion(["ID_001", "ID_002"], "discuss challenge",
                             rounds=2, narrative_dir=NARRATIVE_DIR, client=client)
    assert len(results) == 2  # 2 rounds
    assert len(results[0]) == 2  # 2 personas per round


def test_discussion_includes_round_number():
    client = _mock_client("response")
    results = run_discussion(["ID_001"], "question", rounds=1,
                             narrative_dir=NARRATIVE_DIR, client=client)
    assert results[0][0]["round"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_agent_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent_runner'`

- [ ] **Step 3: Implement agent_runner.py**

Create `backend/agent_runner.py`:
```python
from pathlib import Path
from narrative_parser import parse_narrative
import anthropic


def run_individual(
    persona_ids: list[str],
    question: str,
    narrative_dir: str,
    client: anthropic.Anthropic,
) -> list[dict]:
    results = []
    for pid in persona_ids:
        md_path = _find_md(pid, narrative_dir)
        if md_path is None:
            continue
        autobiography = parse_narrative(md_path)["autobiography"]
        system = _build_system_prompt(autobiography)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": question}],
        )
        results.append({"id": pid, "response": message.content[0].text})
    return results


def run_discussion(
    persona_ids: list[str],
    question: str,
    rounds: int,
    narrative_dir: str,
    client: anthropic.Anthropic,
) -> list[list[dict]]:
    autobiographies = {}
    for pid in persona_ids:
        md_path = _find_md(pid, narrative_dir)
        if md_path:
            autobiographies[pid] = parse_narrative(md_path)["autobiography"]

    all_rounds: list[list[dict]] = []
    history: list[dict] = []

    for round_num in range(1, rounds + 1):
        round_responses: list[dict] = []
        for pid in persona_ids:
            if pid not in autobiographies:
                continue
            prior = (
                "\n".join(f"{r['id']}: {r['response']}" for r in history)
                if history else "You are starting the discussion."
            )
            system = _build_system_prompt(autobiographies[pid]) + (
                f"\n\nPrior responses in this discussion:\n{prior}"
            )
            user_msg = question if round_num == 1 else f"Continue the discussion about: {question}"
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            resp = {"id": pid, "response": message.content[0].text, "round": round_num}
            round_responses.append(resp)
            history.append(resp)
        all_rounds.append(round_responses)

    return all_rounds


def _build_system_prompt(autobiography: str) -> str:
    return (
        "You are a real person sharing your lived experience. "
        "Here is your life story:\n\n"
        f"{autobiography}\n\n"
        "Answer authentically in first person, grounded in your actual experiences. "
        "Be specific. Do not make up events not in your story."
    )


def _find_md(persona_id: str, narrative_dir: str) -> Path | None:
    matches = [p for p in Path(narrative_dir).glob(f"{persona_id}_*.md")
               if "neutral" not in p.name]
    return matches[0] if matches else None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_agent_runner.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/kaypark/social-labs
git add backend/agent_runner.py backend/tests/test_agent_runner.py
git commit -m "feat: add agent runner for individual and discussion simulation modes"
```

---

## Task 7: API Router + Server

**Files:**
- Create: `backend/routers/__init__.py`
- Create: `backend/routers/v1.py`
- Create: `backend/api_server.py`
- Create: `backend/tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

Create `backend/tests/test_api.py`:
```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

os.environ["API_KEYS"] = "test-key"
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

NARRATIVE_DIR = str(Path(__file__).parent.parent.parent / "output" / "narratives")


@pytest.fixture(scope="module")
def client():
    # Patch EmbeddingEngine to avoid slow model loading in tests
    mock_engine = MagicMock()
    mock_engine.search.return_value = [("ID_001", 0.9), ("ID_002", 0.8)]

    with patch("api_server.EmbeddingEngine", return_value=mock_engine), \
         patch("api_server.anthropic.Anthropic"):
        from api_server import app
        return TestClient(app)


def test_query_requires_api_key(client):
    resp = client.post("/v1/query", json={"context": "career change", "n": 3})
    assert resp.status_code == 403


def test_query_returns_personas(client):
    resp = client.post(
        "/v1/query",
        json={"context": "career change", "n": 2},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "personas" in data
    assert len(data["personas"]) <= 2


def test_simulate_invalid_mode_rejected(client):
    resp = client.post(
        "/v1/simulate",
        json={"persona_ids": ["ID_001"], "question": "test", "mode": "invalid"},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 422


def test_trajectory_404_for_unknown_persona(client):
    resp = client.get(
        "/v1/personas/ID_999/trajectory",
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 404


def test_trajectory_returns_timeline_for_valid_persona(client):
    resp = client.get(
        "/v1/personas/ID_001/trajectory",
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "timeline" in data
    assert "background" in data
    assert len(data["timeline"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'api_server'`

- [ ] **Step 3: Create router files**

Create `backend/routers/__init__.py` (empty).

Create `backend/routers/v1.py`:
```python
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from schemas import (
    QueryRequest, QueryResponse,
    SimulateRequest, SimulateResponse, AgentResponse,
    TrajectoryResponse, TimelineEntry,
)
from persona_retriever import retrieve, PersonaSummary
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

    parsed = parse_narrative(matches[0])
    meta = parsed["metadata"]

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
    if (m := meta.get("marital", {}).get(year)):
        events.append(m)
    if (c := meta.get("children", {}).get(year)):
        events.append(f"children: {c}")
    return events
```

- [ ] **Step 4: Create api_server.py**

Create `backend/api_server.py`:
```python
import os
import anthropic
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from embeddings import EmbeddingEngine
from routers.v1 import router as v1_router

NARRATIVE_DIR = str(Path(__file__).parent.parent / "output" / "narratives")
CACHE_DIR = str(Path(__file__).parent.parent / "output")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = EmbeddingEngine(NARRATIVE_DIR, CACHE_DIR)
    app.state.narrative_dir = NARRATIVE_DIR
    app.state.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    yield


app = FastAPI(title="Social Labs Persona API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(v1_router)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
python -m pytest tests/test_api.py -v
```

Expected: 5 PASSED

- [ ] **Step 6: Smoke test — start server and hit /docs**

```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
API_KEYS=dev-key ANTHROPIC_API_KEY=<your-key> uvicorn api_server:app --reload --port 8001
```

Open `http://localhost:8001/docs` — verify 3 endpoints are visible.

- [ ] **Step 7: Commit**

```bash
cd /Users/kaypark/social-labs
git add backend/routers/ backend/api_server.py backend/tests/test_api.py
git commit -m "feat: add FastAPI REST API with /v1/query, /v1/simulate, /v1/trajectory endpoints"
```

---

## Task 8: PersonaCard Component

**Files:**
- Create: `frontend/components/PersonaCard.tsx`

- [ ] **Step 1: Create PersonaCard.tsx**

Create `frontend/components/PersonaCard.tsx`:
```tsx
interface PersonaSummary {
  id: string;
  match_score: number;
  summary: string;
  key_events: string[];
}

interface Props {
  persona: PersonaSummary;
  selected: boolean;
  onClick: () => void;
}

export default function PersonaCard({ persona, selected, onClick }: Props) {
  return (
    <div
      onClick={onClick}
      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
        selected
          ? "border-blue-500 bg-blue-50"
          : "border-gray-200 hover:border-gray-300 bg-white"
      }`}
    >
      <div className="flex justify-between items-center mb-1">
        <span className="font-mono text-xs text-gray-400">{persona.id}</span>
        <span className="text-sm font-semibold text-blue-600">
          {(persona.match_score * 100).toFixed(0)}% 매칭
        </span>
      </div>
      <p className="text-sm text-gray-700">{persona.summary}</p>
      {persona.key_events.length > 0 && (
        <ul className="mt-2 space-y-0.5">
          {persona.key_events.map((e, i) => (
            <li key={i} className="text-xs text-gray-500">▸ {e}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd /Users/kaypark/social-labs/frontend
npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
cd /Users/kaypark/social-labs
git add frontend/components/PersonaCard.tsx
git commit -m "feat: add PersonaCard component"
```

---

## Task 9: TrajectoryChart Component

**Files:**
- Create: `frontend/components/TrajectoryChart.tsx`

- [ ] **Step 1: Create TrajectoryChart.tsx**

Create `frontend/components/TrajectoryChart.tsx`:
```tsx
"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";

interface TimelineEntry {
  year: number;
  income?: number;
  education?: string;
  events: string[];
}

interface PersonaTimeline {
  personaId: string;
  data: TimelineEntry[];
}

interface Props {
  timelines: PersonaTimeline[];
}

const COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"];

export default function TrajectoryChart({ timelines }: Props) {
  const allYears = [...new Set(timelines.flatMap(t => t.data.map(d => d.year)))].sort(
    (a, b) => a - b
  );

  const chartData = allYears.map(year => {
    const entry: Record<string, number | string> = { year: String(year) };
    timelines.forEach(({ personaId, data }) => {
      const point = data.find(d => d.year === year);
      if (point?.income != null) {
        entry[personaId] = point.income;
      }
    });
    return entry;
  });

  return (
    <div className="w-full">
      <p className="text-xs text-gray-500 mb-2">소득 궤적 비교 (연도별)</p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="year" tick={{ fontSize: 11 }} />
          <YAxis
            tickFormatter={v => `$${(Number(v) / 1000).toFixed(0)}k`}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(v: number) => [`$${Number(v).toLocaleString()}`, ""]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {timelines.map(({ personaId }, i) => (
            <Line
              key={personaId}
              type="monotone"
              dataKey={personaId}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              connectNulls
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd /Users/kaypark/social-labs/frontend
npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
cd /Users/kaypark/social-labs
git add frontend/components/TrajectoryChart.tsx
git commit -m "feat: add TrajectoryChart component with recharts"
```

---

## Task 10: SimulationPanel (Researcher Mode)

**Files:**
- Create: `frontend/components/SimulationPanel.tsx`

- [ ] **Step 1: Create SimulationPanel.tsx**

Create `frontend/components/SimulationPanel.tsx`:
```tsx
"use client";
import { useState } from "react";

interface AgentResponse {
  id: string;
  response: string;
  round: number;
}

interface Props {
  apiBase: string;
  headers: Record<string, string>;
}

export default function SimulationPanel({ apiBase, headers }: Props) {
  const [personaInput, setPersonaInput] = useState("ID_001, ID_002, ID_003");
  const [question, setQuestion] = useState("");
  const [rounds, setRounds] = useState(3);
  const [allRounds, setAllRounds] = useState<AgentResponse[][]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleStart() {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    const ids = personaInput.split(",").map(s => s.trim()).filter(Boolean);
    try {
      const res = await fetch(`${apiBase}/v1/simulate`, {
        method: "POST",
        headers,
        body: JSON.stringify({ persona_ids: ids, question, mode: "discussion", rounds }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setAllRounds(data.rounds ?? []);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function handleExportJSON() {
    const blob = new Blob([JSON.stringify(allRounds, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "simulation.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleExportCSV() {
    const rows = [["round", "persona_id", "response"]];
    allRounds.forEach((round, ri) =>
      round.forEach(r => rows.push([String(ri + 1), r.id, `"${r.response.replace(/"/g, '""')}"`]))
    );
    const blob = new Blob([rows.map(r => r.join(",")).join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "simulation.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-4 max-w-3xl mx-auto">
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="font-semibold mb-4 text-gray-800">시나리오 설정</h2>
        <div className="space-y-3">
          <div>
            <label className="text-xs text-gray-500">페르소나 ID (쉼표 구분)</label>
            <input
              value={personaInput}
              onChange={e => setPersonaInput(e.target.value)}
              className="w-full mt-1 px-3 py-2 border border-gray-200 rounded-lg text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">질문 / 시나리오</label>
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              rows={3}
              className="w-full mt-1 px-3 py-2 border border-gray-200 rounded-lg text-sm resize-none"
              placeholder="What was the hardest decision in your career?"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">라운드 수: {rounds}</label>
            <input
              type="range" min={1} max={10} value={rounds}
              onChange={e => setRounds(Number(e.target.value))}
              className="w-full mt-1 accent-blue-600"
            />
          </div>
        </div>
        <div className="flex gap-3 mt-5">
          <button
            onClick={handleStart}
            disabled={loading}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm disabled:opacity-50"
          >
            {loading ? "시뮬레이션 중..." : "시작"}
          </button>
          {allRounds.length > 0 && (
            <>
              <button onClick={handleExportJSON}
                className="px-4 py-2 border border-gray-200 rounded-lg text-sm">
                JSON
              </button>
              <button onClick={handleExportCSV}
                className="px-4 py-2 border border-gray-200 rounded-lg text-sm">
                CSV
              </button>
            </>
          )}
        </div>
        {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      </div>

      {allRounds.map((round, ri) => (
        <div key={ri} className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h3 className="text-sm font-semibold text-gray-500 mb-3">Round {ri + 1}</h3>
          <div className="space-y-4">
            {round.map(resp => (
              <div key={resp.id} className="pl-4 border-l-2 border-gray-200">
                <span className="font-mono text-xs text-gray-400">{resp.id}</span>
                <p className="text-sm text-gray-800 mt-1 leading-relaxed">{resp.response}</p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd /Users/kaypark/social-labs/frontend
npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
cd /Users/kaypark/social-labs
git add frontend/components/SimulationPanel.tsx
git commit -m "feat: add SimulationPanel with discussion mode and JSON/CSV export"
```

---

## Task 11: Refactor page.tsx — Dual-Mode UI

**Files:**
- Modify: `frontend/app/page.tsx`
- Create: `frontend/.env.local` (if not exists)

- [ ] **Step 1: Set up environment variables**

Create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_API_KEY=dev-key
```

- [ ] **Step 2: Replace page.tsx with dual-mode UI**

Replace `frontend/app/page.tsx` with:
```tsx
"use client";
import { useState } from "react";
import PersonaCard from "@/components/PersonaCard";
import TrajectoryChart from "@/components/TrajectoryChart";
import SimulationPanel from "@/components/SimulationPanel";

type Mode = "user" | "researcher";

interface PersonaSummary {
  id: string;
  match_score: number;
  summary: string;
  key_events: string[];
}

interface TimelineEntry {
  year: number;
  income?: number;
  education?: string;
  events: string[];
}

interface AgentResponse {
  id: string;
  response: string;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("user");
  const [context, setContext] = useState("");
  const [question, setQuestion] = useState("");
  const [personas, setPersonas] = useState<PersonaSummary[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [timelines, setTimelines] = useState<{ personaId: string; data: TimelineEntry[] }[]>([]);
  const [advice, setAdvice] = useState<AgentResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "dev-key";
  const headers = { "Content-Type": "application/json", "X-API-Key": API_KEY };

  async function handleQuery() {
    if (!context.trim()) return;
    setLoading(true);
    setError("");
    setPersonas([]);
    setSelectedIds([]);
    setTimelines([]);
    setAdvice([]);
    try {
      const res = await fetch(`${API_BASE}/v1/query`, {
        method: "POST",
        headers,
        body: JSON.stringify({ context, n: 5 }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setPersonas(data.personas);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleGetAdvice() {
    if (selectedIds.length === 0) return;
    setLoading(true);
    setError("");
    try {
      const [trajResults, simResult] = await Promise.all([
        Promise.all(
          selectedIds.map(id =>
            fetch(`${API_BASE}/v1/personas/${id}/trajectory`, { headers })
              .then(r => r.json())
              .then(d => ({ personaId: id, data: d.timeline as TimelineEntry[] }))
          )
        ),
        fetch(`${API_BASE}/v1/simulate`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            persona_ids: selectedIds,
            question: question || context,
            mode: "individual",
          }),
        }).then(r => r.json()),
      ]);
      setTimelines(trajResults);
      setAdvice(simResult.responses);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function toggleSelect(id: string) {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id].slice(0, 5)
    );
  }

  return (
    <main className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-100 px-6 py-4 flex justify-between items-center">
        <h1 className="text-xl font-bold text-gray-900">Social Labs</h1>
        <div className="flex gap-1 bg-gray-100 rounded-full p-1">
          {(["user", "researcher"] as Mode[]).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-1.5 rounded-full text-sm transition-colors ${
                mode === m ? "bg-white shadow text-gray-900" : "text-gray-500"
              }`}
            >
              {m === "user" ? "일반" : "연구자"}
            </button>
          ))}
        </div>
      </header>

      <div className="p-6">
        {mode === "user" ? (
          <div className="max-w-5xl mx-auto space-y-6">
            {/* Input */}
            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                상황을 설명해 주세요
              </label>
              <textarea
                value={context}
                onChange={e => setContext(e.target.value)}
                rows={3}
                className="w-full px-4 py-3 border border-gray-200 rounded-lg text-sm resize-none"
                placeholder="저는 38세 마케터입니다. 15년간 일했는데 창업을 고민하고 있어요..."
              />
              <input
                value={question}
                onChange={e => setQuestion(e.target.value)}
                className="w-full mt-2 px-4 py-2 border border-gray-200 rounded-lg text-sm"
                placeholder="구체적인 질문 (선택) — 비워두면 상황 설명이 질문으로 사용됩니다"
              />
              <button
                onClick={handleQuery}
                disabled={loading}
                className="mt-3 px-6 py-2.5 bg-blue-600 text-white rounded-lg text-sm disabled:opacity-50"
              >
                {loading ? "검색 중..." : "유사 경험자 찾기 →"}
              </button>
              {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
            </div>

            {/* Results */}
            {personas.length > 0 && (
              <div className="grid grid-cols-3 gap-6">
                {/* Persona list */}
                <div className="space-y-3">
                  <p className="text-xs text-gray-500 font-medium">
                    유사 페르소나 — 최대 5명 선택
                  </p>
                  {personas.map(p => (
                    <PersonaCard
                      key={p.id}
                      persona={p}
                      selected={selectedIds.includes(p.id)}
                      onClick={() => toggleSelect(p.id)}
                    />
                  ))}
                  {selectedIds.length > 0 && (
                    <button
                      onClick={handleGetAdvice}
                      disabled={loading}
                      className="w-full py-2.5 bg-gray-900 text-white rounded-lg text-sm disabled:opacity-50"
                    >
                      {loading ? "조언 생성 중..." : `${selectedIds.length}명에게 조언 듣기`}
                    </button>
                  )}
                </div>

                {/* Chart + advice */}
                <div className="col-span-2 space-y-4">
                  {timelines.length > 0 && (
                    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
                      <TrajectoryChart timelines={timelines} />
                    </div>
                  )}
                  {advice.length > 0 && (
                    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-4">
                      <h2 className="font-semibold text-gray-800">에이전트 조언</h2>
                      {advice.map(r => (
                        <div key={r.id} className="pl-4 border-l-2 border-blue-200">
                          <span className="font-mono text-xs text-gray-400">{r.id}</span>
                          <p className="text-sm text-gray-800 mt-1 leading-relaxed">{r.response}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          <SimulationPanel apiBase={API_BASE} headers={headers} />
        )}
      </div>
    </main>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /Users/kaypark/social-labs/frontend
npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 4: Start backend and frontend, test end-to-end**

Terminal 1:
```bash
cd /Users/kaypark/social-labs/backend && source venv/bin/activate
API_KEYS=dev-key ANTHROPIC_API_KEY=<your-key> uvicorn api_server:app --reload --port 8001
```

Terminal 2:
```bash
cd /Users/kaypark/social-labs/frontend
npm run dev
```

Open `http://localhost:3000`.

Golden path test:
1. 일반 모드 → "38세 마케터, 창업 고민" 입력 → "유사 경험자 찾기" → 페르소나 카드 5개 확인
2. 페르소나 2-3개 선택 → "조언 듣기" → 궤적 차트 + 조언 텍스트 확인
3. 연구자 모드 전환 → 시나리오 입력 → "시작" → 라운드별 대화 확인
4. JSON/CSV 내보내기 버튼 동작 확인

- [ ] **Step 5: Commit**

```bash
cd /Users/kaypark/social-labs
git add frontend/app/page.tsx frontend/.env.local
git commit -m "feat: refactor page.tsx to dual-mode UI (user counseling + researcher simulation)"
```

---

## Self-Review

| 스펙 요구사항 | 구현 태스크 |
|---|---|
| `POST /v1/query` | Task 7 |
| `POST /v1/simulate` (individual + discussion) | Task 6 + 7 |
| `GET /v1/personas/{id}/trajectory` | Task 7 |
| API Key 인증 | Task 7 (v1.py `_verify_key`) |
| 임베딩 시맨틱 검색 (1단계) | Task 4 |
| 메타데이터 필터 (2단계) | Task 5 |
| 일반 사용자 UI (상담 + 궤적) | Task 8, 9, 11 |
| 연구자 UI (discussion 모드) | Task 10 |
| JSON/CSV 내보내기 | Task 10 |
| discussion 기본 3라운드, 최대 10 | Task 6 (`run_discussion`) + Task 7 (`SimulateRequest`) |
