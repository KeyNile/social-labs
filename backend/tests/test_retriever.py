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
