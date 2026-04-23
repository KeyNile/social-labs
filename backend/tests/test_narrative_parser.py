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
