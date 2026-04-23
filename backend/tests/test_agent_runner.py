from pathlib import Path
from unittest.mock import MagicMock
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
    assert len(results) == 2
    assert len(results[0]) == 2


def test_discussion_includes_round_number():
    client = _mock_client("response")
    results = run_discussion(["ID_001"], "question", rounds=1,
                             narrative_dir=NARRATIVE_DIR, client=client)
    assert results[0][0]["round"] == 1
