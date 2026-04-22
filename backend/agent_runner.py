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
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=_system_prompt(autobiography),
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
            system = _system_prompt(autobiographies[pid]) + (
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


def _system_prompt(autobiography: str) -> str:
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
