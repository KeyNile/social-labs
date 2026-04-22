from dataclasses import dataclass, field
from pathlib import Path
from narrative_parser import parse_narrative

CURRENT_YEAR = 2026


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
