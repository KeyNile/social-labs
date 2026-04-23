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

    parts = md_path.stem.split("_")
    persona_id = f"{parts[0]}_{parts[1]}"

    details_start = content.find("<details>")
    autobiography = content[:details_start].strip() if details_start != -1 else content.strip()

    json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    metadata = json.loads(json_match.group(1)) if json_match else {}

    return {"persona_id": persona_id, "autobiography": autobiography, "metadata": metadata}
