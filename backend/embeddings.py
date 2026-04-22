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
