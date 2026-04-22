import numpy as np
import shutil
from pathlib import Path
import pytest
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
    assert engine.matrix.shape[0] > 0


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
    engine2 = EmbeddingEngine(NARRATIVE_DIR, CACHE_DIR)
    assert np.allclose(engine1.matrix, engine2.matrix)
