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
