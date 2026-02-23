"""EO Visual Scout — FastAPI server."""

import time
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .search import SemanticSearcher

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Incoming search payload."""

    query: str = Field(
        ...,
        min_length=2,
        description="Text to search for",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return",
    )


class SearchResultItem(BaseModel):
    """Single result returned by the API."""

    id: int
    filename: str
    class_name: str
    score: float


class SearchResponse(BaseModel):
    """Full search response envelope."""

    query: str
    results: List[SearchResultItem]
    latency_ms: float


# ---------------------------------------------------------------------------
# Global state & lifespan
# ---------------------------------------------------------------------------
searcher_instance: SemanticSearcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the heavy CLIP model once at startup, release on shutdown."""
    global searcher_instance
    try:
        logger.info("Loading Semantic Searcher...")
        searcher_instance = SemanticSearcher()
        logger.info("Semantic Searcher ready.")
    except FileNotFoundError as exc:
        logger.error(
            "Cannot start: required data files are missing. "
            "Run ingest + build_embeddings first. (%s)",
            exc,
        )
    yield
    searcher_instance = None
    logger.info("Searcher released.")


# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EO Visual Scout API",
    description="Multilingual semantic search over EuroSAT imagery",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Lightweight liveness / readiness probe."""
    return {
        "status": "ok",
        "model_loaded": searcher_instance is not None,
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Run a semantic search and return ranked results."""
    if searcher_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    start = time.time()
    raw_results = searcher_instance.search(
        request.query,
        top_k=request.top_k,
    )
    latency_ms = round((time.time() - start) * 1000, 2)

    items = [
        SearchResultItem(
            id=r.id,
            filename=r.filename,
            class_name=r.class_name,
            score=r.score,
        )
        for r in raw_results
    ]

    return SearchResponse(
        query=request.query,
        results=items,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "eovs.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
