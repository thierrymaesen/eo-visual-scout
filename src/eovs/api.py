"""EO Visual Scout — FastAPI server."""

import base64
import time
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
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
        default="",
        description="Text to search for (optional if image_base64 is provided)",
    )
    image_base64: str = Field(
        default="",
        description="Base64 encoded image string (without data URI prefix)",
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
searcher_instance: Optional[SemanticSearcher] = None


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
    version="0.5.0",
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
    """Run a semantic search and return ranked results.

    Accepts either a text ``query``, a ``image_base64`` payload,
    or both (image takes precedence).
    """
    if searcher_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    if not request.query and not request.image_base64:
        raise HTTPException(
            status_code=400,
            detail="Provide either query or image_base64",
        )

    # --- prepare inputs ------------------------------------------
    text_query: str = request.query
    pil_image: Optional[Image.Image] = None

    if request.image_base64:
        try:
            b64_data = (
                request.image_base64.split(",")[-1]
                if "," in request.image_base64
                else request.image_base64
            )
            image_data = base64.b64decode(b64_data)
            pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image data: {exc}",
            ) from exc

    # --- search --------------------------------------------------
    start = time.time()
    raw_results = searcher_instance.search(
        query=text_query,
        image=pil_image,
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

    response_query = "[Image Search]" if pil_image is not None else text_query

    return SearchResponse(
        query=response_query,
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
