"""Semantic search engine for EO images using CLIP embeddings."""

import json
import logging
import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_DATA_DIR = Path("data/eurosat")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Single search result returned by SemanticSearcher."""

    id: int
    filename: str
    class_name: str
    score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool) -> None:
    """Configure root logger level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class SemanticSearcher:
    """Load CLIP embeddings once, then answer text or image queries
    via cosine similarity."""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.data_dir = Path(data_dir)

        meta_path: Path = self.data_dir / "metadata.json"
        npz_path: Path = self.data_dir / "embeddings.npz"

        # --- guard: required files -----------------------------------
        for path in (meta_path, npz_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

        # --- metadata ------------------------------------------------
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.metadata: List[Dict[str, Any]] = json.load(fh)

        # --- embeddings ----------------------------------------------
        data = np.load(str(npz_path))
        self.image_embeddings: np.ndarray = data["embeddings"]

        # --- CLIP model ----------------------------------------------
        self.model = SentenceTransformer(model_name)

        logger.info(
            "Searcher initialized with %d images",
            len(self.metadata),
        )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str = "",
        image: Optional[Image.Image] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Return the *top_k* images most similar to *query* or *image*.

        Provide either a text query or a PIL image.  If both are
        given the image takes precedence.  Raises ValueError when
        neither is supplied.
        """
        # --- encode --------------------------------------------------
        if image is not None:
                        tmp_path = os.path.join(
                tempfile.gettempdir(), "_eovs_query.jpg"
            )
            image.save(tmp_path)
            query_emb = self.model.encode(
                [tmp_path],
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            os.unlink(tmp_path)
        elif query and query.strip():
            query_emb = self.model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            raise ValueError(
                "Must provide either a text query or an image."
            )

        # --- cosine similarity ---------------------------------------
        cos_scores = util.cos_sim(
            query_emb, self.image_embeddings
        )[0]

        # Convert to numpy if needed (tensor -> cpu -> numpy)
        if hasattr(cos_scores, "cpu"):
            cos_scores_np: np.ndarray = cos_scores.cpu().numpy()
        else:
            cos_scores_np = np.asarray(cos_scores)

        # Top-K indices (descending order)
        top_indices = np.argsort(cos_scores_np)[::-1][:top_k]

        results: List[SearchResult] = []
        for idx in top_indices:
            idx_int = int(idx)
            meta = self.metadata[idx_int]
            results.append(
                SearchResult(
                    id=idx_int,
                    filename=meta["filename"],
                    class_name=meta["class_name"],
                    score=round(
                        float(cos_scores_np[idx_int]), 4
                    ),
                )
            )

        return results


# ---------------------------------------------------------------------------
# CLI entry-point  (text-only for simplicity)
# ---------------------------------------------------------------------------


def main() -> None:
    """Quick command-line test for SemanticSearcher."""
    parser = argparse.ArgumentParser(
        description="Semantic search over EO satellite images",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query (any language supported by CLIP).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the data directory (default: data/eurosat).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    searcher = SemanticSearcher(
        data_dir=args.data_dir,
        model_name=MODEL_NAME,
    )

    results = searcher.search(query=args.query, top_k=args.top_k)

    print(f"\n{'='*60}")
    print(f"  Query : {args.query}")
    print(f"  Top-{args.top_k} results")
    print(f"{'='*60}\n")

    if not results:
        print("  No results found.")
    else:
        for rank, r in enumerate(results, start=1):
            print(
                f"  {rank}. {r.filename}"
                f"  (Class: {r.class_name})"
                f"  -- Score: {r.score}"
            )

    print()


if __name__ == "__main__":
    main()
