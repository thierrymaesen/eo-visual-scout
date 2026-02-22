"""Build CLIP image embeddings for the EuroSAT dataset.

Sprint 2/10 – eo-visual-scout
Loads the multilingual CLIP model, encodes every image referenced in
metadata.json and persists the resulting matrix as a compressed .npz file.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────

MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_DATA_DIR = Path("data/eurosat")
BATCH_SIZE = 32

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def setup_logging(verbose: bool) -> None:
    """Configure root logger level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_metadata(
    metadata_path: Path,
) -> List[Dict[str, Any]]:
    """Read *metadata.json* and return the list of records.

    Raises
    ------
    FileNotFoundError
        If the metadata file does not exist.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        data: List[Dict[str, Any]] = json.load(fh)
    logger.info("Loaded %d records from %s", len(data), metadata_path)
    return data


# ── Core ───────────────────────────────────────────────────────────────


def build_image_embeddings(
    data_dir: Path,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> None:
    """Encode every image into a 512-d CLIP vector and save to .npz.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing *metadata.json* and *images/*.
    batch_size : int
        Number of images per encoding batch.
    force : bool
        When *True*, overwrite an existing embeddings file.
    """
    metadata_path = data_dir / "metadata.json"
    images_dir = data_dir / "images"
    output_npz = data_dir / "embeddings.npz"

    # Skip if already computed
    if output_npz.exists() and not force:
        logger.info("Embeddings already exist. Use --force.")
        return

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Validate images directory
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Load model
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded. Starting encoding...")

    all_embeddings: List[np.ndarray] = []
    total_batches = (len(metadata) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(metadata), batch_size),
        total=total_batches,
        desc="Encoding",
    ):
        batch_meta = metadata[start : start + batch_size]
        batch_images: List[Image.Image] = []

        for item in batch_meta:
            img_path = images_dir / item["filename"]
            img = Image.open(img_path).convert("RGB")
            batch_images.append(img)

        batch_emb: np.ndarray = model.encode(
            batch_images,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embeddings.append(batch_emb)

        # Explicitly close images to avoid memory leaks
        for img in batch_images:
            img.close()

    embeddings_matrix = np.vstack(all_embeddings)
    np.savez_compressed(output_npz, embeddings=embeddings_matrix)
    logger.info(
        "Saved matrix of shape %s to %s",
        embeddings_matrix.shape,
        output_npz,
    )


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and launch the embedding pipeline."""
    parser = argparse.ArgumentParser(
        description="Build CLIP image embeddings for EuroSAT.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (default: data/eurosat).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Images per batch (default: 32).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    build_image_embeddings(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
