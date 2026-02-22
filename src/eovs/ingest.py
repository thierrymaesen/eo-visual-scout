"""EO Visual Scout — EuroSAT dataset ingestion pipeline."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET_NAME: str = "jonathan-roberts1/EuroSAT"
DEFAULT_DATA_DIR: Path = Path("data/eurosat")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool) -> None:
    """Configure the root logger level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def ingest_eurosat(
    output_dir: Path,
    limit: Optional[int] = None,
    force: bool = False,
) -> None:
    """Download EuroSAT from Hugging Face and save images + metadata.

    Parameters
    ----------
    output_dir:
        Root directory where *images/* and *metadata.json* will be stored.
    limit:
        If set, only ingest the first *limit* images (useful for testing).
    force:
        When True, re-ingest even if *metadata.json* already exists.
    """
    images_dir: Path = output_dir / "images"
    metadata_path: Path = output_dir / "metadata.json"

    # Guard: skip if already ingested ----------------------------------------
    if metadata_path.exists() and not force:
        logger.info("Dataset already ingested at %s", metadata_path)
        return

    # Ensure output directories exist ----------------------------------------
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset from Hugging Face ------------------------------------
        logger.info("Loading dataset '%s' from Hugging Face…", HF_DATASET_NAME)
        ds = load_dataset(HF_DATASET_NAME, split="train")
        class_names: List[str] = ds.features["label"].names
        logger.info("Found %d classes: %s", len(class_names), class_names)

        metadata: List[Dict[str, Any]] = []
        total: int = len(ds) if limit is None else min(limit, len(ds))

        for index, item in enumerate(tqdm(ds, total=total, desc="Ingesting images")):
            if limit is not None and index >= limit:
                break

            image = item["image"]
            label: int = item["label"]
            class_name: str = class_names[label]

            # Convert to RGB and save as JPEG --------------------------------
            image = image.convert("RGB")
            filename: str = f"{class_name}_{index}.jpg"
            image.save(images_dir / filename, format="JPEG")

            metadata.append(
                {
                    "id": index,
                    "filename": filename,
                    "label_int": label,
                    "class_name": class_name,
                }
            )

        # Persist metadata ---------------------------------------------------
        metadata_path.write_text(
            json.dumps(metadata, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Successfully ingested %d images into %s",
            len(metadata),
            output_dir,
        )

    except Exception:
        logger.critical("Failed to ingest dataset '%s'", HF_DATASET_NAME, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and launch ingestion."""
    parser = argparse.ArgumentParser(
        description="Download and prepare the EuroSAT dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory for ingested data (default: data/eurosat)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images for testing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if data already exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    ingest_eurosat(
        output_dir=args.data_dir,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
