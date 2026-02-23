# 🛰️ EO Visual Scout

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Multilingual semantic search engine for Earth Observation imagery (EuroSAT) using CLIP.

## Overview

**EO Visual Scout** lets you search satellite images by natural-language queries in
French and English. It encodes EuroSAT images with
`sentence-transformers/clip-ViT-B-32-multilingual-v1` and retrieves the most
relevant tiles via cosine similarity.

## Installation

```bash
git clone https://github.com/thierrymaesen/eo-visual-scout.git
cd eo-visual-scout
poetry install
```

## Usage

```bash
# Ingest EuroSAT dataset (Sprint 1)
python -m eovs.ingest --verbose
python -m eovs.ingest --limit 100 --verbose # quick test with 100 images

# Build CLIP image embeddings (Sprint 2)
python -m eovs.build_embeddings --verbose
python -m eovs.build_embeddings --force -v # re-generate embeddings

# Semantic search (Sprint 3)
python -m eovs.search --query "a river in a forest" --top-k 5
python -m eovs.search --query "zone industrielle" --top-k 3 --verbose

# Dev commands
poetry run pytest tests/ -v # run tests
poetry run ruff check src/ tests/ # lint
poetry run black src/ tests/ app/ # format
```

## Status

✅ **Sprint 1 / 10 — Completed** — EuroSAT datashet ingestion pipeline.
✅ **Sprint 2 / 10 — Completed** — CLIP image embeddings generation.
✅ **Sprint 3 / 10 — Completed** — Semantic search engine (SemanticSearcher + CLI).
✅ **Sprint 4 / 10 — Completed** — FastAPI server (REST API).
✅ **Sprint 5 / 10 — Completed** — Gradio frontend (semantic image search UI).

## License

[MIT](LICENSE) © 2026 Thierry Maesen
