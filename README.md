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
poetry run pytest tests/ -v          # run tests
poetry run ruff check src/ tests/    # lint
poetry run black src/ tests/ app/    # format
```

## Status

🚧 **Sprint 0 / 10 — Completed** — Project scaffolding & configuration.

## License

[MIT](LICENSE) © 2026 Thierry Maesen
