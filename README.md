# 🛰️ EO Visual Scout — Semantic Satellite Image Search

[![CI](https://github.com/thierrymaesen/eo-visual-scout/actions/workflows/ci.yml/badge.svg)](https://github.com/thierrymaesen/eo-visual-scout/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/thierrymaesen/eo-visual-scout)
[![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?logo=docker&logoColor=white)](https://huggingface.co/spaces/thierrymaesen/eo-visual-scout)

> **[Live Demo / Démo en ligne](https://huggingface.co/spaces/thierrymaesen/eo-visual-scout)** — Try it now on Hugging Face Spaces! / Essayez-le maintenant sur Hugging Face Spaces !

---

## 🌍 What is EO Visual Scout?

An **AI-powered search engine** for Earth Observation imagery (EuroSAT).
Type *"a river in a forest"* or **upload your own satellite photo**, and let
the AI find visually similar areas instantly using OpenAI’s CLIP model.

### Key Engineering Features

- 🧠 **Multilingual Semantic Search** — Uses `clip-ViT-B-32-multilingual-v1`
  to encode text and images into **512-dimension vectors** and perform
  blazing-fast cosine similarity against **27,000 satellite images**.
- 📸 **Killer Feature — Image-to-Image** — Upload any satellite image to
  find similar patterns globally, no text required.
- 🛡️ **Production-Ready Architecture** — Clean Python, 100% mocked
  Pytest coverage, and a GitHub Actions CI pipeline cached to run in
  under 3 minutes.
- ⚡ **Full-Stack** — FastAPI REST backend + modern Gradio UI.

### Architecture

```text
eo-visual-scout/
├── app/app.py              # Gradio frontend (dark-themed UI)
├── src/eovs/
│   ├── ingest.py          # EuroSAT dataset downloader
│   ├── build_embeddings.py # CLIP vector builder (512-d)
│   ├── search.py          # Semantic search engine
│   └── api.py             # FastAPI REST backend
├── tests/                   # Pytest unit tests (mocked)
├── .github/workflows/ci.yml # CI pipeline (< 3 min)
├── Dockerfile               # HF Spaces deployment
└── pyproject.toml           # Poetry config
```

---

## 🚀 Installation & Usage

### 1. Clone & install

```bash
git clone https://github.com/thierrymaesen/eo-visual-scout.git
cd eo-visual-scout
poetry install
```

### 2. Download data & build embeddings

```bash
poetry run python -m eovs.ingest --verbose
poetry run python -m eovs.build_embeddings --verbose
```

### 3. Launch the application

Open **two terminals**:

```bash
# Terminal 1 — FastAPI backend
poetry run uvicorn eovs.api:app --host 0.0.0.0 --port 8000
```

```bash
# Terminal 2 — Gradio frontend
poetry run python app/app.py
```

Then open **http://localhost:7860** in your browser.

### 4. Dev commands

```bash
poetry run pytest tests/ -v      # run tests
poetry run ruff check src/ tests/ # lint
poetry run black src/ tests/ app/ # format
```

---

## 🇫🇷 Version française


### ☁️ Cloud Deployment

This application is deployed on **Hugging Face Spaces** using Docker.

👉 **[Try the live app here](https://huggingface.co/spaces/thierrymaesen/eo-visual-scout)**

The Space runs a Docker container with:
- **FastAPI** backend (port 8000) serving the semantic search API
- - **Gradio** frontend (port 7860) providing the web interface
  - - **CLIP multilingual model** loaded at startup for text and image encoding
    - - **27,000 EuroSAT satellite images** indexed for instant search
### 🌍 Qu’est-ce que EO Visual Scout ?

**Moteur de recherche par intelligence artificielle** pour l’observation de
la Terre (EuroSAT). Tapez *« un fleuve dans une forêt »* ou **uploadez une
photo satellite**, et l’IA retrouve instantanément les zones similaires
grâce au modèle CLIP d’OpenAI.

### Points clés

- 🧠 **Recherche sémantique multilingue** — Encode textes et images en
  vecteurs de **512 dimensions** et effectue une similarité cosinus
  ultra-rapide sur **27 000 images satellite**.
- 📸 **Killer Feature — Image-to-Image** — Uploadez n’importe quelle image
  satellite pour trouver des motifs similaires, aucun texte requis.
- 🛡️ **Architecture d’ingénieur senior** — Code Python propre, couverture
  Pytest 100% mockée, pipeline CI GitHub Actions en moins de 3 minutes.
- ⚡ **Full-Stack** — Backend REST FastAPI + interface Gradio moderne.

### Installation rapide

```bash
git clone https://github.com/thierrymaesen/eo-visual-scout.git
cd eo-visual-scout
poetry install
poetry run python -m eovs.ingest --verbose
poetry run python -m eovs.build_embeddings --verbose
```

Lancez ensuite **deux terminaux** :

```bash
# Terminal 1 — Backend FastAPI
poetry run uvicorn eovs.api:app --host 0.0.0.0 --port 8000
```

```bash
# Terminal 2 — Frontend Gradio
poetry run python app/app.py
```

Ouvrez **http://localhost:7860** dans votre navigateur.


### ☁️ Déploiement Cloud

Cette application est déployée sur **Hugging Face Spaces** via Docker.

👉 **[Essayez l'application en ligne ici](https://huggingface.co/spaces/thierrymaesen/eo-visual-scout)**

Le Space exécute un conteneur Docker avec :
- **FastAPI** backend (port 8000) pour l'API de recherche sémantique
- - **Gradio** frontend (port 7860) pour l'interface web
  - - **Modèle CLIP multilingue** chargé au démarrage pour l'encodage texte et image
    - - **27 000 images satellite EuroSAT** indexées pour une recherche instantanée
---

## 📋 Sprint Progress

✅ **Sprint 1 / 10** — EuroSAT dataset ingestion pipeline.
✅ **Sprint 2 / 10** — CLIP image embeddings generation.
✅ **Sprint 3 / 10** — Semantic search engine (SemanticSearcher + CLI).
✅ **Sprint 4 / 10** — FastAPI server (REST API).
✅ **Sprint 5 / 10** — Gradio frontend (semantic image search UI).
✅ **Sprint 6 / 10** — Unit tests & evaluation (Pytest).
✅ **Sprint 7 / 10** — Continuous Integration (GitHub Actions CI pipeline).
✅ **Sprint 8 / 10** — Killer Feature (Image-to-Image Search).
✅ **Sprint 9 / 10** — UI Image-to-Image (Frontend).
✅ **Sprint 10 / 10** — Documentation Bilingue & Préparation Déploiement Cloud.

---

## 📜 License

[MIT](LICENSE) © 2026 Thierry Maesen
