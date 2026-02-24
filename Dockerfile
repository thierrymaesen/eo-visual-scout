# ---------------------------------------------------------------------------
# EO Visual Scout — Dockerfile for Hugging Face Spaces (Docker Space)
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Non-root user (required by HF Spaces, UID 1000) ---
RUN useradd -m -u 1000 appuser

# --- Install Poetry ---
ENV POETRY_VERSION=1.8.4
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# --- Working directory ---
WORKDIR /app

# --- Copy dependency files first (Docker cache layer) ---
COPY pyproject.toml poetry.lock ./

# --- Install dependencies (no dev, no virtualenv inside container) ---
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi

# --- Copy application code ---
COPY src/ src/
COPY app/ app/
COPY data/ data/

# --- Fix permissions ---
RUN chown -R appuser:appuser /app

# --- Switch to non-root user ---
USER appuser

# --- Expose ports (FastAPI 8000, Gradio 7860) ---
EXPOSE 7860 8000

# --- Launch: FastAPI backend + Gradio frontend ---
CMD ["bash", "-c", "uvicorn eovs.api:app --host 0.0.0.0 --port 8000 & sleep 3 && python app/app.py"]
