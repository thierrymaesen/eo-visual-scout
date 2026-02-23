"""EO Visual Scout — Gradio frontend.

Provides a dark-themed web UI that queries the FastAPI backend
(POST /search) and displays matching satellite images in a gallery.

Sprint 5/10 — Gradio Interface.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import requests

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

API_URL = "http://localhost:8000"
IMAGES_DIR = Path("data/eurosat/images")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------


def search_api(
    query: str,
    top_k: int,
) -> Tuple[List[Tuple[str, str]], str]:
    """Call the FastAPI */search* endpoint and return gallery items."""
    if not query or not query.strip():
        return [], "Please enter a search query."

    payload = {"query": query.strip(), "top_k": int(top_k)}
    logger.info("Searching: %s (top_k=%d)", payload["query"], payload["top_k"])

    try:
        response = requests.post(
            f"{API_URL}/search",
            json=payload,
            timeout=10,
        )
    except requests.exceptions.RequestException as exc:
        logger.error("API request failed: %s", exc)
        return (
            [],
            "\u274c Error: API is unreachable. Is the backend running?",
        )

    if response.status_code != 200:
        msg = f"\u274c HTTP {response.status_code}: {response.text[:200]}"
        logger.warning(msg)
        return [], msg

    data: dict = response.json()
    results: list = data.get("results", [])

    gallery_items: List[Tuple[str, str]] = []
    for item in results:
        img_path = IMAGES_DIR / item["filename"]
        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue
        caption = (
            f"[{item['class_name']}] Score: {item['score']:.3f}"
        )
        gallery_items.append((str(img_path), caption))

    latency = data.get("latency_ms", 0)
    status = (
        f"\u2705 Found {len(gallery_items)} results "
        f"in {latency:.1f}\u202fms."
    )
    logger.info(status)
    return gallery_items, status


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Monochrome(),
    title="EO Visual Scout",
) as demo:
    # -- Header -------------------------------------------------------------
    gr.Markdown(
        """
        <div style="text-align:center">
        <h1>\U0001f6f0\ufe0f EO Visual Scout \u2014 Semantic Image Search</h1>
        <p style="font-size:1.1em;opacity:.85">
        Describe an Earth Observation scene in natural language
        (English or French) and let AI find the matching satellite
        imagery instantly.
        </p>
        </div>
        """
    )

    # -- Main layout --------------------------------------------------------
    with gr.Row():
        # Controls (left) --------------------------------------------------
        with gr.Column(scale=1):
            query_box = gr.Textbox(
                label="Search Query",
                placeholder=(
                    "e.g. 'a river crossing a dense forest'"
                ),
                lines=2,
            )
            slider = gr.Slider(
                minimum=1,
                maximum=24,
                value=9,
                step=1,
                label="Number of results",
            )
            btn = gr.Button(
                "\U0001f50d Search Images",
                variant="primary",
            )
            gr.Markdown(
                "**Examples:** "
                "'zone r\u00e9sidentielle', "
                "'highway', "
                "'agricultural fields'"
            )

        # Results (right) --------------------------------------------------
        with gr.Column(scale=2):
            status_text = gr.Markdown(
                value="Ready.",
                elem_id="status_text",
            )
            gallery = gr.Gallery(
                label="Results",
                show_label=False,
                elem_id="gallery",
                columns=[3],
                rows=[3],
                object_fit="contain",
                height="auto",
            )

    # -- Footer -------------------------------------------------------------
    gr.Markdown(
        "<div style='text-align:center;opacity:.6;margin-top:1.5em'>"
        "Powered by CLIP &amp; EuroSAT Dataset "
        "| \U0001f680 Spacebel Portfolio Project"
        "</div>"
    )

    # -- Events -------------------------------------------------------------
    btn.click(
        fn=search_api,
        inputs=[query_box, slider],
        outputs=[gallery, status_text],
    )
    query_box.submit(
        fn=search_api,
        inputs=[query_box, slider],
        outputs=[gallery, status_text],
    )

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
