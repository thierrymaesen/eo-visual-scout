"""EO Visual Scout — Gradio frontend.

Provides a dark-themed web UI that queries the FastAPI backend
(POST /search) and displays matching satellite images in a gallery.

Sprint 9/10 — UI Image-to-Image (Frontend).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import requests
from PIL import Image

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
    image_pil: Optional[Image.Image],
    top_k: int,
) -> Tuple[List[Tuple[str, str]], str]:
    """Call the FastAPI */search* endpoint and return gallery items.

    Accepts either a text query **or** an uploaded PIL image.
    The image is encoded to base64 before being sent to the API.
    """
    query = (query or "").strip()

    if not query and image_pil is None:
        return (
            [],
            "⚠️ Please enter text or upload an image.",
        )

    payload: dict = {"top_k": int(top_k)}

    if image_pil is not None:
        buffered = BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(
            buffered.getvalue()
        ).decode("utf-8")
        payload["image_base64"] = img_str
        logger.info(
            "Image search (top_k=%d, size=%d bytes)",
            payload["top_k"],
            len(img_str),
        )
    else:
        payload["query"] = query
        logger.info(
            "Text search: %s (top_k=%d)",
            query,
            payload["top_k"],
        )

    try:
        response = requests.post(
            f"{API_URL}/search",
            json=payload,
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        logger.error("API request failed: %s", exc)
        return (
            [],
            "❌ Error: API is unreachable. "
            "Is the backend running?",
        )

    if response.status_code != 200:
        msg = (
            f"❌ HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
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
            f"[{item['class_name']}] "
            f"Score: {item['score']:.3f}"
        )
        gallery_items.append((str(img_path), caption))

    latency = data.get("latency_ms", 0)
    mode = "image" if image_pil is not None else "text"
    status = (
        f"✅ Found {len(gallery_items)} results "
        f"in {latency:.1f}\u202fms ({mode} search)."
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
    # -- Header ---------------------------------------------------------
    gr.Markdown(
        """
        <div style="text-align:center">
        <h1>\U0001f6f0\ufe0f EO Visual Scout"""
        + """ \u2014 Semantic Image Search</h1>
        <p style="font-size:1.1em;opacity:.85">
        Search satellite imagery by text <b>or</b> by image.
        Describe a scene in natural language or upload a photo
        and let AI find the most similar EuroSAT tiles.
        </p>
        </div>
        """
    )

    # -- Main layout ----------------------------------------------------
    with gr.Row():
        # Controls (left) ----------------------------------------------
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem(
                    "\U0001f4dd Text Search"
                ):
                    query_box = gr.Textbox(
                        label="Describe the scene",
                        placeholder=(
                            "e.g. 'a river crossing"
                            " a dense forest'"
                        ),
                        lines=2,
                    )
                with gr.TabItem(
                    "\U0001f5bc\ufe0f Image Search"
                ):
                    image_box = gr.Image(
                        label="Upload a satellite image",
                        type="pil",
                        height=200,
                    )

            slider = gr.Slider(
                minimum=1,
                maximum=24,
                value=9,
                step=1,
                label="Number of results",
            )
            search_btn = gr.Button(
                "\U0001f50d Search Images",
                variant="primary",
            )
            gr.Markdown(
                "**Examples:** "
                "'zone r\u00e9sidentielle', "
                "'highway', "
                "'agricultural fields'"
            )

        # Results (right) ----------------------------------------------
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

    # -- Footer ---------------------------------------------------------
    gr.Markdown(
        "<div style='text-align:center;opacity:.6;"
        "margin-top:1.5em'>"
        "Powered by CLIP &amp; EuroSAT Dataset "
        "| \U0001f680 Space EO Visual Scout "
        "Portfolio Project - By Thierry Maesen "
        "</div>"
    )

    # -- Events ---------------------------------------------------------
    search_btn.click(
        fn=search_api,
        inputs=[query_box, image_box, slider],
        outputs=[gallery, status_text],
    )
    query_box.submit(
        fn=search_api,
        inputs=[query_box, image_box, slider],
        outputs=[gallery, status_text],
    )

    # Clear the other input when one is used
    image_box.change(
        fn=lambda _img: "",
        inputs=[image_box],
        outputs=[query_box],
    )
    query_box.change(
        fn=lambda _txt: None,
        inputs=[query_box],
        outputs=[image_box],
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
