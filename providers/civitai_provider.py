"""
providers/civitai_provider.py — Image generation via the Civitai API.
"""

import random
from io import BytesIO

import gradio as gr
import requests
from PIL import Image

from config import CIVITAI_API_KEY


_CIVITAI_BASE = "https://civitai.com/api/v1"


def generate_with_civitai(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    seed: int = 0,
    model_id: str = "",
    cfg_scale: float = 7.0,
    sampler: str = "Euler a",
) -> Image.Image:
    """Generate an image using the Civitai Inference API."""
    if not CIVITAI_API_KEY:
        raise gr.Error("CIVITAI_API_KEY is missing from environment secrets.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    payload: dict = {
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "seed": int(seed),
        "cfgScale": float(cfg_scale),
        "sampler": sampler,
    }
    if model_id:
        payload["modelVersionId"] = int(model_id)

    auth_header = "Token " + CIVITAI_API_KEY
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    resp = requests.post(
        f"{_CIVITAI_BASE}/images/generate",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    images = data.get("images") or []
    if not images:
        raise gr.Error(f"Civitai returned no images: {data}")

    url = images[0].get("url") or images[0]
    img_resp = requests.get(url, timeout=120)
    img_resp.raise_for_status()
    return Image.open(BytesIO(img_resp.content)).convert("RGB")
