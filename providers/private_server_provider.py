"""
providers/private_server_provider.py — Image-to-video via a private server.
"""

import io
import base64
import random

import gradio as gr
import requests
from PIL import Image

from config import PRIVATE_SERVER_URL


def generate_private_server_video(
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    seed: int = 0,
    duration: int = 5,
) -> str:
    """
    Submit an image-to-video job to the private server.
    Returns a video URL or path.
    """
    if not PRIVATE_SERVER_URL:
        raise gr.Error("PRIVATE_SERVER_URL is not configured.")
    if init_image is None:
        raise gr.Error("Please upload an image.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    buf = io.BytesIO()
    init_image.convert("RGB").save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "image_base64": image_b64,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": int(steps),
        "seed": int(seed),
        "duration": int(duration),
    }

    resp = requests.post(
        f"{PRIVATE_SERVER_URL.rstrip('/')}/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    video_url = data.get("video_url") or data.get("url")
    if not video_url:
        raise gr.Error(f"Private server returned no video URL: {data}")
    return video_url
