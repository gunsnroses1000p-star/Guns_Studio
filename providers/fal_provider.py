"""
providers/fal_provider.py — Image and video generation via Fal.ai.
"""

import gradio as gr
import fal_client
from PIL import Image
import requests
from io import BytesIO

from config import FAL_KEY


def _ensure_fal_key():
    if not FAL_KEY:
        raise gr.Error("FAL_KEY is missing from environment secrets.")


def generate_image_with_fal(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    seed: int = 0,
    model: str = "fal-ai/flux/dev",
) -> Image.Image:
    """Generate an image using the Fal.ai API."""
    _ensure_fal_key()
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    result = fal_client.subscribe(
        model,
        arguments={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_size": {"width": int(width), "height": int(height)},
            "num_inference_steps": int(steps),
            "seed": int(seed) if seed and int(seed) > 0 else None,
        },
    )

    images = result.get("images") or []
    if not images:
        raise gr.Error("Fal.ai returned no images.")

    url = images[0].get("url") or images[0]
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def generate_video_with_fal(
    prompt: str,
    image: Image.Image | None = None,
    model: str = "fal-ai/kling-video/v1/standard/image-to-video",
    duration: str = "5",
    aspect_ratio: str = "16:9",
) -> str:
    """Generate a video using the Fal.ai API. Returns the video URL."""
    _ensure_fal_key()
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    arguments: dict = {
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
    }

    if image is not None:
        import io, base64

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        arguments["image_url"] = f"data:image/png;base64,{b64}"

    result = fal_client.subscribe(model, arguments=arguments)
    video = result.get("video") or {}
    url = video.get("url") if isinstance(video, dict) else str(video)
    if not url:
        raise gr.Error("Fal.ai returned no video URL.")
    return url
