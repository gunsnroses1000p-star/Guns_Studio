"""
utils/helpers.py — General-purpose helper utilities.
"""

import io
import base64
import random

import gradio as gr
from PIL import Image

from config import REPLICATE_API_TOKEN


def check_token() -> None:
    """Raise a Gradio error if the Replicate API token is missing."""
    if not REPLICATE_API_TOKEN:
        raise gr.Error(
            "Missing REPLICATE_API_TOKEN in Hugging Face Secrets."
        )


def extract_output(output) -> str:
    """Extract a URL string from a provider output (list or string)."""
    if isinstance(output, list):
        output = output[0]
    return str(output)


def save_single_reference(image: Image.Image) -> str:
    """Save a PIL image as a temporary PNG and return the path."""
    path = "/tmp/reference_image.png"
    image.convert("RGB").save(path)
    return path


def save_combo_image(image1: Image.Image, image2: Image.Image) -> str:
    """
    Side-by-side combine two reference images (normalised to 512 px height)
    and save to /tmp.  Returns the saved path.
    """
    target_height = 512

    def _resize(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        scale = target_height / h
        new_w = max(8, (int(w * scale) // 8) * 8)
        return img.resize((new_w, target_height), Image.LANCZOS)

    img1 = _resize(image1)
    img2 = _resize(image2)
    total_width = img1.width + img2.width
    combo = Image.new("RGB", (total_width, target_height))
    combo.paste(img1, (0, 0))
    combo.paste(img2, (img1.width, 0))
    path = "/tmp/ai_seamless_reference.png"
    combo.save(path)
    return path


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def random_seed() -> int:
    """Return a random positive seed."""
    return random.randint(1, 2_147_483_647)


def resolve_seed(seed) -> int:
    """Return a random seed when *seed* is None or <= 0."""
    if seed is None or int(seed) <= 0:
        return random_seed()
    return int(seed)
