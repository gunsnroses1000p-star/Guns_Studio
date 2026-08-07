"""
providers/hf_provider.py — Image generation via Hugging Face InferenceClient.
"""

import random

import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image

from config import DEFAULT_IMAGE_MODEL, HF_TOKEN

# Single shared client (lazy-initialised per token availability)
_hf_client: InferenceClient | None = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(token=HF_TOKEN)
    return _hf_client


def generate_with_hf(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    model: str = DEFAULT_IMAGE_MODEL,
) -> Image.Image:
    """Generate an image using the Hugging Face Inference API."""
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    client = _get_hf_client()
    result = client.text_to_image(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        seed=int(seed),
        model=model,
    )
    return result
