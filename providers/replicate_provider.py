"""
providers/replicate_provider.py — Image generation via Replicate API.
"""

import random

import gradio as gr
import replicate

from config import DEFAULT_IMAGE_MODEL, REPLICATE_API_TOKEN
from utils.helpers import extract_output


def generate_with_replicate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    model: str = DEFAULT_IMAGE_MODEL,
) -> str:
    """Run a text-to-image prediction on Replicate and return the output URL."""
    if not REPLICATE_API_TOKEN:
        raise gr.Error("Missing REPLICATE_API_TOKEN in Hugging Face Secrets.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    output = replicate.run(
        model,
        input={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(steps),
            "seed": int(seed),
        },
    )
    return extract_output(output)
