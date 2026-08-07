"""
providers/lora_provider.py — LoRA image generation and image-to-video flow.
"""

import random

import gradio as gr
import replicate

from config import DEFAULT_NEGATIVE, LORA_URL, DEFAULT_LORA_NAMES
from utils.helpers import extract_output

# Mutable list of available LoRA names (populated at runtime)
lora_names: list[str] = list(DEFAULT_LORA_NAMES)


def generate_with_lora(
    prompt: str,
    lora_name: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    seed: int = 0,
    lora_scale: float = 0.8,
    model: str = "black-forest-labs/FLUX.1-dev",
) -> str:
    """
    Generate an image using a user-selected LoRA weight via Replicate.
    Returns a URL string pointing to the result image.
    """
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    # Build the LoRA hf_lora path from the selected name and LORA_URL base
    lora_path = f"{LORA_URL}/{lora_name}" if LORA_URL and lora_name else lora_name

    output = replicate.run(
        model,
        input={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(steps),
            "seed": int(seed),
            "hf_lora": lora_path,
            "lora_scale": float(lora_scale),
        },
    )
    return extract_output(output)


def generate_lora_image_to_video(
    image_url: str,
    prompt: str,
    duration: int = 5,
    seed: int = 0,
    video_model: str = "minimax/video-01-live",
) -> str:
    """
    Convert a LoRA-generated image to a short video via Replicate.
    Returns the video URL/path.
    """
    if not image_url:
        raise gr.Error("Generate a LoRA image first.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    output = replicate.run(
        video_model,
        input={
            "prompt": prompt or "Cinematic motion, smooth animation",
            "first_frame_image": image_url,
            "duration": int(duration),
            "seed": int(seed),
        },
    )
    return extract_output(output)
