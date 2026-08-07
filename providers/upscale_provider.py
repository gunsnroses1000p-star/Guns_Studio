"""
providers/upscale_provider.py – Image upscaling via a Stable Diffusion x4 upscaler.
"""

from __future__ import annotations

import os
import uuid

import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

from config import (
    HF_TOKEN,
    UPSCALE_MODEL_DEFAULT,
    UPSCALE_STEPS_DEFAULT,
    OUTPUT_DIR,
)

_pipe = None
_loaded_model: str = ""


def _load(model_name: str) -> None:
    global _pipe, _loaded_model
    if _loaded_model == model_name and _pipe is not None:
        return
    _pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        token=HF_TOKEN or None,
    )
    _pipe = _pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    _loaded_model = model_name


def upscale(
    image_path: str,
    prompt: str = "",
    model_name: str = UPSCALE_MODEL_DEFAULT,
    steps: int = UPSCALE_STEPS_DEFAULT,
) -> str:
    """Upscale *image_path* 4× and return the saved file path."""
    _load(model_name)

    low_res = Image.open(image_path).convert("RGB")
    result_image: Image.Image = _pipe(
        prompt=prompt,
        image=low_res,
        num_inference_steps=steps,
    ).images[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"upscaled_{uuid.uuid4().hex[:8]}.png")
    result_image.save(out_path)
    return out_path
