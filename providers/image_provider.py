"""
providers/image_provider.py – Text-to-image generation via Diffusers.
"""

from __future__ import annotations

import os
from PIL import Image

from config import (
    HF_TOKEN,
    IMAGE_MODEL_DEFAULT,
    IMAGE_STEPS_DEFAULT,
    IMAGE_GUIDANCE_DEFAULT,
    IMAGE_WIDTH_DEFAULT,
    IMAGE_HEIGHT_DEFAULT,
    OUTPUT_DIR,
)

_pipe = None
_loaded_model = ""
torch = None
StableDiffusionXLPipeline = None
DPMSolverMultistepScheduler = None


def _load(model_name: str) -> None:
    global _pipe, _loaded_model
    global torch, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

    if torch is None:
        import torch as _torch
        from diffusers import (
            StableDiffusionXLPipeline as _Pipeline,
            DPMSolverMultistepScheduler as _Scheduler,
        )

        torch = _torch
        StableDiffusionXLPipeline = _Pipeline
        DPMSolverMultistepScheduler = _Scheduler

    if _loaded_model == model_name and _pipe is not None:
        return

    # ...the rest of your existing _load() code...

def generate(
    prompt: str,
    negative_prompt: str = "",
    model_name: str = IMAGE_MODEL_DEFAULT,
    steps: int = IMAGE_STEPS_DEFAULT,
    guidance_scale: float = IMAGE_GUIDANCE_DEFAULT,
    width: int = IMAGE_WIDTH_DEFAULT,
    height: int = IMAGE_HEIGHT_DEFAULT,
    seed: int = -1,
) -> str:
    """Generate an image and return the saved file path."""
    _load(model_name)
    generator = None
    if seed >= 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    image: Image.Image = _pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"image_{_unique_id()}.png")
    image.save(out_path)
    return out_path


def _unique_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]
