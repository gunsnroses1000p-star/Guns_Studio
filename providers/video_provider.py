"""
providers/video_provider.py – Text-to-video generation via Diffusers.
"""

from __future__ import annotations

import os
import uuid

import torch
from diffusers import DiffusionPipeline
import imageio

from config import (
    HF_TOKEN,
    VIDEO_MODEL_DEFAULT,
    VIDEO_FRAMES_DEFAULT,
    VIDEO_FPS_DEFAULT,
    OUTPUT_DIR,
)

_pipe = None
_loaded_model: str = ""


def _load(model_name: str) -> None:
    global _pipe, _loaded_model
    if _loaded_model == model_name and _pipe is not None:
        return
    _pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        token=HF_TOKEN or None,
    )
    _pipe = _pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    _loaded_model = model_name


def generate(
    prompt: str,
    model_name: str = VIDEO_MODEL_DEFAULT,
    num_frames: int = VIDEO_FRAMES_DEFAULT,
    fps: int = VIDEO_FPS_DEFAULT,
) -> str:
    """Generate a short video and return the saved .mp4 file path."""
    _load(model_name)
    result = _pipe(prompt, num_frames=num_frames)
    frames = result.frames[0]  # list of PIL Images

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"video_{uuid.uuid4().hex[:8]}.mp4")
    imageio.mimsave(out_path, frames, fps=fps)
    return out_path
