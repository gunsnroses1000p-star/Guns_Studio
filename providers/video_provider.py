"""
providers/video_provider.py

HunyuanVideo 1.5 Image-to-Video backend.

User-facing controls:
    - Source image
    - Motion prompt

All technical settings remain backend-only.
"""

from __future__ import annotations

import os
import uuid

import spaces
import torch

from PIL import Image, ImageOps
from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video


# =========================================================
# MODEL
# =========================================================

MODEL_ID = (
    "hunyuanvideo-community/"
    "HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"
)

OUTPUT_DIR = "outputs"


# =========================================================
# HIDDEN BACKEND SETTINGS
# =========================================================

MAX_DIMENSION = 512

NUM_FRAMES = 121

NUM_INFERENCE_STEPS = 12

FPS = 24


# =========================================================
# PIPELINE CACHE
# =========================================================

_pipe = None


# =========================================================
# LOAD PIPELINE
# =========================================================

def _load_pipeline():

    global _pipe

    if _pipe is not None:
        return _pipe

    print(
        f"Loading HunyuanVideo 1.5: {MODEL_ID}",
        flush=True,
    )

    dtype = torch.bfloat16

    _pipe = (
        HunyuanVideo15ImageToVideoPipeline
        .from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
    )

    _pipe.enable_model_cpu_offload()

    try:
        _pipe.vae.enable_tiling()
    except Exception as exc:
        print(
            f"VAE tiling unavailable: {exc}",
            flush=True,
        )

    print(
        "HunyuanVideo 1.5 ready.",
        flush=True,
    )

    return _pipe


# =========================================================
# IMAGE PREPARATION
# =========================================================

def _prepare_image(
    image: Image.Image,
) -> Image.Image:

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError(
            "Invalid source image."
        )

    scale = min(
        MAX_DIMENSION / max(width, height),
        1.0,
    )

    new_width = max(
        64,
        int(width * scale),
    )

    new_height = max(
        64,
        int(height * scale),
    )

    new_width = max(
        64,
        (new_width // 8) * 8,
    )

    new_height = max(
        64,
        (new_height // 8) * 8,
    )

    print(
        f"Video source: "
        f"{width}x{height} -> "
        f"{new_width}x{new_height}",
        flush=True,
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


# =========================================================
# IMAGE → VIDEO
# =========================================================

@spaces.GPU(duration=180)
def generate_video(
    image: Image.Image,
    prompt: str,
) -> str:

    if image is None:
        raise ValueError(
            "Please upload a source image."
        )

    if not prompt or not prompt.strip():
        raise ValueError(
            "Please describe the motion you want."
        )

    prompt = prompt.strip()

    pipe = _load_pipeline()

    source = _prepare_image(image)

    # -----------------------------------------------------
    # Internal random seed.
    # Never exposed to the user.
    # -----------------------------------------------------

    seed = torch.randint(
        0,
        2**31 - 1,
        (1,),
    ).item()

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(seed)

    # -----------------------------------------------------
    # Motion instruction.
    # -----------------------------------------------------

    video_prompt = (
        f"{prompt}. "
        "Preserve the same person, face, identity, "
        "clothing, environment, camera framing and "
        "overall composition from the source image. "
        "Create natural realistic motion. "
        "Maintain realistic human anatomy and facial "
        "features throughout the video. "
        "No morphing, no identity drift, no melting, "
        "no sudden scene changes."
    )

    print(
        f"Hunyuan I2V prompt: {prompt}",
        flush=True,
    )

    print(
        f"Hunyuan seed: {seed}",
        flush=True,
    )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    result = pipe(
        image=source,
        prompt=video_prompt,
        generator=generator,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    frames = result.frames[0]

    # -----------------------------------------------------
    # Save video
    # -----------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"hunyuan_i2v_{uuid.uuid4().hex[:10]}.mp4",
    )

    export_to_video(
        frames,
        output_path,
        fps=FPS,
    )

    print(
        f"Hunyuan video complete: {output_path}",
        flush=True,
    )

    return output_path