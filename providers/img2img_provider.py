"""
providers/img2img_provider.py

Hugging Face SDXL InstructPix2Pix backend.

Simple user interface:
    - Source image
    - Editing instruction

All technical settings remain backend-only.
"""

from __future__ import annotations

import os
import uuid

import spaces
import torch
from PIL import Image, ImageOps


# =========================================================
# MODEL
# =========================================================

MODEL_ID = "diffusers/sdxl-instructpix2pix-768"

OUTPUT_DIR = "outputs"


# =========================================================
# HIDDEN BACKEND SETTINGS
# =========================================================

STEPS = 35

# Stronger instruction following.
GUIDANCE_SCALE = 7.5

# Slightly less image locking so requested edits can happen.
IMAGE_GUIDANCE_SCALE = 1.2

NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, "
    "distorted face, deformed face, bad anatomy, "
    "extra fingers, malformed hands, duplicate person, "
    "duplicate face, distorted eyes, asymmetrical eyes, "
    "cartoon, anime, CGI, 3d render, digital painting, "
    "plastic skin, wax skin, doll skin, porcelain skin, "
    "overprocessed skin, beauty filter, airbrushed skin, "
    "unnatural skin texture"
)


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

    from diffusers import (
        StableDiffusionXLInstructPix2PixPipeline,
        EulerAncestralDiscreteScheduler,
    )

    print(
        f"Loading Img2Img model: {MODEL_ID}",
        flush=True,
    )

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    _pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    _pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        _pipe.scheduler.config
    )

    if torch.cuda.is_available():
        _pipe.enable_model_cpu_offload()

        try:
            _pipe.enable_vae_tiling()
        except Exception:
            pass

    else:
        _pipe.to("cpu")

    print(
        "SDXL InstructPix2Pix ready.",
        flush=True,
    )

    return _pipe


# =========================================================
# PREPARE IMAGE
# =========================================================

def _prepare_image(
    image: Image.Image,
) -> Image.Image:

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Invalid source image.")

    # Preserve the exact aspect ratio.
    max_dimension = 768

    scale = min(
        max_dimension / max(width, height),
        1.0,
    )

    new_width = max(
        64,
        round(width * scale),
    )

    new_height = max(
        64,
        round(height * scale),
    )

    # Only round dimensions to multiples of 8.
    # Do NOT independently scale width and height.
    new_width = max(
        64,
        (new_width // 8) * 8,
    )

    new_height = max(
        64,
        (new_height // 8) * 8,
    )

    print(
        f"Source size: {width}x{height} -> "
        f"Img2Img size: {new_width}x{new_height}",
        flush=True,
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


# =========================================================
# GENERATE
# =========================================================

@spaces.GPU
def generate_img2img(
    image: Image.Image,
    prompt: str,
) -> str:

    if image is None:
        raise ValueError(
            "Please upload a source image."
        )

    if not prompt or not prompt.strip():
        raise ValueError(
            "Please describe the change you want."
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    generator = torch.Generator(
        device=device
    ).manual_seed(seed)

    # -----------------------------------------------------
    # Give the model a very direct editing instruction.
    # -----------------------------------------------------

    edit_prompt = (
        f"Edit the provided image according to this instruction: "
        f"{prompt}. "
        "Make the requested change clearly visible. "
        "Do not ignore the requested change. "
        "Preserve the same person, face, pose, camera angle, "
        "background, lighting, and composition unless the "
        "instruction specifically asks to change them. "
        "Photorealistic result."
    )

    print(
        f"Img2Img instruction: {prompt}",
        flush=True,
    )

    result = pipe(
        prompt=edit_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=source,
        guidance_scale=GUIDANCE_SCALE,
        image_guidance_scale=IMAGE_GUIDANCE_SCALE,
        num_inference_steps=STEPS,
        generator=generator,
    )

    output_image = result.images[0]

    # -----------------------------------------------------
    # Guarantee the output keeps the same aspect ratio
    # as the prepared source.
    # -----------------------------------------------------

    if output_image.size != source.size:
        output_image = output_image.resize(
            source.size,
            Image.Resampling.LANCZOS,
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"img2img_{uuid.uuid4().hex[:10]}.png",
    )

    output_image.save(
        output_path,
        format="PNG",
    )

    print(
        f"Img2Img complete: {output_path}",
        flush=True,
    )

    return output_path