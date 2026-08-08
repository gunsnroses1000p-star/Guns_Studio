"""
providers/img2img_provider.py

Hugging Face SDXL InstructPix2Pix backend.

The user supplies only:
    - source image
    - editing instruction

All technical generation settings remain backend-only.
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

STEPS = 30

# How strongly the written instruction influences the edit.
GUIDANCE_SCALE = 4.0

# How strongly the original image is preserved.
IMAGE_GUIDANCE_SCALE = 1.5

# Internal negative prompt.
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
# LOAD MODEL
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
# PREPARE SOURCE IMAGE
# =========================================================

def _prepare_image(
    image: Image.Image,
) -> Image.Image:

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Invalid source image.")

    # Keep aspect ratio.
    # SDXL InstructPix2Pix works around 768px well.
    max_dimension = 768

    scale = min(
        max_dimension / max(width, height),
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

    # Diffusion dimensions should be divisible by 8.
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


# =========================================================
# GENERATE EDIT
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
    # Strengthen the editing instruction without changing
    # what the user asked for.
    # -----------------------------------------------------

    edit_prompt = (
        f"{prompt}. "
        "Make the requested change clearly and accurately. "
        "Keep the identity, composition, and all unrelated "
        "details of the original image unchanged whenever "
        "the instruction does not ask for them to change. "
        "Maintain a photorealistic natural appearance."
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