"""
providers/img2img_provider.py

Hugging Face Diffusers Img2Img backend.

All technical generation settings stay backend-only.
The UI should expose only:
    - source image
    - prompt
    - generate button
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

import spaces
from PIL import Image

from config import (
    HF_TOKEN,
    IMAGE_STEPS_DEFAULT,
    IMAGE_GUIDANCE_DEFAULT,
    IMAGE_WIDTH_DEFAULT,
    IMAGE_HEIGHT_DEFAULT,
    OUTPUT_DIR,
)


# =========================================================
# BACKEND DEFAULTS
# =========================================================

# Dedicated Img2Img model.
# Keep this separate from the normal text-to-image model.
IMG2IMG_MODEL_DEFAULT = os.environ.get(
    "IMG2IMG_MODEL",
    "stabilityai/stable-diffusion-xl-base-1.0",
)

# Internal strength.
# Lower = preserve more of the original image.
# Higher = allow more transformation.
IMG2IMG_STRENGTH_DEFAULT = 0.65

# Internal negative prompt.
IMG2IMG_NEGATIVE_DEFAULT = (
    "blurry, low quality, distorted face, deformed face, "
    "bad anatomy, extra fingers, malformed hands, "
    "duplicate person, duplicate face, "
    "cartoon, anime, CGI, 3d render, digital painting, "
    "plastic skin, wax skin, doll skin, "
    "overprocessed face, beauty filter, "
    "oversized eyes, oversized lips, "
    "asymmetrical eyes, distorted eyes"
)


# =========================================================
# GLOBAL PIPELINE STATE
# =========================================================

_pipe = None
_loaded_model = ""


# =========================================================
# MODEL LOADING
# =========================================================

def _load(model_name: str) -> None:
    global _pipe, _loaded_model

    if _pipe is not None and _loaded_model == model_name:
        return

    import torch
    from diffusers import (
        StableDiffusionXLImg2ImgPipeline,
        DPMSolverMultistepScheduler,
    )

    print(
        f"Loading Img2Img model {model_name}...",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    _pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=HF_TOKEN or None,
    )

    # Better quality/speed scheduler.
    _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        _pipe.scheduler.config
    )

    if device == "cuda":
        _pipe.enable_model_cpu_offload()
        _pipe.enable_vae_tiling()
    else:
        _pipe = _pipe.to(device)

    _loaded_model = model_name

    print(
        f"Img2Img model ready: {model_name}",
        flush=True,
    )


# =========================================================
# IMAGE PREPARATION
# =========================================================

def _prepare_image(
    image: Image.Image,
    max_width: int = IMAGE_WIDTH_DEFAULT,
    max_height: int = IMAGE_HEIGHT_DEFAULT,
) -> Image.Image:

    image = image.convert("RGB")

    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Invalid input image.")

    # Preserve the original aspect ratio.
    scale = min(
        max_width / original_width,
        max_height / original_height,
        1.0,
    )

    new_width = max(64, int(original_width * scale))
    new_height = max(64, int(original_height * scale))

    # Diffusion models work best with dimensions divisible by 8.
    new_width = max(64, (new_width // 8) * 8)
    new_height = max(64, (new_height // 8) * 8)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    return image


# =========================================================
# GENERATION
# =========================================================

@spaces.GPU
def generate_img2img(
    image: Image.Image,
    prompt: str,
) -> str:

    if image is None:
        raise ValueError("Please upload an image.")

    if not prompt or not prompt.strip():
        raise ValueError("Please enter a prompt.")

    prompt = prompt.strip()

    _load(IMG2IMG_MODEL_DEFAULT)

    prepared_image = _prepare_image(
        image,
        IMAGE_WIDTH_DEFAULT,
        IMAGE_HEIGHT_DEFAULT,
    )

    # Generate with a random seed internally.
    # The user never sees or controls it.
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = torch.randint(
        0,
        2**31 - 1,
        (1,),
    ).item()

    generator = torch.Generator(
        device=device
    ).manual_seed(seed)

    result = _pipe(
        prompt=prompt,
        negative_prompt=IMG2IMG_NEGATIVE_DEFAULT,
        image=prepared_image,
        strength=IMG2IMG_STRENGTH_DEFAULT,
        guidance_scale=IMAGE_GUIDANCE_DEFAULT,
        num_inference_steps=IMAGE_STEPS_DEFAULT,
        generator=generator,
    )

    output_image = result.images[0]

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"img2img_{uuid.uuid4().hex[:8]}.png",
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