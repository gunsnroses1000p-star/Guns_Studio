"""
providers/img2img_provider.py

Hugging Face RealVisXL Img2Img backend.

User-facing controls are intentionally kept simple.
All generation parameters remain backend-only.
"""

from __future__ import annotations

import os
import uuid

import spaces
import torch
from PIL import Image


# =========================================================
# BACKEND CONFIGURATION
# =========================================================

MODEL_ID = "SG161222/RealVisXL_V4.0"

# Hidden backend defaults.
STRENGTH = 0.55
STEPS = 30
GUIDANCE = 5.5

NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted face, "
    "deformed face, bad anatomy, extra fingers, malformed hands, "
    "duplicate person, duplicate face, mutated hands, "
    "cartoon, anime, CGI, 3d render, digital painting, "
    "plastic skin, wax skin, doll skin, porcelain skin, "
    "overprocessed face, beauty filter, airbrushed skin, "
    "oversized eyes, oversized lips, asymmetrical eyes, "
    "distorted eyes, unnatural skin texture"
)

OUTPUT_DIR = "outputs"

_pipe = None


# =========================================================
# MODEL LOADING
# =========================================================

def _load_pipeline():
    global _pipe

    if _pipe is not None:
        return _pipe

    from diffusers import (
        StableDiffusionXLImg2ImgPipeline,
        DPMSolverMultistepScheduler,
    )

    print(
        f"Loading Img2Img model: {MODEL_ID}",
        flush=True,
    )

    hf_token = os.environ.get("HF_TOKEN")

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    _pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=hf_token or None,
    )

    # DPM++ 2M Karras is a strong general-purpose choice
    # for realistic SDXL generation.
    _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        _pipe.scheduler.config,
        use_karras_sigmas=True,
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
        "RealVisXL Img2Img pipeline ready.",
        flush=True,
    )

    return _pipe


# =========================================================
# IMAGE PREPARATION
# =========================================================

def _prepare_image(image: Image.Image) -> Image.Image:
    """
    Preserve the source aspect ratio while keeping the image
    at a sensible SDXL resolution.
    """

    image = image.convert("RGB")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Invalid source image.")

    # Keep the original aspect ratio.
    max_dimension = 1024

    scale = min(
        max_dimension / max(width, height),
        1.0,
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    # SDXL works best with dimensions divisible by 8.
    new_width = max(64, (new_width // 8) * 8)
    new_height = max(64, (new_height // 8) * 8)

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


# =========================================================
# IMG2IMG GENERATION
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
            "Please enter a prompt."
        )

    prompt = prompt.strip()

    pipe = _load_pipeline()

    source = _prepare_image(image)

    # Random seed is deliberately generated internally.
    # The user never sees or controls it.
    seed = torch.randint(
        0,
        2**31 - 1,
        (1,),
    ).item()

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(seed)

    print(
        f"Generating Img2Img | seed={seed}",
        flush=True,
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=source,
        strength=STRENGTH,
        guidance_scale=GUIDANCE,
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