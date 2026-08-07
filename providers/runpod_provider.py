"""
providers/runpod_provider.py — Text-to-image and img2img via RunPod.
"""

import io
import base64
import random

import gradio as gr
from PIL import Image

from utils.runpod import runpod_job, decode_runpod_output
from utils.face import preserve_original_face


# -------------------------------------------------------------------------
# Text-to-image
# -------------------------------------------------------------------------

def generate_with_runpod(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> Image.Image:
    """Submit a text-to-image job to RunPod and return the result image."""
    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    final = runpod_job(
        {
            "task": "txt2img",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "seed": int(seed),
        },
        max_wait_seconds=900,
    )
    return decode_runpod_output(final.get("output"))


# -------------------------------------------------------------------------
# Img2img (local GPU)
# -------------------------------------------------------------------------

_img2img_pipe = None
_current_pipe_repo: str | None = None


def _get_img2img_pipe(repo_id: str):
    global _img2img_pipe, _current_pipe_repo
    if _img2img_pipe is not None and _current_pipe_repo == repo_id:
        return _img2img_pipe

    import torch
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        DPMSolverMultistepScheduler,
    )

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    _img2img_pipe = pipe
    _current_pipe_repo = repo_id
    return pipe


def generate_img2img_local(
    prompt: str,
    init_image: Image.Image,
    init_image_2,
    model: str,
    strength: float,
    guidance: float,
    steps: int,
    seed: int,
    preserve_face: bool,
    face_blend: float,
):
    """Run SD img2img locally on GPU."""
    import torch

    if not prompt:
        raise gr.Error("Please enter a prompt.")
    if init_image is None:
        raise gr.Error("Please upload an image.")

    repo_id = model or "runwayml/stable-diffusion-v1-5"
    pipe = _get_img2img_pipe(repo_id)

    orig_w, orig_h = init_image.size
    max_dim = 768
    scale = max_dim / max(orig_w, orig_h)
    new_w = max(8, (int(orig_w * scale) // 8) * 8)
    new_h = max(8, (int(orig_h * scale) // 8) * 8)
    input_image = init_image.convert("RGB").resize(
        (new_w, new_h), Image.LANCZOS
    )

    if seed is None or int(seed) == 0:
        seed = random.randint(1, 2_147_483_647)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=input_image,
            strength=float(strength),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=generator,
        ).images[0]

    result = result.resize((orig_w, orig_h), Image.LANCZOS)

    if preserve_face:
        result = preserve_original_face(
            init_image, result, strength=float(face_blend)
        )

    return result, f"✅ Img2Img complete. Seed: {seed}"


# -------------------------------------------------------------------------
# Img2img (RunPod)
# -------------------------------------------------------------------------

def generate_img2img_runpod(
    prompt: str,
    init_image: Image.Image,
    init_image_2,
    model: str,
    strength: float,
    guidance: float,
    steps: int,
    seed: int,
    preserve_face: bool,
    face_blend: float,
):
    """Submit an img2img job to RunPod."""
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    if init_image is None:
        raise gr.Error("Please upload an image.")

    buf = io.BytesIO()
    init_image.convert("RGB").save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    if seed is None or int(seed) == 0:
        seed = random.randint(1, 2_147_483_647)

    final = runpod_job(
        {
            "task": "img2img",
            "prompt": prompt,
            "image_base64": image_base64,
            "strength": float(strength),
            "guidance_scale": float(guidance),
            "steps": int(steps),
            "seed": int(seed),
        },
        max_wait_seconds=900,
    )
    result = decode_runpod_output(final.get("output"))

    if preserve_face and result is not None:
        result = preserve_original_face(
            init_image, result, strength=float(face_blend)
        )

    return result, f"✅ RunPod Img2Img complete. Seed: {seed}"


# -------------------------------------------------------------------------
# Provider dispatcher
# -------------------------------------------------------------------------

def generate_img2img_with_provider(
    provider: str,
    prompt: str,
    init_image: Image.Image,
    init_image_2,
    model: str,
    strength: float,
    guidance: float,
    steps: int,
    seed: int,
    preserve_face: bool,
    face_blend: float,
):
    if provider == "RunPod":
        return generate_img2img_runpod(
            prompt, init_image, init_image_2,
            model, strength, guidance, steps, seed,
            preserve_face, face_blend,
        )
    return generate_img2img_local(
        prompt, init_image, init_image_2,
        model, strength, guidance, steps, seed,
        preserve_face, face_blend,
    )
