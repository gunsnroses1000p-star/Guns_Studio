"""
providers/video_provider.py — Local Hugging Face video generation
                               (LTX Image-to-Video and CogVideoX).
"""

import random
import tempfile
from pathlib import Path

import gradio as gr
import spaces
import torch
from PIL import Image

from config import HF_TOKEN, DEFAULT_NEGATIVE


# -------------------------------------------------------------------------
# LTX Image-to-Video (local GPU)
# -------------------------------------------------------------------------

_ltx_pipe = None


def _get_ltx_pipe():
    global _ltx_pipe
    if _ltx_pipe is None:
        from diffusers import LTXImageToVideoPipeline

        _ltx_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
        ).to("cuda")
    return _ltx_pipe


@spaces.GPU(duration=180)
def generate_ltx_video(
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    width: int = 768,
    height: int = 512,
    num_frames: int = 25,
    fps: int = 8,
    steps: int = 50,
    seed: int = 0,
) -> str:
    """Run LTX image-to-video locally on GPU. Returns saved video path."""
    from diffusers.utils import export_to_video

    if init_image is None:
        raise gr.Error("Please upload an image.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    pipe = _get_ltx_pipe()
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    with torch.no_grad():
        frames = pipe(
            image=init_image.convert("RGB"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            generator=generator,
        ).frames[0]

    safe_seed = int(seed) & 0x7FFFFFFF  # cast to safe int for filename
    out_path = Path("outputs") / f"ltx_video_{safe_seed}.mp4"
    out_path.parent.mkdir(exist_ok=True)
    export_to_video(frames, str(out_path), fps=int(fps))
    return str(out_path)


# -------------------------------------------------------------------------
# CogVideoX (local GPU)
# -------------------------------------------------------------------------

_cogvideo_pipe = None


def _get_cogvideo_pipe():
    global _cogvideo_pipe
    if _cogvideo_pipe is None:
        from diffusers import CogVideoXPipeline

        _cogvideo_pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
        ).to("cuda")
    return _cogvideo_pipe


@spaces.GPU(duration=300)
def generate_cogvideo(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    width: int = 720,
    height: int = 480,
    num_frames: int = 49,
    fps: int = 8,
    steps: int = 50,
    seed: int = 0,
) -> str:
    """Run CogVideoX text-to-video locally on GPU. Returns saved video path."""
    from diffusers.utils import export_to_video

    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    pipe = _get_cogvideo_pipe()
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    with torch.no_grad():
        frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            generator=generator,
        ).frames[0]

    safe_seed = int(seed) & 0x7FFFFFFF  # cast to safe int for filename
    out_path = Path("outputs") / f"cogvideo_{safe_seed}.mp4"
    out_path.parent.mkdir(exist_ok=True)
    export_to_video(frames, str(out_path), fps=int(fps))
    return str(out_path)
