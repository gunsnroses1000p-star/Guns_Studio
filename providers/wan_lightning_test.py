"""
Temporary Wan 2.2 Lightning I2V test.
Sandbox only — does not replace Hunyuan.
"""

from __future__ import annotations

import os
import uuid

import spaces
import torch

from PIL import Image, ImageOps
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video


MODEL_ID = (
    "fdk6566/"
    "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers"
)

OUTPUT_DIR = "outputs"

MAX_DIMENSION = 512
NUM_FRAMES = 49
NUM_INFERENCE_STEPS = 12
FPS = 24

_pipe = None


def _load_pipeline():

    global _pipe

    if _pipe is not None:
        return _pipe

    print(
        f"Loading Wan 2.2 Lightning: {MODEL_ID}",
        flush=True,
    )

    _pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    _pipe.to("cuda")

    print(
        "Wan 2.2 Lightning ready.",
        flush=True,
    )

    return _pipe


def _prepare_image(image: Image.Image) -> Image.Image:

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    width, height = image.size

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
        (new_width // 16) * 16,
    )

    new_height = max(
        64,
        (new_height // 16) * 16,
    )

    print(
        f"Wan Lightning source: "
        f"{width}x{height} -> "
        f"{new_width}x{new_height}",
        flush=True,
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


@spaces.GPU(duration=300)
def test_wan_lightning(
    image: Image.Image,
    prompt: str,
) -> str:

    if image is None:
        raise ValueError(
            "Please provide a source image."
        )

    if not prompt or not prompt.strip():
        raise ValueError(
            "Please provide a motion prompt."
        )

    pipe = _load_pipeline()

    source = _prepare_image(image)

    prompt = prompt.strip()

    seed = torch.randint(
        0,
        2**31 - 1,
        (1,),
    ).item()

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(seed)

    print(
        f"Wan Lightning prompt: {prompt}",
        flush=True,
    )

    print(
        f"Wan Lightning seed: {seed}",
        flush=True,
    )

    result = pipe(
        image=source,
        prompt=prompt,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
    )

    frames = result.frames[0]

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"wan_lightning_test_{uuid.uuid4().hex[:10]}.mp4",
    )

    export_to_video(
        frames,
        output_path,
        fps=FPS,
    )

    print(
        f"Wan Lightning test complete: {output_path}",
        flush=True,
    )

    return output_path
