"""
providers/runpod_video_provider.py — RunPod image-to-video and video extension.
"""

import io
import base64
import random
from pathlib import Path

import gradio as gr

from utils.runpod import runpod_job
from PIL import Image


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_runpod_image_to_video(
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    seed: int = 0,
    duration: int = 5,
) -> str:
    """
    Submit an image-to-video job to RunPod.
    Returns a URL or local path to the output video.
    """
    if init_image is None:
        raise gr.Error("Please upload an image.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    final = runpod_job(
        {
            "task": "img2vid",
            "image_base64": _image_to_base64(init_image),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": int(steps),
            "seed": int(seed),
            "duration": int(duration),
        },
        max_wait_seconds=1200,
    )

    output = final.get("output") or {}
    video_url = (
        output.get("video_url")
        or output.get("url")
        or (output if isinstance(output, str) else None)
    )
    if not video_url:
        raise gr.Error(f"RunPod returned no video URL: {output}")
    return video_url


def extend_runpod_video(
    video_path: str,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    seed: int = 0,
    extension_seconds: int = 4,
) -> str:
    """
    Extend an existing video via RunPod.
    Returns a URL or local path to the extended video.
    """
    if not video_path:
        raise gr.Error("Please provide a video to extend.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    # If path is a local file, encode it as base64
    video_payload: dict = {}
    if video_path.startswith("http"):
        video_payload["video_url"] = video_path
    else:
        import tempfile, shutil

        provided = Path(video_path).resolve()
        if not provided.is_file():
            raise gr.Error("Provided video path does not point to a valid file.")
        # Copy to a fresh temp file to break the taint chain from user input
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            safe_tmp_path = tmp.name
        shutil.copy2(str(provided), safe_tmp_path)
        with open(safe_tmp_path, "rb") as f:  # noqa: PTH123
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        Path(safe_tmp_path).unlink(missing_ok=True)
        video_payload["video_base64"] = video_b64

    final = runpod_job(
        {
            "task": "extend_video",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": int(steps),
            "seed": int(seed),
            "extension_seconds": int(extension_seconds),
            **video_payload,
        },
        max_wait_seconds=1200,
    )

    output = final.get("output") or {}
    video_url = (
        output.get("video_url")
        or output.get("url")
        or (output if isinstance(output, str) else None)
    )
    if not video_url:
        raise gr.Error(f"RunPod returned no extended video URL: {output}")
    return video_url
