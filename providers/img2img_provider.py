"""
providers/img2img_provider.py

Hugging Face Qwen Image Edit backend.

User-facing controls:
    - Source image
    - Editing instruction

All technical/provider settings remain backend-only.
"""

from __future__ import annotations

import io
import os
import uuid

from PIL import Image
from huggingface_hub import InferenceClient


# =========================================================
# BACKEND CONFIGURATION
# =========================================================

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"

OUTPUT_DIR = "outputs"


# =========================================================
# HUGGING FACE CLIENT
# =========================================================

_client = None


def _get_client() -> InferenceClient:
    global _client

    if _client is not None:
        return _client

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not configured in the Space secrets."
        )

    _client = InferenceClient(
        api_key=token,
        provider="auto",
    )

    return _client


# =========================================================
# IMAGE PREPARATION
# =========================================================

def _prepare_image(image: Image.Image) -> bytes:
    """
    Preserve the original aspect ratio.

    We do not force the image into a fixed width/height.
    """

    image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# =========================================================
# IMAGE EDITING
# =========================================================

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

    client = _get_client()

    image_bytes = _prepare_image(image)

    # -----------------------------------------------------
    # Make the instruction explicit while preserving the
    # user's actual request.
    # -----------------------------------------------------

    edit_prompt = (
        f"{prompt}. "
        "Only make the requested change. "
        "Preserve the same person and facial identity. "
        "Preserve the original pose, body position, "
        "camera angle, framing, background, lighting, "
        "and composition unless the instruction explicitly "
        "asks for them to change. "
        "Keep all unrelated details unchanged. "
        "Maintain a photorealistic appearance."
    )

    print(
        f"Qwen Image Edit instruction: {prompt}",
        flush=True,
    )

    try:

        result = client.image_to_image(
            image_bytes,
            prompt=edit_prompt,
            model=MODEL_ID,
        )

    except Exception as exc:

        print(
            f"Qwen Image Edit error: {exc}",
            flush=True,
        )

        raise RuntimeError(
            "The Hugging Face image editing service "
            "could not complete the request."
        ) from exc

    # -----------------------------------------------------
    # Save result locally for Gradio.
    # -----------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"img2img_{uuid.uuid4().hex[:10]}.png",
    )

    if isinstance(result, Image.Image):

        result.save(
            output_path,
            format="PNG",
        )

    else:

        # Handle raw image bytes if returned by the
        # inference provider.
        if isinstance(result, bytes):

            result_image = Image.open(
                io.BytesIO(result)
            ).convert("RGB")

        else:

            raise RuntimeError(
                "Unexpected image response from "
                "Hugging Face."
            )

        result_image.save(
            output_path,
            format="PNG",
        )

    print(
        f"Img2Img complete: {output_path}",
        flush=True,
    )

    return output_path