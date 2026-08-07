"""
providers/ip_adapter_provider.py — IP-Adapter FaceID image generation.
"""

import random
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from config import HF_TOKEN, DEFAULT_NEGATIVE
from utils.face import load_face_analyzer


_ip_model = None


def _get_ip_model():
    global _ip_model
    if _ip_model is None:
        from diffusers import StableDiffusionXLPipeline
        from huggingface_hub import hf_hub_download
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL

        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
            token=HF_TOKEN,
        ).to("cuda")

        ip_ckpt = hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID",
            filename="ip-adapter-faceid-plusv2_sdxl.bin",
            token=HF_TOKEN,
        )
        _ip_model = IPAdapterFaceIDPlusXL(
            pipe,
            image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            ip_ckpt=ip_ckpt,
            device="cuda",
            torch_dtype=torch.float16,
        )
    return _ip_model


def generate_with_ip_adapter(
    reference_image: Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    seed: int = 0,
    scale: float = 0.8,
) -> Image.Image:
    """Generate a face-conditioned image using IP-Adapter FaceID Plus XL."""
    import numpy as np

    if reference_image is None:
        raise gr.Error("Please upload a reference face image.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if seed is None or int(seed) <= 0:
        seed = random.randint(1, 2_147_483_647)

    analyzer = load_face_analyzer()
    faces = analyzer.get(np.asarray(reference_image.convert("RGB")))
    if not faces:
        raise gr.Error("No face detected in the reference image.")

    face_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    ip = _get_ip_model()
    images = ip.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        face_image=reference_image,
        faceid_embeds=face_embeds,
        shortcut=True,
        s_scale=float(scale),
        num_samples=1,
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        seed=int(seed),
    )
    return images[0]
