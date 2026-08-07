"""
config.py – Central configuration for Guns AI Studio.
All model names, paths, and tuneable defaults live here.
"""

import os

# ── Hugging Face auth ────────────────────────────────────────────────────────
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "outputs")

# ── Text / LLM ───────────────────────────────────────────────────────────────
TEXT_MODEL_DEFAULT: str = "mistralai/Mistral-7B-Instruct-v0.2"
TEXT_MAX_NEW_TOKENS: int = 512
TEXT_TEMPERATURE: float = 0.7
TEXT_TOP_P: float = 0.9

# ── Image generation ─────────────────────────────────────────────────────────
IMAGE_MODEL_DEFAULT: str = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_STEPS_DEFAULT: int = 30
IMAGE_GUIDANCE_DEFAULT: float = 7.5
IMAGE_WIDTH_DEFAULT: int = 1024
IMAGE_HEIGHT_DEFAULT: int = 1024

# ── Audio / TTS ──────────────────────────────────────────────────────────────
TTS_MODEL_DEFAULT: str = "facebook/mms-tts-eng"

# ── Audio / Music ────────────────────────────────────────────────────────────
MUSIC_MODEL_DEFAULT: str = "facebook/musicgen-small"
MUSIC_DURATION_DEFAULT: int = 10

# ── Video generation ─────────────────────────────────────────────────────────
VIDEO_MODEL_DEFAULT: str = "damo-vilab/text-to-video-ms-1.7b"
VIDEO_FRAMES_DEFAULT: int = 16
VIDEO_FPS_DEFAULT: int = 8

# ── Face swap ────────────────────────────────────────────────────────────────
FACE_SWAP_MODEL: str = "inswapper_128.onnx"  # must be placed in models/ dir

# ── Upscaling ────────────────────────────────────────────────────────────────
UPSCALE_MODEL_DEFAULT: str = "stabilityai/stable-diffusion-x4-upscaler"
UPSCALE_STEPS_DEFAULT: int = 20
