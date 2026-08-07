"""
providers/audio_provider.py – Text-to-speech and music generation.
"""

from __future__ import annotations

import os
import uuid
import numpy as np
import soundfile as sf

import torch
from transformers import pipeline as hf_pipeline

from config import (
    HF_TOKEN,
    TTS_MODEL_DEFAULT,
    MUSIC_MODEL_DEFAULT,
    MUSIC_DURATION_DEFAULT,
    OUTPUT_DIR,
)

_tts_pipe = None
_tts_model: str = ""

_music_model = None
_music_processor = None
_music_model_name: str = ""


# ── TTS ─────────────────────────────────────────────────────────────────────

def _load_tts(model_name: str) -> None:
    global _tts_pipe, _tts_model
    if _tts_model == model_name and _tts_pipe is not None:
        return
    _tts_pipe = hf_pipeline(
        "text-to-speech",
        model=model_name,
        token=HF_TOKEN or None,
    )
    _tts_model = model_name


def tts(
    text: str,
    model_name: str = TTS_MODEL_DEFAULT,
) -> str:
    """Synthesise speech and return the saved .wav file path."""
    _load_tts(model_name)
    result = _tts_pipe(text)
    audio_array = result["audio"]
    sample_rate = result["sampling_rate"]
    if isinstance(audio_array, np.ndarray) and audio_array.ndim > 1:
        audio_array = audio_array.squeeze()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"tts_{uuid.uuid4().hex[:8]}.wav")
    sf.write(out_path, audio_array, sample_rate)
    return out_path


# ── Music generation ─────────────────────────────────────────────────────────

def _load_music(model_name: str) -> None:
    global _music_model, _music_processor, _music_model_name
    if _music_model_name == model_name and _music_model is not None:
        return
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    _music_processor = AutoProcessor.from_pretrained(
        model_name, token=HF_TOKEN or None
    )
    _music_model = MusicgenForConditionalGeneration.from_pretrained(
        model_name, token=HF_TOKEN or None
    )
    _music_model_name = model_name


def generate_music(
    prompt: str,
    model_name: str = MUSIC_MODEL_DEFAULT,
    duration: int = MUSIC_DURATION_DEFAULT,
) -> str:
    """Generate music and return the saved .wav file path."""
    _load_music(model_name)
    inputs = _music_processor(text=[prompt], padding=True, return_tensors="pt")
    tokens_per_sec = _music_model.config.audio_encoder.frame_rate
    max_new_tokens = int(duration * tokens_per_sec)

    with torch.no_grad():
        audio_values = _music_model.generate(**inputs, max_new_tokens=max_new_tokens)

    sample_rate = _music_model.config.audio_encoder.sampling_rate
    audio_np = audio_values[0, 0].cpu().numpy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"music_{uuid.uuid4().hex[:8]}.wav")
    sf.write(out_path, audio_np, sample_rate)
    return out_path
