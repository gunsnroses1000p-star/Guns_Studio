""" providers/audio_provider.py – Text-to-speech and music generation. """
from __future__ import annotations
import os
import uuid
import spaces 
from config import (
    HF_TOKEN, 
    TTS_MODEL_DEFAULT, 
    MUSIC_MODEL_DEFAULT, 
    MUSIC_DURATION_DEFAULT, 
    OUTPUT_DIR,
)

# Global variables for TTS
_tts_pipe = None
_tts_model_name: str = ""

# Global variables for Music
_music_model = None
_music_processor = None
_music_model_name: str = ""

def _load_audio_io():
    """Lazy import and return audio tools"""
    import numpy as np
    import soundfile as sf
    return np, sf

# ── TTS ─────────────────────────────────────────────────────────────────────

def _load_tts(model_name: str) -> None:
    global _tts_pipe, _tts_model_name
    
    if _tts_model_name == model_name and _tts_pipe is not None:
        return

    from transformers import pipeline
    print(f"Loading TTS model {model_name}... please wait.")
    
    _tts_pipe = pipeline(
        "text-to-speech", 
        model=model_name, 
        token=HF_TOKEN or None,
    )
    _tts_model_name = model_name

def tts(
    text: str, 
    model_name: str = TTS_MODEL_DEFAULT,
) -> str:
    """Synthesise speech and return the saved .wav file path."""
    _load_tts(model_name)
    np, sf = _load_audio_io()
    
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

    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    print(f"Loading Music model {model_name}... please wait.")
    
    _music_processor = AutoProcessor.from_pretrained(
        model_name, 
        token=HF_TOKEN or None
    )
    _music_model = MusicgenForConditionalGeneration.from_pretrained(
        model_name, 
        token=HF_TOKEN or None
    )
    
    # RTX 6000 Optimization: Move to GPU and use half-precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _music_model = _music_model.to(device).half() if device == "cuda" else _music_model
    
    _music_model_name = model_name
@spaces.GPU
def generate_music(
    prompt: str, 
    model_name: str = MUSIC_MODEL_DEFAULT, 
    duration: int = MUSIC_DURATION_DEFAULT,
) -> str:
    """Generate music and return the saved .wav file path."""
    import torch
    _load_music(model_name)
    np, sf = _load_audio_io()
    
    # Process inputs
    inputs = _music_processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = inputs.to(_music_model.device) # Ensure inputs are on the same device as model
    
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