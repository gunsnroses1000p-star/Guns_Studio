""" providers/text_provider.py – Text generation via a causal-LM pipeline. """
from __future__ import annotations
import threading
import spaces
from typing import Iterator
from config import (
    HF_TOKEN, 
    TEXT_MODEL_DEFAULT, 
    TEXT_MAX_NEW_TOKENS, 
    TEXT_TEMPERATURE, 
    TEXT_TOP_P,
)

# Global variables to store the model and state
_pipe = None
_loaded_model: str = ""

def _load(model_name: str) -> None:
    global _pipe, _loaded_model
    
    # 1. If the correct model is already loaded, do nothing
    if _loaded_model == model_name and _pipe is not None:
        return

    # 2. Lazy import transformers ONLY when needed
    from transformers import pipeline as hf_pipeline
    
    print(f"Loading text model {model_name} into VRAM... please wait.")
    
    # 3. Load the pipeline
    # device_map="auto" will utilize your RTX 6000's 48GB VRAM efficiently
    _pipe = hf_pipeline(
        "text-generation", 
        model=model_name, 
        token=HF_TOKEN or None, 
        device_map="auto",
    )
    
    _loaded_model = model_name
@spaces.GPU
def generate(
    prompt: str, 
    model_name: str = TEXT_MODEL_DEFAULT, 
    max_new_tokens: int = TEXT_MAX_NEW_TOKENS, 
    temperature: float = TEXT_TEMPERATURE, 
    top_p: float = TEXT_TOP_P,
) -> str:
    """Return generated text for *prompt*."""
    # Ensure model is loaded
    _load(model_name)
    
    result = _pipe(
        prompt, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        do_sample=True, 
        return_full_text=False,
    )
    return result[0]["generated_text"]

def stream(
    prompt: str, 
    model_name: str = TEXT_MODEL_DEFAULT, 
    max_new_tokens: int = TEXT_MAX_NEW_TOKENS, 
    temperature: float = TEXT_TEMPERATURE, 
    top_p: float = TEXT_TOP_P,
) -> Iterator[str]:
    """Yield tokens one-by-one for streaming UIs."""
    # Ensure model is loaded
    _load(model_name)
    
    # Local import to prevent startup crashes
    from transformers import TextIteratorStreamer
    
    # Create a fresh streamer for EVERY call to prevent "stuck" streams
    streamer = TextIteratorStreamer(
        _pipe.tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True,
    )
    
    inputs = _pipe.tokenizer(prompt, return_tensors="pt").to(_pipe.device)
    
    gen_kwargs = dict(
        **inputs, 
        streamer=streamer, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        do_sample=True,
    )
    
    # Run generation in a separate thread so the UI doesn't freeze
    thread = threading.Thread(target=_pipe.model.generate, kwargs=gen_kwargs)
    thread.start()
    
    accumulated = ""
    for token in streamer:
        accumulated += token
        yield accumulated
    
    thread.join()