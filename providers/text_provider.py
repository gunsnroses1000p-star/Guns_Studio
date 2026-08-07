"""
providers/text_provider.py – Text generation via a causal-LM pipeline.
"""

from __future__ import annotations

from typing import Iterator

from transformers import pipeline, TextIteratorStreamer
import threading

from config import (
    HF_TOKEN,
    TEXT_MODEL_DEFAULT,
    TEXT_MAX_NEW_TOKENS,
    TEXT_TEMPERATURE,
    TEXT_TOP_P,
)

_pipe = None
_loaded_model: str = ""


def _load(model_name: str) -> None:
    global _pipe, _loaded_model
    if _loaded_model == model_name and _pipe is not None:
        return
    _pipe = pipeline(
        "text-generation",
        model=model_name,
        token=HF_TOKEN or None,
        device_map="auto",
    )
    _loaded_model = model_name


def generate(
    prompt: str,
    model_name: str = TEXT_MODEL_DEFAULT,
    max_new_tokens: int = TEXT_MAX_NEW_TOKENS,
    temperature: float = TEXT_TEMPERATURE,
    top_p: float = TEXT_TOP_P,
) -> str:
    """Return generated text for *prompt*."""
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
    _load(model_name)
    tokenizer = _pipe.tokenizer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(_pipe.device)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    thread = threading.Thread(target=_pipe.model.generate, kwargs=gen_kwargs)
    thread.start()
    accumulated = ""
    for token in streamer:
        accumulated += token
        yield accumulated
    thread.join()
