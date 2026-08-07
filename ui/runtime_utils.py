"""
ui/runtime_utils.py – Lightweight UI runtime helpers.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable, TypeVar

import gradio as gr

from config import FACE_SWAP_MODEL

T = TypeVar("T")


def _format_error_message(tab_name: str, exc: Exception) -> str:
    return f"⚠️ {tab_name} failed: {exc.__class__.__name__}: {exc}"


def run_with_ui_error(
    tab_name: str,
    action: Callable[[], T],
    *,
    error_result: T | Callable[[str], T] | None = None,
) -> T | None:
    try:
        return action()
    except Exception as exc:
        message = _format_error_message(tab_name, exc)
        print(message, flush=True)
        try:
            gr.Warning(message)
        except Exception:
            pass
        if callable(error_result):
            return error_result(message)
        return error_result


def build_startup_status_markdown() -> str:
    lines = ["### 🩺 Startup health", ""]

    has_hf_token = bool(os.environ.get("HF_TOKEN"))
    lines.append(
        f"- {'✅' if has_hf_token else '⚠️'} `HF_TOKEN`: "
        f"{'set' if has_hf_token else 'missing (gated models may fail)'}"
    )

    model_path = Path(__file__).resolve().parent.parent / "models" / FACE_SWAP_MODEL
    has_face_swap_model = model_path.exists()
    lines.append(
        f"- {'✅' if has_face_swap_model else '⚠️'} Face-swap model `{FACE_SWAP_MODEL}`: "
        f"{'found' if has_face_swap_model else f'missing at {model_path}'}"
    )

    provider_modules = (
        "providers.text_provider",
        "providers.image_provider",
        "providers.audio_provider",
        "providers.video_provider",
        "providers.face_swap_provider",
        "providers.upscale_provider",
    )
    for module_name in provider_modules:
        try:
            importlib.import_module(module_name)
            lines.append(f"- ✅ `{module_name}` import: ok")
        except Exception as exc:
            lines.append(
                f"- ⚠️ `{module_name}` import: failed ({exc.__class__.__name__}: {exc})"
            )

    return "\n".join(lines)
