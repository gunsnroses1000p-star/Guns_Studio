"""
app.py – Guns AI Studio entrypoint.

Run:
    python app.py

Environment variables (optional):
    HF_TOKEN – Hugging Face access token for gated models
    PORT – server port (default: 7860)
"""

import os
import importlib

# IMPORTANT: Import spaces BEFORE torch/diffusers or anything CUDA-related.
try:
    import spaces
except Exception as exc:
    print(f"Failed while importing spaces: {exc!r}", flush=True)
    raise

try:
    import gradio as gr
except Exception as exc:
    print(f"Failed while importing gradio: {exc!r}", flush=True)
    raise


def build_app() -> gr.Blocks:
    print("Importing UI modules...", flush=True)
    from ui.runtime_utils import build_startup_status_markdown

    tab_modules = (
        ("text", "ui.text_tab"),
        ("image", "ui.image_tab"),
        ("audio", "ui.audio_tab"),
        ("video", "ui.video_tab"),
        ("face swap", "ui.face_swap_tab"),
        ("upscale", "ui.upscale_tab"),
    )
    tab_builders = []
    tab_import_errors = {}
    for tab_name, module_name in tab_modules:
        try:
            module = importlib.import_module(module_name)
            tab_builders.append((tab_name, module.build))
        except Exception as exc:
            print(f"Failed while importing {tab_name} tab module: {exc!r}", flush=True)
            tab_import_errors[tab_name] = exc

    def _render_unavailable_tab(tab_name: str, exc: Exception, reason: str) -> None:
        with gr.Tab(f"⚠️ {tab_name.title()} unavailable"):
            gr.Markdown(
                f"⚠️ **{tab_name.title()}** {reason}: "
                f"`{exc.__class__.__name__}: {exc}`"
            )

    with gr.Blocks(title="Guns AI Studio") as demo:
        gr.Markdown(
            """
# 🔫 Guns AI Studio

A multi-modal AI creative suite — text, images, audio, video,
face swap, and upscaling.
"""
        )
        with gr.Accordion("Startup health/status", open=False):
            gr.Markdown(build_startup_status_markdown(tab_import_errors=tab_import_errors))

        for tab_name, builder in tab_builders:
            print(f"Building {tab_name} tab...", flush=True)
            try:
                builder()
            except Exception as exc:
                print(f"Failed while building {tab_name} tab: {exc!r}", flush=True)
                _render_unavailable_tab(tab_name, exc, "is currently unavailable")
        for tab_name, exc in tab_import_errors.items():
            _render_unavailable_tab(tab_name, exc, "could not be loaded")

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    print(f"Starting Guns AI Studio on port {port}...", flush=True)

    try:
        print("Building Gradio app...", flush=True)
        app = build_app()
        print("Gradio app built successfully.", flush=True)
    except Exception as exc:
        print(f"Startup failed during app construction: {exc!r}", flush=True)
        raise

    try:
        print("Launching Gradio app...", flush=True)
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
        )
    except Exception as exc:
        print(f"Startup failed during app launch: {exc!r}", flush=True)
        raise