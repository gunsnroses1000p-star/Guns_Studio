"""
app.py – Guns AI Studio entrypoint.

Run:
    python app.py

Environment variables (optional):
    HF_TOKEN – Hugging Face access token for gated models
    PORT – server port (default: 7860)
"""

import os

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
    try:
        from ui import (
            text_tab,
            image_tab,
            audio_tab,
            video_tab,
            face_swap_tab,
            upscale_tab,
        )
        from ui.runtime_utils import build_startup_status_markdown
    except Exception as exc:
        print(f"Failed while importing UI modules: {exc!r}", flush=True)
        raise

    with gr.Blocks(title="Guns AI Studio") as demo:
        gr.Markdown(
            """
# 🔫 Guns AI Studio

A multi-modal AI creative suite — text, images, audio, video,
face swap, and upscaling.
"""
        )
        with gr.Accordion("Startup health/status", open=False):
            gr.Markdown(build_startup_status_markdown())

        tab_builders = (
            ("text", text_tab.build),
            ("image", image_tab.build),
            ("audio", audio_tab.build),
            ("video", video_tab.build),
            ("face swap", face_swap_tab.build),
            ("upscale", upscale_tab.build),
        )
        for tab_name, builder in tab_builders:
            print(f"Building {tab_name} tab...", flush=True)
            try:
                builder()
            except Exception as exc:
                print(f"Failed while building {tab_name} tab: {exc!r}", flush=True)
                raise

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