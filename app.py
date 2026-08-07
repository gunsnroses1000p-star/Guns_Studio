"""
app.py – Guns AI Studio entrypoint.

Run:
    python app.py

Environment variables (optional):
    HF_TOKEN   – Hugging Face access token for gated models
    PORT       – server port (default: 7860)
"""

import os
import gradio as gr

from ui import text_tab, image_tab, audio_tab, video_tab, face_swap_tab, upscale_tab


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Guns AI Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔫 Guns AI Studio
            A multi-modal AI creative suite — text, images, audio, video, face swap, and upscaling.
            """
        )
        text_tab.build()
        image_tab.build()
        audio_tab.build()
        video_tab.build()
        face_swap_tab.build()
        upscale_tab.build()
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app = build_app()
    app.launch(server_port=port, share=False)
