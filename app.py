"""
app.py – Guns AI Studio entrypoint.
"""

import os

import gradio as gr

from ui import img2img_tab


def build_app() -> gr.Blocks:

    with gr.Blocks(
        title="Guns AI Studio"
    ) as demo:

        gr.Markdown(
            """
            # 🔫 Guns AI Studio
            """
        )

        with gr.Tabs():

            with gr.Tab("🖌️ Img2Img"):
                img2img_tab.build()

    return demo


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860,
        )
    )

    app = build_app()

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )