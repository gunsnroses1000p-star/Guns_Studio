"""
app.py – Guns AI Studio entrypoint.
"""

import os

import gradio as gr

# ZeroGPU startup compatibility.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_startup_check():
        return None

except ImportError:
    pass


from ui import img2img_tab
from ui import image_to_video_tab
from ui import extended_video_tab


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

            # =================================================
            # IMG2IMG
            # =================================================

            with gr.Tab("🖌️ Img2Img"):

                img2img_tab.build()


            # =================================================
            # IMAGE TO VIDEO — HUNYUAN
            # =================================================

            with gr.Tab("🎥 Image to Video"):

                image_to_video_tab.build()


            # =================================================
            # EXTENDED VIDEO — LTX
            # =================================================

            with gr.Tab("⏩ Extended Video"):

                extended_video_tab.build()


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