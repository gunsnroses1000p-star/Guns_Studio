"""
app.py – Guns AI Studio entrypoint.
"""

import os

import gradio as gr

from ui import img2img_tab
from ui import image_to_video_tab
from providers.wan_lightning_test import test_wan_lightning

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

            with gr.Tab("🎥 Image to Video"):
                image_to_video_tab.build()
        with gr.Tab("🧪 Wan Lightning Test"):

            gr.Markdown(
        """
        ## 🧪 Wan 2.2 Lightning Test
        Temporary test tab — this will be removed after the audition.
        """
    )

    with gr.Row():

        with gr.Column():

            test_image = gr.Image(
                label="📷 Source Image",
                type="pil",
            )

            test_prompt = gr.Textbox(
                label="💬 Motion Prompt",
                placeholder="Describe the motion...",
                lines=4,
            )

            test_button = gr.Button(
                "🚀 Test Wan Lightning",
                variant="primary",
            )

        with gr.Column():

            test_result = gr.Video(
                label="🎥 Wan Lightning Result",
                format="mp4",
            )

    test_button.click(
        fn=test_wan_lightning,
        inputs=[
            test_image,
            test_prompt,
        ],
        outputs=test_result,
    )
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