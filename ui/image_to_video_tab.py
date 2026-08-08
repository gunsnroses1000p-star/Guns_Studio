"""
ui/image_to_video_tab.py

Simple user-facing Image → Video interface.
"""

import gradio as gr

from providers.video_provider import generate_video


def build():
    gr.Markdown(
        """
        ## 🎥 Image to Video
        Bring an image to life with natural motion.
        """
    )

    with gr.Row():

        with gr.Column():

            source_image = gr.Image(
                label="📷 Source Image",
                type="pil",
            )

            motion_prompt = gr.Textbox(
                label="💬 Describe the motion",
                placeholder=(
                    "Example: She gently moves her hair "
                    "in the breeze and smiles naturally."
                ),
                lines=4,
            )

            generate_button = gr.Button(
                "🎬 Generate Video",
                variant="primary",
            )

        with gr.Column():

            result = gr.Video(
                label="🎥 Result",
                format="mp4",
            )

    generate_button.click(
        fn=generate_video,
        inputs=[
            source_image,
            motion_prompt,
        ],
        outputs=result,
    )