"""
ui/extended_video_tab.py

Guns AI Studio - LTX Extended Video tab.
"""

import gradio as gr

from providers.ltx_provider import extend_video


def build():

    gr.Markdown(
        """
        ## ⏩ Extended Video

        Extend an existing video using LTX while keeping the
        final portion of the original video as the motion context.
        """
    )

    with gr.Row():

        with gr.Column():

            input_video = gr.Video(
                label="🎬 Source Video",
                sources=["upload"],
            )

            prompt = gr.Textbox(
                label="💬 Extension Prompt",
                placeholder=(
                    "Example: The person continues walking naturally "
                    "toward the camera while the camera slowly follows."
                ),
                lines=4,
            )

            extension_length = gr.Dropdown(
                choices=[
                    ("~3.4 seconds", 81),
                    ("~6.8 seconds", 161),
                    ("~10.1 seconds", 241),
                ],
                value=81,
                label="⏱️ Extension Length",
            )

            extend_button = gr.Button(
                "⏩ Extend Video",
                variant="primary",
            )

        with gr.Column():

            output_video = gr.Video(
                label="🎥 Extended Video",
                autoplay=False,
            )

    extend_button.click(
        fn=extend_video,
        inputs=[
            input_video,
            prompt,
            extension_length,
        ],
        outputs=output_video,
    )