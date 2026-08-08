
Guns N Roses <gunsnroses1000p@gmail.com>
21:33 (0 minutes ago)
to me

"""
ui/img2img_tab.py

Simple user-facing Img2Img interface.

Technical generation settings are intentionally hidden.
"""

import gradio as gr

from providers.img2img_provider import generate_img2img


def build():
    gr.Markdown(
        """
        ## 🖌️ Img2Img
        Transform an existing image using a simple text prompt.
        """
    )

    with gr.Row():

        with gr.Column():

            source_image = gr.Image(
                label="📷 Source Image",
                type="pil",
            )

            prompt = gr.Textbox(
                label="💬 What would you like to change?",
                placeholder=(
                    "Describe how you want the image transformed..."
                ),
                lines=4,
            )

            generate_button = gr.Button(
                "🎨 Generate",
                variant="primary",
            )

        with gr.Column():

            result = gr.Image(
                label="✨ Result",
                type="filepath",
            )

    generate_button.click(
        fn=generate_img2img,
        inputs=[
            source_image,
            prompt,
        ],
        outputs=result,
    )
