"""
ui/image_tab.py – Gradio tab for image generation.
"""

import gradio as gr
from providers import image_provider
from config import (
    IMAGE_MODEL_DEFAULT,
    IMAGE_STEPS_DEFAULT,
    IMAGE_GUIDANCE_DEFAULT,
    IMAGE_WIDTH_DEFAULT,
    IMAGE_HEIGHT_DEFAULT,
)


def build() -> gr.Tab:
    with gr.Tab("🖼️ Image Generation") as tab:
        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(label="Model", value=IMAGE_MODEL_DEFAULT)
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the image…")
                negative_prompt = gr.Textbox(
                    label="Negative prompt", lines=2, placeholder="What to avoid…"
                )
                with gr.Row():
                    steps = gr.Slider(1, 100, step=1, value=IMAGE_STEPS_DEFAULT, label="Steps")
                    guidance = gr.Slider(
                        1.0, 20.0, step=0.5, value=IMAGE_GUIDANCE_DEFAULT, label="Guidance scale"
                    )
                with gr.Row():
                    width = gr.Slider(256, 2048, step=64, value=IMAGE_WIDTH_DEFAULT, label="Width")
                    height = gr.Slider(
                        256, 2048, step=64, value=IMAGE_HEIGHT_DEFAULT, label="Height"
                    )
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Generated image", type="filepath")

        def _generate(prompt, neg, model, steps_val, guidance_val, w, h, seed_val):
            return image_provider.generate(
                prompt=prompt,
                negative_prompt=neg,
                model_name=model,
                steps=int(steps_val),
                guidance_scale=float(guidance_val),
                width=int(w),
                height=int(h),
                seed=int(seed_val),
            )

        generate_btn.click(
            _generate,
            inputs=[prompt, negative_prompt, model_name, steps, guidance, width, height, seed],
            outputs=output_image,
        )
    return tab
