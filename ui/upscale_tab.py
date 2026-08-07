"""
ui/upscale_tab.py – Gradio tab for image upscaling.
"""

import gradio as gr
from providers import upscale_provider
from config import UPSCALE_MODEL_DEFAULT, UPSCALE_STEPS_DEFAULT
from ui.runtime_utils import run_with_ui_error


def build() -> gr.Tab:
    with gr.Tab("🔍 Upscale") as tab:
        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(label="Model", value=UPSCALE_MODEL_DEFAULT)
                input_image = gr.Image(label="Input image", type="filepath")
                prompt = gr.Textbox(
                    label="Optional prompt", lines=2, placeholder="Describe the image (optional)…"
                )
                steps = gr.Slider(10, 75, step=5, value=UPSCALE_STEPS_DEFAULT, label="Steps")
                upscale_btn = gr.Button("Upscale 4×", variant="primary")
            with gr.Column():
                result_image = gr.Image(label="Upscaled image", type="filepath")

        def _upscale(image, prompt_val, model, steps_val):
            return run_with_ui_error(
                "Upscale",
                lambda: upscale_provider.upscale(
                    image, prompt=prompt_val, model_name=model, steps=int(steps_val)
                ),
            )

        upscale_btn.click(
            _upscale,
            inputs=[input_image, prompt, model_name, steps],
            outputs=result_image,
        )
    return tab
