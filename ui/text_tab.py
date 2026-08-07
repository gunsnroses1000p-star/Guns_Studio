"""
ui/text_tab.py – Gradio tab for text generation.
"""

import gradio as gr
from providers import text_provider
from config import TEXT_MODEL_DEFAULT, TEXT_MAX_NEW_TOKENS, TEXT_TEMPERATURE, TEXT_TOP_P
from ui.runtime_utils import run_with_ui_error


def build() -> gr.Tab:
    with gr.Tab("✍️ Text Generation") as tab:
        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(
                    label="Model", value=TEXT_MODEL_DEFAULT, lines=1
                )
                prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Enter your prompt…")
                with gr.Row():
                    max_tokens = gr.Slider(
                        64, 2048, step=64, value=TEXT_MAX_NEW_TOKENS, label="Max new tokens"
                    )
                    temperature = gr.Slider(
                        0.01, 2.0, step=0.05, value=TEXT_TEMPERATURE, label="Temperature"
                    )
                    top_p = gr.Slider(
                        0.1, 1.0, step=0.05, value=TEXT_TOP_P, label="Top-p"
                    )
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Output", lines=20)

        def _generate(prompt, model, max_tok, temp, top_p_val):
            return run_with_ui_error(
                "Text Generation",
                lambda: text_provider.generate(
                    prompt,
                    model_name=model,
                    max_new_tokens=int(max_tok),
                    temperature=float(temp),
                    top_p=float(top_p_val),
                ),
                error_result=lambda message: message,
            )

        generate_btn.click(
            _generate,
            inputs=[prompt, model_name, max_tokens, temperature, top_p],
            outputs=output,
        )
    return tab
