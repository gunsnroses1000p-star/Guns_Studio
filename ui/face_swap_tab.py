"""
ui/face_swap_tab.py – Gradio tab for face swapping.
"""

import gradio as gr
from providers import face_swap_provider
from ui.runtime_utils import run_with_ui_error


def build() -> gr.Tab:
    with gr.Tab("🎭 Face Swap") as tab:
        gr.Markdown(
            "Upload a **source** image (the face you want to use) and a "
            "**target** image (where the face will be placed)."
        )
        with gr.Row():
            with gr.Column():
                source_img = gr.Image(label="Source face", type="filepath")
                target_img = gr.Image(label="Target image", type="filepath")
                swap_btn = gr.Button("Swap Faces", variant="primary")
            with gr.Column():
                result_img = gr.Image(label="Result", type="filepath")

        swap_btn.click(
            lambda source, target: run_with_ui_error(
                "Face Swap",
                lambda: face_swap_provider.swap(source, target),
            ),
            inputs=[source_img, target_img],
            outputs=result_img,
        )
    return tab
