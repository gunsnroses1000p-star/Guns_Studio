"""
ui/audio_tab.py – Gradio tab for TTS and music generation.
"""

import gradio as gr
from providers import audio_provider
from config import TTS_MODEL_DEFAULT, MUSIC_MODEL_DEFAULT, MUSIC_DURATION_DEFAULT
from ui.runtime_utils import run_with_ui_error


def build() -> gr.Tab:
    with gr.Tab("🔊 Audio") as tab:
        with gr.Tabs():
            with gr.Tab("Text-to-Speech"):
                with gr.Row():
                    with gr.Column():
                        tts_model = gr.Textbox(label="TTS Model", value=TTS_MODEL_DEFAULT)
                        tts_text = gr.Textbox(
                            label="Text", lines=4, placeholder="Enter text to speak…"
                        )
                        tts_btn = gr.Button("Synthesise", variant="primary")
                    with gr.Column():
                        tts_audio = gr.Audio(label="Speech output", type="filepath")

                tts_btn.click(
                    lambda text, model: run_with_ui_error(
                        "Audio (Text-to-Speech)",
                        lambda: audio_provider.tts(text, model_name=model),
                    ),
                    inputs=[tts_text, tts_model],
                    outputs=tts_audio,
                )

            with gr.Tab("Music Generation"):
                with gr.Row():
                    with gr.Column():
                        music_model = gr.Textbox(label="Music Model", value=MUSIC_MODEL_DEFAULT)
                        music_prompt = gr.Textbox(
                            label="Prompt", lines=3, placeholder="Describe the music…"
                        )
                        music_duration = gr.Slider(
                            5, 120, step=5, value=MUSIC_DURATION_DEFAULT, label="Duration (s)"
                        )
                        music_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        music_audio = gr.Audio(label="Music output", type="filepath")

                music_btn.click(
                    lambda prompt, model, dur: run_with_ui_error(
                        "Audio (Music Generation)",
                        lambda: audio_provider.generate_music(
                            prompt, model_name=model, duration=int(dur)
                        ),
                    ),
                    inputs=[music_prompt, music_model, music_duration],
                    outputs=music_audio,
                )
    return tab
