""" ui/video_tab.py – Gradio tab for text-to-video generation. """
import gradio as gr
from providers import video_provider
from config import VIDEO_MODEL_DEFAULT, VIDEO_FRAMES_DEFAULT, VIDEO_FPS_DEFAULT

def build() -> gr.Tab:
    with gr.Tab("🎬 Video Generation") as tab:
        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(label="Model", value=VIDEO_MODEL_DEFAULT)
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the video…")
                
                with gr.Row():
                    num_frames = gr.Slider(
                        8, 64, step=4, value=VIDEO_FRAMES_DEFAULT, label="Frames"
                    )
                    fps = gr.Slider(4, 30, step=1, value=VIDEO_FPS_DEFAULT, label="FPS")
                
                generate_btn = gr.Button("Generate", variant="primary")
                
            with gr.Column():
                output_video = gr.Video(label="Generated video")

        def _generate(prompt_val, model_val, frames, fps_val):
            # The model will now only load inside the provider when this button is clicked
            return video_provider.generate(
                prompt_val, 
                model_name=model_val, 
                num_frames=int(frames), 
                fps=int(fps_val)
            )

        generate_btn.click(
            _generate, 
            inputs=[prompt, model_name, num_frames, fps], 
            outputs=output_video,
        )
        
    return tab
