import gradio as gr

def hunyuan_ui(minimal=False):
    """
    ULTRA-CLEAN UI for Hunyuan Image-to-Video.
    Technical settings are kept in the backend only.
    """
    with gr.Column():
        # --- VISIBLE USER INTERFACE ---
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    label="Input Image", 
                    type="pil", 
                    interactive=True
                )
                prompt = gr.Textbox(
                    label="Motion Prompt", 
                    placeholder="Describe how the image should move...", 
                    lines=3
                )
                
                generate_btn = gr.Button("Generate Video", variant="primary")

        # --- INVISIBLE BACKEND DEFAULTS ---
        # These exist for the code but are hidden from the user
        seed = gr.Number(value=42, visible=False)
        cfg = gr.Slider(value=7.0, visible=False)
        steps = gr.Slider(value=30, visible=False)
        fps = gr.Slider(value=24, visible=False)
        width = gr.Slider(value=1024, visible=False)
        height = gr.Slider(value=576, visible=False)

        # --- OUTPUT ---
        output_video = gr.Video(
            label="Result", 
            show_label=False, 
            elem_classes="gallery-container"
        )
        
        return {
            "image": image,
            "prompt": prompt,
            "seed": seed,
            "cfg": cfg,
            "steps": steps,
            "fps": fps,
            "width": width,
            "height": height,
            "generate_btn": generate_btn,
            "output": output_video
        }
