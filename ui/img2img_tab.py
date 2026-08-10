import gradio as gr

def img2img_ui(minimal=False):
    """
    ULTRA-CLEAN UI for Img2Img.
    Removed all technical clutter for the average user.
    """
    with gr.Column():
        # --- THE ONLY THINGS THE USER SEES ---
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    label="Input Image", 
                    type="pil", 
                    interactive=True
                )
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="Describe the transformation...", 
                    lines=3
                )
                
                # High-end primary button
                generate_btn = gr.Button("Generate Magic", variant="primary")

        # --- THE INVISIBLE DEFAULTS ---
        # These are NOT displayed in the UI, but they exist as objects 
        # so your backend logic doesn't crash.
        seed = gr.Number(value=42, visible=False)
        cfg = gr.Slider(value=7.5, visible=False)
        strength = gr.Slider(value=0.6, visible=False)
        steps = gr.Slider(value=30, visible=False)
        width = gr.Slider(value=1024, visible=False)
        height = gr.Slider(value=1024, visible=False)

        # --- OUTPUT ---
        output_gallery = gr.Gallery(
            label="Result", 
            show_label=False, 
            elem_classes="gallery-container"
        )
        
        # Return everything so the backend still receives the default values
        return {
            "image": image,
            "prompt": prompt,
            "seed": seed,
            "cfg": cfg,
            "strength": strength,
            "steps": steps,
            "width": width,
            "height": height,
            "generate_btn": generate_btn,
            "output": output_gallery
        }
