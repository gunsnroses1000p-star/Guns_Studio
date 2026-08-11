import gradio as gr
import spaces

# ==============================================================================
# 1. LUXURY "EYE CANDY" CSS (Tuned to your logo)
# ==============================================================================
custom_css = """
body, .gradio-container {
    background-color: #0a0a0c !important; 
    color: #e0e0e0 !important;
    font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

.tabs {
    border: none !important;
    gap: 15px !important;
}

.tab-nav {
    border-bottom: 1px solid rgba(138, 43, 226, 0.3) !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: #a855f7 !important; 
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}

.tab-nav button.selected {
    color: #d8b4fe !important; 
    border-bottom: 3px solid #a855f7 !important;
    text-shadow: 0px 0px 15px rgba(168, 85, 247, 0.8) !important;
}

.tab-icon {
    width: 20px;
    height: 20px;
    vertical-align: middle;
    margin-right: 8px;
    fill: #a855f7;
    filter: drop-shadow(0px 0px 3px rgba(168, 85, 247, 0.8));
}

.gr-box, .gr-form {
    background: rgba(10, 10, 15, 0.8) !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.9) !important;
}

#studio-logo {
    filter: drop-shadow(0px 0px 20px rgba(168, 85, 247, 0.4));
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

#studio-logo:hover {
    transform: scale(1.08);
    filter: drop-shadow(0px 0px 30px rgba(168, 85, 247, 0.7));
}

button.primary {
    background: linear-gradient(135deg, #7c3aed 0%, #c084fc 50%, #7c3aed 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.5) !important;
    text-transform: uppercase !important;
}

button.primary:hover {
    box-shadow: 0 0 25px rgba(168, 85, 247, 0.8) !important;
    transform: translateY(-3px) !important;
    filter: brightness(1.2) !important;
}

label {
    color: #d8b4fe !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
"""

# ==============================================================================
# SVG ICON DEFINITIONS
# ==============================================================================
ICON_SPARKLE = '<svg class="tab-icon" viewBox="0 0 24 24"><path d="M12,2L14.5,9L22,11.5L14.5,14L12,21L9.5,14L2,11.5L9.5,9L12,2Z"/></svg>'
ICON_MOVIE = '<svg class="tab-icon" viewBox="0 0 24 24"><path d="M18,17V11H6V17H18M18,19H6V21H18V19M18,9V7H6V9H18M4,5H2V19H4V5Z"/></svg>'
ICON_ROCKET = '<svg class="tab-icon" viewBox="0 0 24 24"><path d="M13.13,16.57L11.45,17.7L9.87,16.57L11.55,15.4L13.13,16.57M17.7,12.5L16.5,11.3L17.7,10.1L18.9,11.3L17.7,12.5M5.6,12.5L4.4,11.3L5.6,10.1L6.8,11.3L5.6,12.5M12,2L14,8L18,10L14,12L12,18L10,12L6,10L10,8L12,2Z"/></svg>'

# ==============================================================================
# 2. BACKEND PROCESSING FUNCTIONS
# ==============================================================================

@spaces.GPU
def img2img_process(image, prompt, seed, cfg, strength, steps, width, height):
    try:
        from ui.img2img import img2img_process as process_logic
        return process_logic(image, prompt, seed, cfg, strength, steps, width, height)
    except Exception as e:
        return f"Error: {str(e)}"

@spaces.GPU
def hunyuan_process(image, prompt, seed, cfg, steps, fps, width, height):
    try:
        from ui.hunyuan import hunyuan_process as process_logic
        return process_logic(image, prompt, seed, cfg, steps, fps, width, height)
    except Exception as e:
        return f"Error: {str(e)}"

@spaces.GPU
def ltx_process(image, prompt, seed, cfg, steps, guidance, width, height):
    try:
        from ui.ltx import ltx_process as process_logic
        return process_logic(image, prompt, seed, cfg, steps, guidance, width, height)
    except Exception as e:
        return f"Error: {str(e)}"

# ==============================================================================
# 3. MAIN APP BUILDING
# ==============================================================================

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    # --- Header ---
    with gr.Row(elem_id="header"):
        with gr.Column(scale=1):
            gr.Image("logo.png", elem_id="studio-logo", show_label=False, interactive=False, width=150)
        
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # <span style='color: #a855f7;'>GUNS AI STUDIO</span>
                <p style='color: #888; font-size: 1.2rem; letter-spacing: 1px;'>IMAGINE • CREATE • INNOVATE</p>
                """
            )

    with gr.Tabs():
        # Tab 1: Img2Img
        with gr.TabItem(f"{ICON_SPARKLE} Img2Img"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        i2i_img = gr.Image(label="Input Image", type="pil")
                        i2i_prompt = gr.Textbox(label="Prompt", placeholder="Describe the transformation...", lines=3)
                        i2i_btn = gr.Button("Generate Magic", variant="primary")
                
                i2i_seed = gr.Number(value=42, visible=False)
                i2i_cfg = gr.Slider(value=7.5, visible=False)
                i2i_strength = gr.Slider(value=0.6, visible=False)
                i2i_steps = gr.Slider(value=30, visible=False)
                i2i_width = gr.Slider(value=1024, visible=False)
                i2i_height = gr.Slider(value=1024, visible=False)
                
                i2i_output = gr.Gallery(label="Result", show_label=False, elem_classes="gallery-container")
                
                i2i_btn.click(
                    fn=img2img_process,
                    inputs=[i2i_img, i2i_prompt, i2i_seed, i2i_cfg, i2i_strength, i2i_steps, i2i_width, i2i_height],
                    outputs=i2i_output
                )

        # Tab 2: Image to Video (Hunyuan)
        with gr.TabItem(f"{ICON_MOVIE} Image to Video"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        h_img = gr.Image(label="Input Image", type="pil")
                        h_prompt = gr.Textbox(label="Motion Prompt", placeholder="Describe movement...", lines=3)
                        h_btn = gr.Button("Generate Video", variant="primary")
                
                h_seed = gr.Number(value=42, visible=False)
                h_cfg = gr.Slider(value=7.0, visible=False)
                h_steps = gr.Slider(value=30, visible=False)
                h_fps = gr.Slider(value=24, visible=False)
                h_width = gr.Slider(value=1024, visible=False)
                h_height = gr.Slider(value=576, visible=False)
                
                h_output = gr.Video(label="Result", show_label=False, elem_classes="gallery-container")
                
                h_btn.click(
                    fn=hunyuan_process,
                    inputs=[h_img, h_prompt, h_seed, h_cfg, h_steps, h_fps, h_width, h_height],
                    outputs=h_output
                )

        # Tab 3: Extended Video (LTX)
        with gr.TabItem(f"{ICON_ROCKET} Extended Video"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        l_img = gr.Image(label="Reference Image", type="pil")
                        l_prompt = gr.Textbox(label="Video Prompt", placeholder="What happens in this scene?", lines=3)
                        l_btn = gr.Button("Extend Video", variant="primary")
                
                l_seed = gr.Number(value=42, visible=False)
                l_cfg = gr.Slider(value=3.0, visible=False)
                l_steps = gr.Slider(value=20, visible=False)
                l_guidance = gr.Slider(value=1.0, visible=False)
                l_width = gr.Slider(value=768, visible=False)
                l_height = gr.Slider(value=512, visible=False)
                
                l_output = gr.Video(label="Result", show_label=False, elem_classes="gallery-container")
                
                l_btn.click(
                    fn=ltx_process,
                    inputs=[l_img, l_prompt, l_seed, l_cfg, l_steps, l_guidance, l_width, l_height],
                    outputs=l_output
                )

if __name__ == "__main__":
    demo.launch()
