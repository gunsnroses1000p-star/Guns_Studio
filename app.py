import gradio as gr
from ui.img2img import img2img_ui
from ui.hunyuan import hunyuan_ui
from ui.ltx import ltx_ui

# --- CUSTOM LUXURY CSS ---
# This CSS is specifically tuned to match your Royal Purple and Silver Metallic logo.
custom_css = """
/* General Background & Text */
body, .gradio-container {
    background-color: #0a0a0c !important; 
    color: #e0e0e0 !important;
    font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

/* Tab Styling - Matching the Royal Neon Purple of the logo */
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

/* Container Polish - Deeper blacks and purple accents */
.gr-box, .gr-form {
    background: rgba(10, 10, 15, 0.8) !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.9) !important;
}

/* Logo Area - Soft glow to blend with the dark background */
#studio-logo {
    filter: drop-shadow(0px 0px 20px rgba(168, 85, 247, 0.4));
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

#studio-logo:hover {
    transform: scale(1.08);
    filter: drop-shadow(0px 0px 30px rgba(168, 85, 247, 0.7));
}

/* Button Styling - "Silver-Purple" Metallic Gradient */
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

/* Removing technical labels and borders */
label {
    color: #d8b4fe !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    # --- Header Section ---
    with gr.Row(elem_id="header"):
        with gr.Column(scale=1):
            # Replace 'logo.png' with the actual filename of your uploaded logo
            gr.Image("logo.png", elem_id="studio-logo", show_label=False, interactive=False, width=150)
        
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # <span style='color: #a855f7;'>GUNS AI STUDIO</span>
                <p style='color: #888; font-size: 1.2rem; letter-spacing: 1px;'>IMAGINE • CREATE • INNOVATE</p>
                """
            )

    # --- Main Studio Tabs ---
    with gr.Tabs():
        # Tab 1: Img2Img
        with gr.TabItem("✨ Img2Img"):
            with gr.Row():
                with gr.Column():
                    # 'minimal=True' tells your UI files to hide the seeds/CFG/etc.
                    img2img_ui(minimal=True) 

        # Tab 2: Image to Video
        with gr.TabItem("🎬 Image to Video"):
            with gr.Row():
                with gr.Column():
                    hunyuan_ui(minimal=True)

        # Tab 3: Extended Video
        with gr.TabItem("🚀 Extended Video"):
            with gr.Row():
                with gr.Column():
                    ltx_ui(minimal=True)

if __name__ == "__main__":
    demo.launch()
