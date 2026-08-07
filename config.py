"""
config.py — Environment variables, constants, and custom CSS/theme.
"""

import os

# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================
HF_TOKEN = os.getenv("HF_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
FAL_KEY = os.getenv("FAL_KEY")
CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY")
PRIVATE_SERVER_URL = os.getenv("PRIVATE_SERVER_URL", "")

RUNPOD_BASE = "https://api.runpod.ai/v2"

# =========================================================
# APP CONFIGURATION
# =========================================================
DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_NEGATIVE = "blurry, distorted, low quality, deformed, ugly"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 28
DEFAULT_SEED = 0

# LoRA defaults
DEFAULT_LORA_NAMES: list[str] = []
LORA_URL = os.getenv("LORA_URL", "")

# =========================================================
# CUSTOM CSS / THEME
# =========================================================
CUSTOM_CSS = """
body {
  background-color: #000000 !important;
  overflow-x: hidden !important;
}
.gradio-container {
  background-color: #000000 !important;
  max-width: 1400px !important;
  margin: 0 auto !important;
  padding-left: 24px !important;
  padding-right: 24px !important;
}

/* MOBILE FIX */
@media (max-width: 768px) {
  .gradio-container {
    max-width: 100% !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    margin: 0 auto !important;
  }
}

footer {visibility: hidden}

.gr-button-primary {
  background: linear-gradient(45deg, #6b21a8, #a855f7) !important;
  border: none !important;
  color: white !important;
  font-weight: bold !important;
}
.gr-button-primary:hover {
  box-shadow: 0px 0px 15px #a855f7 !important;
}

/* TABS / NAVIGATION */
.tab-nav {
  gap: 8px !important;
  padding: 6px 0px 14px 0px !important;
  overflow-x: auto !important;
  flex-wrap: nowrap !important;
  scrollbar-width: none;
}
.tab-nav::-webkit-scrollbar { display: none; }
.tab-nav button {
  flex: 0 0 auto !important;
  min-width: auto !important;
  padding: 10px 16px !important;
  border-radius: 999px !important;
  border: 1px solid var(--guns-border) !important;
  background: var(--guns-surface) !important;
  color: var(--guns-muted) !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  box-shadow: none !important;
}

/* PANELS / CARDS */
.block, .form, .panel {
  border-radius: 18px !important;
  border: 1px solid var(--guns-border) !important;
  background: rgba(18, 18, 25, 0.88) !important;
  box-shadow: none !important;
}

/* INPUTS */
input, textarea {
  background: var(--guns-surface-2) !important;
  color: var(--guns-text) !important;
  border-color: var(--guns-border) !important;
  border-radius: 12px !important;
}
textarea:focus, input:focus {
  border-color: var(--guns-violet) !important;
  box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.15) !important;
}

/* BUTTONS */
button.primary, button.lg {
  min-height: 48px !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, #7c3aed, #a855f7, #d946ef) !important;
  color: white !important;
  border: none !important;
  font-weight: 800 !important;
  box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25) !important;
}

/* IMAGE / VIDEO UPLOAD AREAS */
.image-container, .video-container {
  border-radius: 18px !important;
  overflow: hidden !important;
}

/* MOBILE */
@media (max-width: 700px) {
  .gradio-container { padding: 12px !important; }
  .guns-header { padding-top: 18px; }
  .guns-brand { font-size: 30px; }
  .tab-nav button { padding: 9px 14px !important; font-size: 13px !important; }
  .gr-row { gap: 10px !important; }
}

/* GUNS AI — PREMIUM TOOL WORKSPACE */
.tabitem { padding-top: 14px !important; }

.block {
  padding: 18px !important;
  margin-bottom: 14px !important;
  background: linear-gradient(145deg, rgba(20,20,29,0.96), rgba(10,10,16,0.96)) !important;
  border: 1px solid rgba(168, 85, 247, 0.12) !important;
  box-shadow: 0 12px 35px rgba(0,0,0,0.22) !important;
}

label span {
  color: #d8d8e3 !important;
  font-weight: 650 !important;
  letter-spacing: 0.1px;
}

input, textarea {
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
}

.wrap, .secondary-wrap { border-radius: 12px !important; }

.image-container, .video-container {
  background: linear-gradient(145deg, rgba(18,18,27,0.95), rgba(8,8,14,0.95)) !important;
  border: 1px solid rgba(168, 85, 247, 0.18) !important;
  min-height: 220px;
}

button.primary, button.lg {
  position: relative !important;
  overflow: hidden !important;
  letter-spacing: 0.2px !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
button.primary:hover, button.lg:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 14px 38px rgba(124, 58, 237, 0.38) !important;
}

.output-class, [data-testid="output"] {
  border: 1px solid rgba(217, 70, 239, 0.15) !important;
  box-shadow: inset 0 0 30px rgba(124, 58, 237, 0.035) !important;
}

input[type="range"] { accent-color: var(--guns-violet) !important; }

* { scrollbar-color: var(--guns-purple) var(--guns-surface); }

@media (max-width: 700px) {
  .tabitem { padding-top: 10px !important; }
  .block { padding: 14px !important; border-radius: 16px !important; margin-bottom: 12px !important; }
  .image-container, .video-container { min-height: 200px; }
  button.primary, button.lg { width: 100% !important; min-height: 52px !important; }
}

/* IMG2IMG MOBILE WIDTH FIX */
@media (max-width: 700px) {
  .gradio-container { width: 100% !important; max-width: 100% !important; overflow-x: hidden !important; }
  .tabitem { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; overflow-x: hidden !important; }
  .tabitem > div { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
  .guns-tool-header { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; overflow: hidden !important; }
  .guns-tool-title, .guns-tool-subtitle { max-width: 100% !important; white-space: normal !important; overflow-wrap: anywhere !important; word-break: normal !important; }
  .gradio-row { width: 100% !important; max-width: 100% !important; min-width: 0 !important; flex-wrap: wrap !important; }
  .gradio-row > *, .gradio-column { min-width: 0 !important; max-width: 100% !important; flex: 1 1 100% !important; }
}

@media screen and (max-width: 700px) {
  html, body { width: 100% !important; max-width: 100vw !important; margin: 0 !important; padding: 0 !important; overflow-x: hidden !important; }
  .gradio-container { width: 100vw !important; max-width: 100vw !important; min-width: 0 !important; margin-left: 0 !important; margin-right: 0 !important; padding-left: 12px !important; padding-right: 12px !important; box-sizing: border-box !important; overflow-x: hidden !important; }
  .contain { width: 100% !important; max-width: 100% !important; min-width: 0 !important; margin-left: 0 !important; margin-right: 0 !important; box-sizing: border-box !important; }
  .tabs, .tabitem, .block, .form { width: 100% !important; max-width: 100% !important; min-width: 0 !important; margin-left: 0 !important; margin-right: 0 !important; box-sizing: border-box !important; }
  .guns-tool-header { width: 100% !important; max-width: 100% !important; min-width: 0 !important; margin-left: 0 !important; margin-right: 0 !important; box-sizing: border-box !important; }
}

* { box-sizing: border-box; }
html, body { width: 100%; max-width: 100%; overflow-x: hidden !important; }
.gradio-container { width: 100% !important; max-width: 1100px !important; margin: 0 auto !important; overflow-x: hidden !important; }

.gradio-container .row { min-width: 0 !important; max-width: 100% !important; }
.gradio-container .column { min-width: 0 !important; max-width: 100% !important; }

.gradio-container img,
.gradio-container video,
.gradio-container canvas,
.gradio-container input,
.gradio-container textarea,
.gradio-container select { max-width: 100% !important; }

@media (max-width: 768px) {
  .gradio-container { width: 100% !important; max-width: 100% !important; padding-left: 10px !important; padding-right: 10px !important; }
  .guns-header { padding: 18px 4px 16px 4px !important; }
  .guns-brand { font-size: 27px !important; }
  .tab-nav { width: 100% !important; max-width: 100% !important; overflow-x: auto !important; }
  .block, .form, .panel { min-width: 0 !important; max-width: 100% !important; }
}

/* GUNS AI STUDIO HERO */
#hero-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin: 25px 0 35px 0;
  text-align: center;
}
#guns-logo { max-width: 320px !important; margin: 0 auto 18px auto !important; }
#guns-logo img {
  border-radius: 22px !important;
  box-shadow: 0 0 28px rgba(168, 85, 247, 0.75) !important;
  transition: all 0.25s ease;
}
#guns-logo img:hover {
  transform: scale(1.02);
  box-shadow: 0 0 40px rgba(168, 85, 247, 0.95) !important;
}
#hero-section h1 {
  color: #c084fc !important;
  font-size: 52px !important;
  font-weight: 800 !important;
  margin: 8px 0 12px 0 !important;
  letter-spacing: 2px;
}
#hero-section h3 {
  color: #cfcfcf !important;
  font-style: italic;
  font-weight: 400;
  letter-spacing: 3px;
  margin-top: 0 !important;
  opacity: .9;
}
@media (max-width:768px){
  #guns-logo{ max-width:240px !important; }
  #hero-section h1{ font-size:40px !important; }
  #hero-section h3{ font-size:22px !important; letter-spacing:2px; }
}
"""
