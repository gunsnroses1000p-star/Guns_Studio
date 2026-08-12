"""
app.py – Guns AI Studio entrypoint.
"""

import os

import gradio as gr

# ZeroGPU startup compatibility.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_startup_check():
        return None

except ImportError:
    pass


from ui import img2img_tab
from ui import image_to_video_tab
from ui import extended_video_tab


def build_app() -> gr.Blocks:

   CUSTOM_CSS = """
    /* =========================================================
       GUNS AI STUDIO — PURPLE + BLACK TAB UI
       ========================================================= */

    .guns-brand {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px 10px 18px 10px;
        box-sizing: border-box;
    }

    .guns-logo {
        width: 180px;
        height: 180px;
        object-fit: contain;
        display: block;
    }


    /* =========================================================
       THREE-TAB NAVIGATION
       ========================================================= */

    #guns-tabs .tab-nav {
        width: 100%;
        display: flex !important;
        align-items: stretch;
        justify-content: center;

        margin: 8px auto 28px auto;
        padding: 0;

        background: #050505 !important;
        border: 1px solid #2b1645 !important;
        border-radius: 14px;

        overflow: hidden;
    }


    /* =========================================================
       ALL TABS — BLACK
       ========================================================= */

    #guns-tabs .tab-nav button,
    #guns-tabs .tab-nav button[role="tab"] {
        position: relative;

        flex: 1 1 33.333%;

        min-height: 60px;

        margin: 0 !important;
        padding: 8px 8px !important;

        border: 0 !important;
        border-radius: 0 !important;

        background: #050505 !important;
        background-image: none !important;

        color: #ffffff !important;

        font-size: 14px !important;
        font-weight: 600 !important;

        box-shadow: none !important;
        text-shadow: none !important;

        transition:
            background 0.18s ease,
            color 0.18s ease;
    }


    /* =========================================================
       REMOVE GRADIO'S DEFAULT ACTIVE UNDERLINE
       ========================================================= */

    #guns-tabs .tab-nav button::after,
    #guns-tabs .tab-nav button.selected::after,
    #guns-tabs .tab-nav button[aria-selected="true"]::after {
        display: none !important;
        content: none !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }


    /* =========================================================
       ACTIVE TAB — PURPLE
       ========================================================= */

    #guns-tabs .tab-nav button.selected,
    #guns-tabs .tab-nav button[aria-selected="true"] {
        background: #7c3aed !important;
        background-image: none !important;

        color: #ffffff !important;

        border: 0 !important;
        border-bottom: 0 !important;

        box-shadow: none !important;
        text-shadow: none !important;
    }


    /* =========================================================
       HOVER — KEEP IT PURPLE/BLACK ONLY
       ========================================================= */

    #guns-tabs .tab-nav button:hover {
        background: #160d22 !important;
        color: #ffffff !important;
    }

    #guns-tabs .tab-nav button.selected:hover,
    #guns-tabs .tab-nav button[aria-selected="true"]:hover {
        background: #8b5cf6 !important;
        color: #ffffff !important;
    }


    /* =========================================================
       GENERIC SVG ICONS
       ========================================================= */

    #guns-tabs .tab-nav button::before {
        content: "";
        display: inline-block;

        width: 18px;
        height: 18px;

        margin-right: 7px;

        vertical-align: -4px;

        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;

        opacity: 0.95;
    }


    /* Img2Img — image icon */

    #guns-tabs .tab-nav button:nth-child(1)::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='3'/%3E%3Ccircle cx='8.5' cy='8.5' r='1.5'/%3E%3Cpath d='m21 15-5-5L5 21'/%3E%3C/svg%3E");
    }


    /* Image to Video — video/play icon */

    #guns-tabs .tab-nav button:nth-child(2)::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='5' width='15' height='14' rx='2'/%3E%3Cpath d='m18 10 3-2v8l-3-2z'/%3E%3Cpath d='m10 9 4 3-4 3z' fill='%23c084fc' stroke='none'/%3E%3C/svg%3E");
    }


    /* Extended Video — fast-forward icon */

    #guns-tabs .tab-nav button:nth-child(3)::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m5 5 7 7-7 7z'/%3E%3Cpath d='m12 5 7 7-7 7z'/%3E%3C/svg%3E");
    }


    /* Active icons become white */

    #guns-tabs .tab-nav button.selected::before,
    #guns-tabs .tab-nav button[aria-selected="true"]::before {
        filter: brightness(0) invert(1);
    }


    /* =========================================================
       MOBILE
       ========================================================= */

    @media (max-width: 700px) {

        .guns-brand {
            padding: 16px 8px 14px 8px;
        }

        .guns-logo {
            width: 165px;
            height: 165px;
        }

        #guns-tabs .tab-nav {
            margin-top: 6px;
            margin-bottom: 22px;

            border-radius: 12px;
        }

        #guns-tabs .tab-nav button,
        #guns-tabs .tab-nav button[role="tab"] {
            min-height: 58px;

            padding: 7px 3px !important;

            font-size: 12px !important;
            line-height: 1.15 !important;
        }

        #guns-tabs .tab-nav button::before {
            width: 17px;
            height: 17px;

            margin-right: 4px;
        }
    }
    """

    )as demo:

        # ========================================================
        # GUNS AI STUDIO LOGO
        #
        # Plain HTML image — NOT gr.Image.
        # This prevents Gradio Edit / Share controls.
        # ========================================================

    gr.HTML(
            """
            <div class="guns-brand-header">
                <img
                    src="/file=assets/guns_logo.jpg"
                    alt="Guns AI Studio"
                    class="guns-brand-logo"
                />
            </div>
            """
        )

        # ========================================================
        # MAIN TABS
        # ========================================================

    with gr.Tabs():

            # ====================================================
            # IMAGE — IMG2IMG
            # ====================================================

            with gr.Tab("🖌️ Img2Img"):

                img2img_tab.build()

            # ====================================================
            # VIDEO — IMAGE TO VIDEO
            # ====================================================

            with gr.Tab("🎥 Image to Video"):

                image_to_video_tab.build()

            # ====================================================
            # EXTENDED VIDEO — LTX TEST
            # ====================================================

            with gr.Tab("⏩ Extended Video"):

                extended_video_tab.build()

    return demo


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860,
        )
    )

    app = build_app()

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
