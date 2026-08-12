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
       GUNS AI STUDIO — PURPLE + BLACK THEME
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
       Existing Gradio tabs — styling only
       ========================================================= */

    #guns-tabs .tab-nav {
        width: 100%;
        display: flex !important;
        align-items: stretch;
        justify-content: center;

        margin: 8px auto 28px auto;
        padding: 0;

        border: 1px solid #7c3aed;
        border-radius: 16px;

        overflow: hidden;

        background: #000000;
    }


    /* =========================================================
       INDIVIDUAL TABS
       ========================================================= */

    #guns-tabs .tab-nav button {
        flex: 1 1 33.333%;

        min-height: 64px;

        margin: 0 !important;
        padding: 10px 8px !important;

        border: none !important;
        border-radius: 0 !important;

        background: #000000 !important;

        color: #a855f7 !important;

        font-size: 16px !important;
        font-weight: 700 !important;

        transition:
            background 0.2s ease,
            color 0.2s ease,
            box-shadow 0.2s ease;
    }


    /* =========================================================
       HOVER
       ========================================================= */

    #guns-tabs .tab-nav button:hover {
        background: #7c3aed !important;
        color: #000000 !important;
    }


    /* =========================================================
       ACTIVE TAB
       PURPLE BACKGROUND / BLACK TEXT
       ========================================================= */

    #guns-tabs .tab-nav button.selected {
        background: #7c3aed !important;
        color: #000000 !important;

        box-shadow:
            inset 0 -4px 0 #000000;
    }


    /* =========================================================
       SVG-STYLE GENERIC ICONS
       CSS ONLY — DOES NOT CHANGE TAB FUNCTIONALITY
       ========================================================= */

    #guns-tabs .tab-nav button {
        display: flex !important;
        align-items: center;
        justify-content: center;
        gap: 7px;
    }

    /* Hide the original Unicode emoji while keeping the label text */
    #guns-tabs .tab-nav button {
        font-size: 0 !important;
    }

    #guns-tabs .tab-nav button::after {
        font-size: 16px;
        font-weight: 700;
    }

    /* Img2Img */
    #guns-tabs .tab-nav button:nth-child(1)::before {
        content: "";
        width: 22px;
        height: 22px;
        flex: 0 0 22px;

        background-color: currentColor;

        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='3' y='3' width='18' height='18' rx='3' fill='none' stroke='black' stroke-width='2'/%3E%3Ccircle cx='8.5' cy='8.5' r='1.5' fill='black'/%3E%3Cpath d='M4 17l5-5 3 3 2-2 6 6' fill='none' stroke='black' stroke-width='2'/%3E%3C/svg%3E") center / contain no-repeat;

        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='3' y='3' width='18' height='18' rx='3' fill='none' stroke='black' stroke-width='2'/%3E%3Ccircle cx='8.5' cy='8.5' r='1.5' fill='black'/%3E%3Cpath d='M4 17l5-5 3 3 2-2 6 6' fill='none' stroke='black' stroke-width='2'/%3E%3C/svg%3E") center / contain no-repeat;
    }

    /* Image → Video */
    #guns-tabs .tab-nav button:nth-child(2)::before {
        content: "";
        width: 22px;
        height: 22px;
        flex: 0 0 22px;

        background-color: currentColor;

        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='2' y='5' width='14' height='14' rx='2' fill='none' stroke='black' stroke-width='2'/%3E%3Cpath d='M16 9l6-3v12l-6-3z' fill='none' stroke='black' stroke-width='2'/%3E%3C/svg%3E") center / contain no-repeat;

        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='2' y='5' width='14' height='14' rx='2' fill='none' stroke='black' stroke-width='2'/%3E%3Cpath d='M16 9l6-3v12l-6-3z' fill='none' stroke='black' stroke-width='2'/%3E%3C/svg%3E") center / contain no-repeat;
    }

    /* Extended Video */
    #guns-tabs .tab-nav button:nth-child(3)::before {
        content: "";
        width: 22px;
        height: 22px;
        flex: 0 0 22px;

        background-color: currentColor;

        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M5 5l7 7-7 7' fill='none' stroke='black' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M12 5l7 7-7 7' fill='none' stroke='black' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") center / contain no-repeat;

        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M5 5l7 7-7 7' fill='none' stroke='black' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M12 5l7 7-7 7' fill='none' stroke='black' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") center / contain no-repeat;
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
            border-radius: 14px;
        }

        #guns-tabs .tab-nav button {
            min-height: 58px;
            padding: 8px 3px !important;
            gap: 5px;
        }

        #guns-tabs .tab-nav button::before {
            width: 19px;
            height: 19px;
            flex-basis: 19px;
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
