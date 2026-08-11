"""
app.py – Guns AI Studio entrypoint.
"""

import os
from pathlib import Path

import gradio as gr


# ============================================================
# STATIC ASSETS
# ============================================================

gr.set_static_paths(
    paths=[Path(__file__).parent.absolute() / "assets"]
)


# ============================================================
# ZEROGPU STARTUP COMPATIBILITY
# ============================================================

try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_startup_check():
        return None

except ImportError:
    pass


# ============================================================
# TAB MODULES
# ============================================================

from ui import img2img_tab
from ui import image_to_video_tab
from ui import extended_video_tab


# ============================================================
# CUSTOM UI STYLE
# ============================================================

CUSTOM_CSS = """
/* ============================================================
   GUNS AI STUDIO - MAIN BRAND
   ============================================================ */

.guns-brand {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 28px 20px 20px 20px;
    box-sizing: border-box;
}

.guns-logo {
    width: 210px;
    height: 210px;
    object-fit: contain;
    display: block;
}


/* ============================================================
   THREE-TAB NAVIGATION
   ============================================================ */

#guns-tabs .tab-nav {
    width: 100%;
    display: flex !important;
    align-items: stretch;
    justify-content: center;
    margin: 10px auto 28px auto;
    padding: 0;
    border: 1px solid rgba(180, 80, 255, 0.25);
    border-radius: 18px;
    overflow: hidden;
    background: rgba(10, 8, 18, 0.72);
}

#guns-tabs .tab-nav button {
    flex: 1 1 33.333%;
    min-height: 72px;
    margin: 0 !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: #b8b5c4 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}

#guns-tabs .tab-nav button:hover {
    color: #ffffff !important;
    background: rgba(150, 60, 255, 0.10) !important;
}

#guns-tabs .tab-nav button.selected {
    color: #d678ff !important;
    background: linear-gradient(
        180deg,
        rgba(150, 60, 255, 0.20),
        rgba(80, 20, 120, 0.10)
    ) !important;
    box-shadow:
        inset 0 -3px 0 #c45cff,
        inset 0 0 25px rgba(170, 70, 255, 0.10);
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {

    .guns-brand {
        padding: 20px 10px 12px 10px;
    }

    .guns-logo {
        width: 155px;
        height: 155px;
    }

    #guns-tabs .tab-nav {
        border-radius: 14px;
        margin-top: 8px;
        margin-bottom: 22px;
    }

    #guns-tabs .tab-nav button {
        min-height: 62px;
        padding: 8px 4px !important;
        font-size: 14px !important;
        line-height: 1.2 !important;
    }
}
"""


# ============================================================
# BUILD APP
# ============================================================

def build_app() -> gr.Blocks:

    CUSTOM_CSS = """
    /* =========================================================
       GUNS AI STUDIO LOGO
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
       THREE-TAB BANNER
       ========================================================= */

    #guns-tabs .tab-nav {
        width: 100%;
        display: flex !important;
        align-items: stretch;
        justify-content: center;

        margin: 8px auto 28px auto;
        padding: 0;

        border: 1px solid rgba(180, 80, 255, 0.30);
        border-radius: 16px;

        overflow: hidden;

        background: rgba(12, 10, 20, 0.85);
    }


    /* Individual tabs */

    #guns-tabs .tab-nav button {
        flex: 1 1 33.333%;

        min-height: 68px;

        margin: 0 !important;
        padding: 10px 12px !important;

        border: none !important;
        border-radius: 0 !important;

        background: transparent !important;

        color: #b8b5c4 !important;

        font-size: 17px !important;
        font-weight: 600 !important;

        transition:
            background 0.2s ease,
            color 0.2s ease,
            box-shadow 0.2s ease;
    }


    /* Hover */

    #guns-tabs .tab-nav button:hover {
        color: #ffffff !important;

        background: rgba(150, 60, 255, 0.12) !important;
    }


    /* =========================================================
       ACTIVE TAB = PURPLE
       ========================================================= */

    #guns-tabs .tab-nav button.selected {
        color: #ffffff !important;

        background: linear-gradient(
            180deg,
            rgba(150, 60, 255, 0.48),
            rgba(95, 25, 150, 0.30)
        ) !important;

        box-shadow:
            inset 0 -3px 0 #c45cff,
            inset 0 0 25px rgba(180, 70, 255, 0.18);
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
            min-height: 60px;

            padding: 8px 4px !important;

            font-size: 13px !important;
            line-height: 1.15 !important;
        }
    }
    """


    with gr.Blocks(
        title="Guns AI Studio",
        css=CUSTOM_CSS,
    ) as demo:

        # =====================================================
        # CENTERED LOGO
        # =====================================================

        gr.HTML(
            """
            <div class="guns-brand">
                <img
                    class="guns-logo"
                    src="/gradio_api/file=assets/guns_logo.png"
                    alt="Guns AI Studio"
                >
            </div>
            """
        )


        # =====================================================
        # THREE WORKING TABS
        # =====================================================

        with gr.Tabs(elem_id="guns-tabs"):

            # =================================================
            # IMG2IMG — DO NOT CHANGE
            # =================================================

            with gr.Tab("🖌️ Img2Img"):

                img2img_tab.build()


            # =================================================
            # IMAGE TO VIDEO — DO NOT CHANGE
            # =================================================

            with gr.Tab("🎥 Image to Video"):

                image_to_video_tab.build()


            # =================================================
            # EXTENDED VIDEO — DO NOT CHANGE
            # =================================================

            with gr.Tab("⏩ Extended Video"):

                extended_video_tab.build()


    return demo




# ============================================================
# LAUNCH
# ============================================================

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