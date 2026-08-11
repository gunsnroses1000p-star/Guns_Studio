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

    with gr.Blocks(
    title="Guns AI Studio",
    css="""
    .guns-brand-header {
        width: 100%;
        max-width: 1250px;
        margin: 0 auto 18px auto;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 4px 0;
    }

    .guns-brand-logo {
        display: block;
        width: 100%;
        max-width: 620px;
        height: auto;
        object-fit: contain;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        user-select: none;
        -webkit-user-drag: none;
    }

    @media (max-width: 768px) {
        .guns-brand-header {
            margin-bottom: 14px;
            padding: 2px 8px;
        }

        .guns-brand-logo {
            max-width: 100%;
        }
    }
    """
) as demo:


        gr.Markdown(
            """
            # 🔫 Guns AI Studio
            """
        )

        with gr.Tabs():

            # =================================================
            # IMG2IMG
            # =================================================

            with gr.Tab("🖌️ Img2Img"):

                img2img_tab.build()


            # =================================================
            # IMAGE TO VIDEO — HUNYUAN
            # =================================================

            with gr.Tab("🎥 Image to Video"):

                image_to_video_tab.build()


            # =================================================
            # EXTENDED VIDEO — LTX
            # =================================================

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