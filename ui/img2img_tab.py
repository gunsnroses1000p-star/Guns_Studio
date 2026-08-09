"""
ui/img2img_tab.py

Guns AI Studio — Img2Img interface.

The generation backend is intentionally untouched.
This file controls the user-facing presentation only.
"""

import gradio as gr

from providers.img2img_provider import generate_img2img


def build():
    # ============================================================
    # GUNS AI STUDIO — IMG2IMG VISUAL THEME
    # ============================================================

    gr.HTML(
        """
        <style>

        /* ========================================================
           GUNS IMG2IMG — ROOT
           ======================================================== */

        .guns-img2img {
            max-width: 1250px !important;
            margin: 0 auto !important;
        }

        /* ========================================================
           CUSTOM ICONS
           ======================================================== */

        .guns-icon {
            width: 22px;
            height: 22px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 9px;
            filter:
                drop-shadow(0 0 5px rgba(168, 85, 247, 0.55));
        }

        .guns-icon-lg {
            width: 30px;
            height: 30px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
            filter:
                drop-shadow(0 0 8px rgba(168, 85, 247, 0.65));
        }

        /* ========================================================
           HERO
           ======================================================== */

        .guns-img2img-hero {
            padding: 8px 4px 22px 4px;
        }

        .guns-img2img-title {
            margin: 0 !important;
            font-size: 30px !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px !important;
            color: #f4f0ff !important;
        }

        .guns-img2img-subtitle {
            margin-top: 6px !important;
            color: #aaa3b8 !important;
            font-size: 14px !important;
        }

        /* ========================================================
           STUDIO CARDS
           ======================================================== */

        .guns-panel {
            border: 1px solid rgba(168, 85, 247, 0.25) !important;
            border-radius: 18px !important;
            padding: 18px !important;
            background:
                linear-gradient(
                    145deg,
                    rgba(30, 24, 38, 0.96),
                    rgba(13, 12, 16, 0.98)
                ) !important;
            box-shadow:
                0 8px 30px rgba(0, 0, 0, 0.35),
                inset 0 1px 0 rgba(255,255,255,0.025) !important;
        }

        .guns-panel:hover {
            border-color: rgba(168, 85, 247, 0.40) !important;
        }

        .guns-panel-title {
            font-size: 14px !important;
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
            color: #d9c7ff !important;
            margin-bottom: 14px !important;
        }

        .guns-panel-title span {
            color: #ffffff !important;
        }

        /* ========================================================
           IMAGE COMPONENTS
           ======================================================== */

        .guns-image-box {
            border-radius: 14px !important;
            overflow: hidden !important;
            border: 1px solid rgba(168, 85, 247, 0.22) !important;
            background: #0b0a0e !important;
        }

        .guns-image-box:hover {
            border-color: rgba(168, 85, 247, 0.50) !important;
            box-shadow:
                0 0 22px rgba(168, 85, 247, 0.10) !important;
        }

        /* ========================================================
           TEXTBOX
           ======================================================== */

        .guns-prompt textarea {
            border-radius: 12px !important;
            border: 1px solid rgba(168, 85, 247, 0.20) !important;
            background: rgba(8, 7, 10, 0.85) !important;
            color: #f5f2fa !important;
        }

        .guns-prompt textarea:focus {
            border-color: rgba(168, 85, 247, 0.65) !important;
            box-shadow:
                0 0 0 1px rgba(168, 85, 247, 0.20),
                0 0 18px rgba(168, 85, 247, 0.08) !important;
        }

        .guns-prompt textarea::placeholder {
            color: #77717f !important;
        }

        /* ========================================================
           GENERATE BUTTON
           ======================================================== */

        .guns-generate {
            margin-top: 14px !important;
        }

        .guns-generate button {
            min-height: 52px !important;
            border-radius: 13px !important;
            border: 1px solid rgba(198, 146, 255, 0.55) !important;
            background:
                linear-gradient(
                    135deg,
                    #7c3aed,
                    #a855f7
                ) !important;
            color: white !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            letter-spacing: 0.4px !important;
            box-shadow:
                0 7px 24px rgba(124, 58, 237, 0.28) !important;
            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                filter 0.15s ease !important;
        }

        .guns-generate button:hover {
            transform: translateY(-1px) !important;
            filter: brightness(1.08) !important;
            box-shadow:
                0 10px 30px rgba(168, 85, 247, 0.38) !important;
        }

        .guns-generate button:active {
            transform: translateY(0) !important;
        }

        /* ========================================================
           OUTPUT AREA
           ======================================================== */

        .guns-result {
            min-height: 430px !important;
        }

        /* ========================================================
           SMALL STATUS STRIP
           ======================================================== */

        .guns-status {
            margin-top: 14px;
            padding: 10px 13px;
            border-radius: 10px;
            border: 1px solid rgba(168, 85, 247, 0.14);
            background: rgba(168, 85, 247, 0.035);
            color: #8f8898;
            font-size: 12px;
            text-align: center;
        }

        /* ========================================================
           MOBILE
           ======================================================== */

        @media (max-width: 768px) {

            .guns-img2img-title {
                font-size: 25px !important;
            }

            .guns-panel {
                padding: 13px !important;
                border-radius: 15px !important;
            }

            .guns-result {
                min-height: 300px !important;
            }

            .guns-generate button {
                min-height: 50px !important;
            }
        }

        </style>
        """
    )

    # ============================================================
    # MAIN CONTAINER
    # ============================================================

    with gr.Column(elem_classes=["guns-img2img"]):

        # ========================================================
        # HERO
        # ========================================================

        gr.HTML(
            """
            <div class="guns-img2img-hero">
                <div class="guns-img2img-title">
                    <svg class="guns-icon-lg"
                         viewBox="0 0 24 24"
                         fill="none"
                         xmlns="http://www.w3.org/2000/svg">
                        <rect x="3" y="4" width="18" height="16"
                              rx="3"
                              stroke="#c084fc"
                              stroke-width="1.8"/>
                        <circle cx="8.5" cy="9"
                                r="1.5"
                                fill="#c084fc"/>
                        <path d="M5.5 17L10 12.5L13 15L15.5 12.5L19 17"
                              stroke="#c084fc"
                              stroke-width="1.8"
                              stroke-linecap="round"
                              stroke-linejoin="round"/>
                    </svg>
                    Img2Img
                </div>

                <div class="guns-img2img-subtitle">
                    Transform an existing image with natural,
                    photorealistic AI editing.
                </div>
            </div>
            """
        )

        # ========================================================
        # INPUT / OUTPUT
        # ========================================================

        with gr.Row(equal_height=True):

            # ----------------------------------------------------
            # INPUT PANEL
            # ----------------------------------------------------

            with gr.Column(
                scale=1,
                elem_classes=["guns-panel"],
            ):

                gr.HTML(
                    """
                    <div class="guns-panel-title">
                        <svg class="guns-icon"
                             viewBox="0 0 24 24"
                             fill="none"
                             xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 3V15"
                                  stroke="#c084fc"
                                  stroke-width="2"
                                  stroke-linecap="round"/>
                            <path d="M7 8L12 3L17 8"
                                  stroke="#c084fc"
                                  stroke-width="2"
                                  stroke-linecap="round"
                                  stroke-linejoin="round"/>
                            <path d="M5 15V19C5 20.1 5.9 21 7 21H17C18.1 21 19 20.1 19 19V15"
                                  stroke="#c084fc"
                                  stroke-width="2"
                                  stroke-linecap="round"/>
                        </svg>
                        <span>Input</span>
                    </div>
                    """
                )

                source_image = gr.Image(
                    label="Source Image",
                    type="pil",
                    elem_classes=["guns-image-box"],
                )

                prompt = gr.Textbox(
                    label="Transformation Prompt",
                    placeholder=(
                        "Describe how you want the image transformed..."
                    ),
                    lines=5,
                    elem_classes=["guns-prompt"],
                )

                generate_button = gr.Button(
                    "Generate",
                    variant="primary",
                    elem_classes=["guns-generate"],
                )

                gr.HTML(
                    """
                    <div class="guns-status">
                        Your original image is used as the starting point.
                    </div>
                    """
                )

            # ----------------------------------------------------
            # OUTPUT PANEL
            # ----------------------------------------------------

            with gr.Column(
                scale=1,
                elem_classes=["guns-panel"],
            ):

                gr.HTML(
                    """
                    <div class="guns-panel-title">
                        <svg class="guns-icon"
                             viewBox="0 0 24 24"
                             fill="none"
                             xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 3L13.9 8.1L19 10L13.9 11.9L12 17L10.1 11.9L5 10L10.1 8.1L12 3Z"
                                  stroke="#c084fc"
                                  stroke-width="1.6"
                                  stroke-linejoin="round"/>
                            <path d="M19 15L19.8 17.2L22 18L19.8 18.8L19 21L18.2 18.8L16 18L18.2 17.2L19 15Z"
                                  fill="#c084fc"/>
                        </svg>
                        <span>Generated Result</span>
                    </div>
                    """
                )

                result = gr.Image(
                    label="Result",
                    type="filepath",
                    elem_classes=["guns-image-box", "guns-result"],
                )

                gr.HTML(
                    """
                    <div class="guns-status">
                        Generated images will appear here.
                    </div>
                    """
                )

        # ========================================================
        # GENERATION CONNECTION
        # ========================================================

        generate_button.click(
            fn=generate_img2img,
            inputs=[
                source_image,
                prompt,
            ],
            outputs=result,
        )
