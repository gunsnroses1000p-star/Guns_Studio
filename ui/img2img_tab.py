"""
ui/img2img_tab.py

Guns AI Studio — Img2Img interface.

The generation backend is intentionally untouched.
This file controls the user-facing presentation only.
"""

import gradio as gr

from providers.img2img_provider import generate_img2img


def build():

    gr.HTML(
        """
        <style>

        /* ========================================================
           GUNS AI STUDIO — IMG2IMG
           ======================================================== */

        .guns-img2img {
            max-width: 1250px !important;
            margin: 0 auto !important;
        }

        /* ========================================================
           CUSTOM SVG ICONS
           ======================================================== */

        .guns-icon {
            width: 20px;
            height: 20px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;
            filter:
                drop-shadow(0 0 5px rgba(168, 85, 247, 0.55));
        }

        .guns-icon-lg {
            width: 29px;
            height: 29px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 9px;
            filter:
                drop-shadow(0 0 8px rgba(168, 85, 247, 0.65));
        }

        /* ========================================================
           HERO
           ======================================================== */

        .guns-img2img-hero {
            padding: 5px 4px 20px 4px;
        }

        .guns-img2img-title {
            margin: 0 !important;
            font-size: 29px !important;
            line-height: 1.15 !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px !important;
            color: #f4f0ff !important;
        }

        .guns-img2img-subtitle {
            margin-top: 7px !important;
            color: #9e97aa !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }

        /* ========================================================
           MAIN PANELS
           ======================================================== */

        .guns-panel {
            border: 1px solid rgba(168, 85, 247, 0.24) !important;
            border-radius: 18px !important;
            padding: 17px !important;

            background:
                linear-gradient(
                    145deg,
                    rgba(28, 23, 36, 0.98),
                    rgba(10, 9, 13, 0.99)
                ) !important;

            box-shadow:
                0 8px 30px rgba(0, 0, 0, 0.38),
                inset 0 1px 0 rgba(255, 255, 255, 0.025) !important;
        }

        .guns-panel-title {
            display: flex !important;
            align-items: center !important;

            font-size: 13px !important;
            line-height: 1.2 !important;
            font-weight: 700 !important;

            letter-spacing: 1.5px !important;
            text-transform: uppercase !important;

            color: #eee9f5 !important;

            margin: 0 0 13px 0 !important;
        }

        .guns-panel-title span {
            color: #eee9f5 !important;
        }

        /* ========================================================
           IMAGE UPLOAD / OUTPUT
           ======================================================== */

        .guns-image-box {
            border-radius: 14px !important;
            overflow: hidden !important;

            border: 1px solid rgba(168, 85, 247, 0.20) !important;

            background:
                #08070b !important;

            box-shadow:
                inset 0 0 20px rgba(0, 0, 0, 0.25) !important;
        }

        .guns-image-box:hover {
            border-color:
                rgba(168, 85, 247, 0.45) !important;

            box-shadow:
                0 0 20px rgba(168, 85, 247, 0.08),
                inset 0 0 20px rgba(0, 0, 0, 0.25) !important;
        }

        /* Keep image area from becoming ridiculously tall */

        .guns-image-box .image-container {
            min-height: 280px !important;
            height: 360px !important;
        }

        .guns-image-box img {
            object-fit: contain !important;
        }

        /* ========================================================
           PROMPT
           ======================================================== */

        .guns-prompt {
            margin-top: 13px !important;
        }

        .guns-prompt label {
            color: #b9b2c3 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }

        .guns-prompt textarea {
            border-radius: 12px !important;

            border: 1px solid
                rgba(168, 85, 247, 0.18) !important;

            background:
                rgba(8, 7, 10, 0.96) !important;

            color: #f4f0f8 !important;

            box-shadow:
                inset 0 0 15px rgba(0, 0, 0, 0.20) !important;
        }

        .guns-prompt textarea:hover {
            border-color:
                rgba(168, 85, 247, 0.32) !important;
        }

        .guns-prompt textarea:focus {
            border-color:
                rgba(168, 85, 247, 0.62) !important;

            box-shadow:
                0 0 0 1px rgba(168, 85, 247, 0.16),
                0 0 18px rgba(168, 85, 247, 0.08),
                inset 0 0 15px rgba(0, 0, 0, 0.20) !important;
        }

        .guns-prompt textarea::placeholder {
            color: #706a77 !important;
        }

        /* ========================================================
           GENERATE BUTTON
           ======================================================== */

        .guns-generate {
            margin-top: 13px !important;
        }

        /*
           Strong selectors intentionally override
           Gradio's default orange/red styling.
        */

        .guns-generate button,
        .guns-generate button.primary,
        .guns-generate button[data-testid="primary"] {
            width: 100% !important;
            min-height: 51px !important;

            border-radius: 13px !important;

            border: 1px solid
                rgba(216, 180, 254, 0.55) !important;

            background:
                linear-gradient(
                    135deg,
                    #6d28d9 0%,
                    #8b5cf6 50%,
                    #a855f7 100%
                ) !important;

            color: #ffffff !important;

            font-size: 15px !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px !important;

            box-shadow:
                0 7px 24px
                rgba(124, 58, 237, 0.30) !important;

            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                filter 0.15s ease !important;
        }

        .guns-generate button:hover,
        .guns-generate button.primary:hover {
            background:
                linear-gradient(
                    135deg,
                    #7c3aed 0%,
                    #9333ea 50%,
                    #c084fc 100%
                ) !important;

            color: #ffffff !important;

            transform: translateY(-1px) !important;

            filter: brightness(1.07) !important;

            box-shadow:
                0 10px 30px
                rgba(168, 85, 247, 0.40) !important;
        }

        .guns-generate button:active {
            transform: translateY(0) !important;
        }

        /* ========================================================
           STATUS
           ======================================================== */

        .guns-status {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 7px;

            margin-top: 13px;

            padding: 10px 12px;

            border-radius: 10px;

            border: 1px solid
                rgba(168, 85, 247, 0.14);

            background:
                rgba(168, 85, 247, 0.035);

            color: #8e8797;

            font-size: 12px;
            line-height: 1.45;

            text-align: center;
        }

        .guns-status-ready {
            color: #b8a9c8;
        }

        .guns-status-dot {
            width: 7px;
            height: 7px;

            border-radius: 50%;

            background: #a855f7;

            box-shadow:
                0 0 8px rgba(168, 85, 247, 0.75);
        }

        /* ========================================================
           MOBILE
           ======================================================== */

        @media (max-width: 768px) {

            .guns-img2img {
                width: 100% !important;
                max-width: 100% !important;
            }

            .guns-img2img-hero {
                padding:
                    4px 4px 17px 4px;
            }

            .guns-img2img-title {
                font-size: 26px !important;
            }

            .guns-img2img-subtitle {
                font-size: 13px !important;
                line-height: 1.55 !important;
                max-width: 340px !important;
            }

            .guns-panel {
                padding: 13px !important;
                border-radius: 15px !important;
            }

            .guns-panel-title {
                font-size: 12px !important;
                letter-spacing: 1.3px !important;
                margin-bottom: 11px !important;
            }

            .guns-image-box .image-container {
                height: 285px !important;
                min-height: 240px !important;
            }

            .guns-prompt {
                margin-top: 11px !important;
            }

            .guns-prompt textarea {
                min-height: 150px !important;
            }

            .guns-generate button {
                min-height: 50px !important;
            }

            .guns-status {
                font-size: 11.5px;
                padding: 9px 10px;
            }
        }

        /* ========================================================
           VERY SMALL PHONES
           ======================================================== */

        @media (max-width: 430px) {

            .guns-img2img-title {
                font-size: 25px !important;
            }

            .guns-image-box .image-container {
                height: 260px !important;
                min-height: 220px !important;
            }

            .guns-prompt textarea {
                min-height: 140px !important;
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

                        <rect x="3" y="4"
                              width="18"
                              height="16"
                              rx="3"
                              stroke="#c084fc"
                              stroke-width="1.8"/>

                        <circle cx="8.5"
                                cy="9"
                                r="1.5"
                                fill="#c084fc"/>

                        <path
                            d="M5.5 17L10 12.5L13 15L15.5 12.5L19 17"
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

            # ====================================================
            # INPUT
            # ====================================================

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

                            <path
                                d="M12 3V15"
                                stroke="#c084fc"
                                stroke-width="2"
                                stroke-linecap="round"/>

                            <path
                                d="M7 8L12 3L17 8"
                                stroke="#c084fc"
                                stroke-width="2"
                                stroke-linecap="round"
                                stroke-linejoin="round"/>

                            <path
                                d="M5 15V19C5 20.1 5.9 21 7 21H17C18.1 21 19 20.1 19 19V15"
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
                    <div class="guns-status guns-status-ready">

                        <span class="guns-status-dot"></span>

                        <span>READY — upload an image to begin</span>

                    </div>
                    """
                )

            # ====================================================
            # RESULT
            # ====================================================

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

                            <path
                                d="M12 3L13.9 8.1L19 10L13.9 11.9L12 17L10.1 11.9L5 10L10.1 8.1L12 3Z"
                                stroke="#c084fc"
                                stroke-width="1.6"
                                stroke-linejoin="round"/>

                            <path
                                d="M19 15L19.8 17.2L22 18L19.8 18.8L19 21L18.2 18.8L16 18L18.2 17.2L19 15Z"
                                fill="#c084fc"/>

                        </svg>

                        <span>Generated Result</span>

                    </div>
                    """
                )

                result = gr.Image(
                    label="Result",
                    type="filepath",
                    elem_classes=[
                        "guns-image-box",
                        "guns-result",
                    ],
                )

                gr.HTML(
                    """
                    <div class="guns-status">

                        <span>
                            Generated images will appear here.
                        </span>

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
