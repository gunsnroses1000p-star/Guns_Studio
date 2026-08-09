"""
ui/image_to_video_tab.py

Guns AI Studio — Image → Video interface.
"""

import gradio as gr

from providers.video_provider import generate_video


def build():

    # ============================================================
    # GUNS AI STUDIO — IMAGE TO VIDEO THEME
    #
    # IMPORTANT:
    # No global overflow, height, or viewport CSS.
    # Gradio controls layout and scrolling.
    # ============================================================

    gr.HTML(
        """
        <style>

        /* ========================================================
           MAIN CONTAINER
           ======================================================== */

        .guns-img2video {
            max-width: 1250px !important;
            margin: 0 auto !important;
        }

        /* ========================================================
           CUSTOM SVG ICONS
           ======================================================== */

        .guns-v-icon {
            width: 20px;
            height: 20px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;

            filter:
                drop-shadow(
                    0 0 5px rgba(168, 85, 247, 0.55)
                );
        }

        .guns-v-icon-lg {
            width: 29px;
            height: 29px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 9px;

            filter:
                drop-shadow(
                    0 0 8px rgba(168, 85, 247, 0.65)
                );
        }

        /* ========================================================
           HERO
           ======================================================== */

        .guns-img2video-hero {
            padding: 5px 4px 18px 4px;
        }

        .guns-img2video-title {
            margin: 0 !important;

            color: #f4f0ff !important;

            font-size: 29px !important;
            font-weight: 700 !important;

            line-height: 1.15 !important;
            letter-spacing: -0.4px !important;
        }

        .guns-img2video-subtitle {
            margin-top: 7px !important;

            color: #9e97aa !important;

            font-size: 14px !important;
            line-height: 1.55 !important;
        }

        /* ========================================================
           PANELS
           ======================================================== */

        .guns-video-panel {
            border:
                1px solid rgba(
                    168,
                    85,
                    247,
                    0.24
                ) !important;

            border-radius: 18px !important;

            padding: 17px !important;

            background:
                linear-gradient(
                    145deg,
                    rgba(28, 23, 36, 0.98),
                    rgba(10, 9, 13, 0.99)
                ) !important;

            box-shadow:
                0 8px 30px rgba(0, 0, 0, 0.32),
                inset 0 1px 0 rgba(
                    255,
                    255,
                    255,
                    0.025
                ) !important;
        }

        .guns-video-panel-title {
            display: flex !important;
            align-items: center !important;

            margin: 0 0 12px 0 !important;

            color: #eee9f5 !important;

            font-size: 13px !important;
            font-weight: 700 !important;

            letter-spacing: 1.4px !important;
            text-transform: uppercase !important;
        }

        /* ========================================================
           IMAGE INPUT
           ======================================================== */

        .guns-video-image {
            border-radius: 14px !important;

            border:
                1px solid rgba(
                    168,
                    85,
                    247,
                    0.20
                ) !important;

            background: #08070b !important;

            overflow: hidden !important;
        }

        .guns-video-image:hover {
            border-color:
                rgba(
                    168,
                    85,
                    247,
                    0.42
                ) !important;

            box-shadow:
                0 0 20px rgba(
                    168,
                    85,
                    247,
                    0.08
                ) !important;
        }

        /* ========================================================
           MOTION PROMPT
           ======================================================== */

        .guns-motion-prompt {
            margin-top: 12px !important;
        }

        .guns-motion-prompt label {
            color: #b9b2c3 !important;

            font-size: 13px !important;
            font-weight: 500 !important;
        }

        .guns-motion-prompt textarea {
            border-radius: 12px !important;

            border:
                1px solid rgba(
                    168,
                    85,
                    247,
                    0.20
                ) !important;

            background: #0b0a0e !important;

            color: #f4f0f8 !important;
        }

        .guns-motion-prompt textarea:hover {
            border-color:
                rgba(
                    168,
                    85,
                    247,
                    0.34
                ) !important;
        }

        .guns-motion-prompt textarea:focus {
            border-color:
                rgba(
                    168,
                    85,
                    247,
                    0.62
                ) !important;

            box-shadow:
                0 0 0 1px rgba(
                    168,
                    85,
                    247,
                    0.16
                ) !important;
        }

        .guns-motion-prompt textarea::placeholder {
            color: #706a77 !important;
        }

        /* ========================================================
           GENERATE BUTTON
           ======================================================== */

        #guns-video-generate {
            margin-top: 13px !important;
        }

        #guns-video-generate,
        #guns-video-generate button {
            background:
                linear-gradient(
                    135deg,
                    #6d28d9,
                    #8b5cf6,
                    #a855f7
                ) !important;

            background-color:
                #8b5cf6 !important;

            color:
                #ffffff !important;

            border:
                1px solid rgba(
                    216,
                    180,
                    254,
                    0.55
                ) !important;

            border-radius:
                13px !important;
        }

        #guns-video-generate button {
            width: 100% !important;

            min-height: 51px !important;

            color: #ffffff !important;

            font-size: 15px !important;
            font-weight: 700 !important;

            letter-spacing: 0.5px !important;

            box-shadow:
                0 7px 24px rgba(
                    124,
                    58,
                    237,
                    0.30
                ) !important;

            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                filter 0.15s ease !important;
        }

        #guns-video-generate button:hover {
            background:
                linear-gradient(
                    135deg,
                    #7c3aed,
                    #9333ea,
                    #c084fc
                ) !important;

            color: #ffffff !important;

            transform:
                translateY(-1px) !important;

            filter:
                brightness(1.07) !important;

            box-shadow:
                0 10px 30px rgba(
                    168,
                    85,
                    247,
                    0.38
                ) !important;
        }

        #guns-video-generate button:active {
            transform:
                translateY(0) !important;
        }

        /* ========================================================
           VIDEO OUTPUT
           ======================================================== */

        .guns-video-result {
            border-radius: 14px !important;

            border:
                1px solid rgba(
                    168,
                    85,
                    247,
                    0.20
                ) !important;

            background: #08070b !important;

            overflow: hidden !important;
        }

        /* ========================================================
           STATUS
           ======================================================== */

        .guns-video-status {
            display: flex;

            align-items: center;
            justify-content: center;

            gap: 7px;

            margin-top: 12px;

            padding: 10px 12px;

            border-radius: 10px;

            border:
                1px solid rgba(
                    168,
                    85,
                    247,
                    0.14
                );

            background:
                rgba(
                    168,
                    85,
                    247,
                    0.035
                );

            color: #9a92a4;

            font-size: 12px;

            line-height: 1.45;

            text-align: center;
        }

        .guns-video-status-dot {
            width: 7px;
            height: 7px;

            flex:
                0 0 7px;

            border-radius: 50%;

            background:
                #a855f7;

            box-shadow:
                0 0 8px rgba(
                    168,
                    85,
                    247,
                    0.75
                );
        }

        /* ========================================================
           MOBILE
           ======================================================== */

        @media (max-width: 768px) {

            .guns-img2video-hero {
                padding:
                    4px 4px 16px 4px;
            }

            .guns-img2video-title {
                font-size: 26px !important;
            }

            .guns-img2video-subtitle {
                font-size: 13px !important;
                line-height: 1.55 !important;
            }

            .guns-video-panel {
                padding: 13px !important;

                border-radius:
                    15px !important;
            }

            .guns-video-panel-title {
                font-size: 12px !important;

                letter-spacing:
                    1.25px !important;
            }

            .guns-motion-prompt textarea {
                min-height:
                    140px !important;
            }

            #guns-video-generate button {
                min-height:
                    50px !important;
            }
        }

        </style>
        """
    )

    # ============================================================
    # MAIN CONTAINER
    # ============================================================

    with gr.Column(
        elem_classes=["guns-img2video"]
    ):

        # ========================================================
        # HEADER
        # ========================================================

        gr.HTML(
            """
            <div class="guns-img2video-hero">

                <div class="guns-img2video-title">

                    <svg
                        class="guns-v-icon-lg"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg">

                        <rect
                            x="3"
                            y="4"
                            width="18"
                            height="16"
                            rx="3"
                            stroke="#c084fc"
                            stroke-width="1.8"/>

                        <path
                            d="M10 8L16 12L10 16V8Z"
                            fill="#c084fc"/>

                    </svg>

                    Image → Video

                </div>

                <div class="guns-img2video-subtitle">
                    Bring a still image to life with
                    natural, cinematic motion.
                </div>

            </div>
            """
        )

        # ========================================================
        # INPUT / OUTPUT
        #
        # Gradio controls responsive behavior.
        # No forced overflow or viewport rules.
        # ========================================================

        with gr.Row():

            # ====================================================
            # INPUT
            # ====================================================

            with gr.Column(
                elem_classes=["guns-video-panel"]
            ):

                gr.HTML(
                    """
                    <div class="guns-video-panel-title">

                        <svg
                            class="guns-v-icon"
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
                    elem_classes=[
                        "guns-video-image"
                    ],
                )

                motion_prompt = gr.Textbox(
                    label="Motion Prompt",
                    placeholder=(
                        "Describe the motion you want..."
                    ),
                    lines=4,
                    elem_classes=[
                        "guns-motion-prompt"
                    ],
                )

                generate_button = gr.Button(
                    "Generate Video",
                    variant="primary",
                    elem_id="guns-video-generate",
                )

                gr.HTML(
                    """
                    <div class="guns-video-status">

                        <span
                            class="guns-video-status-dot">
                        </span>

                        <span>
                            READY — upload an image to begin
                        </span>

                    </div>
                    """
                )

            # ====================================================
            # OUTPUT
            # ====================================================

            with gr.Column(
                elem_classes=["guns-video-panel"]
            ):

                gr.HTML(
                    """
                    <div class="guns-video-panel-title">

                        <svg
                            class="guns-v-icon"
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg">

                            <path
                                d="M8 5L19 12L8 19V5Z"
                                stroke="#c084fc"
                                stroke-width="1.8"
                                stroke-linejoin="round"/>

                            <path
                                d="M4 8V16"
                                stroke="#c084fc"
                                stroke-width="1.8"
                                stroke-linecap="round"/>

                        </svg>

                        <span>Generated Video</span>

                    </div>
                    """
                )

                result = gr.Video(
                    label="Result",
                    format="mp4",
                    elem_classes=[
                        "guns-video-result"
                    ],
                )

                gr.HTML(
                    """
                    <div class="guns-video-status">

                        <span>
                            Your generated video
                            will appear here.
                        </span>

                    </div>
                    """
                )

        # ========================================================
        # GENERATION CONNECTION
        #
        # BACKEND UNCHANGED.
        # ========================================================

        generate_button.click(
            fn=generate_video,
            inputs=[
                source_image,
                motion_prompt,
            ],
            outputs=result,
        )
