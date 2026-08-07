"""
ui/tabs.py — Gradio tab assembly for Guns AI Studio.

Each function returns a gr.Tab block. They are composed in build_ui().
"""

import gradio as gr

from config import (
    CUSTOM_CSS,
    DEFAULT_HEIGHT,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_NEGATIVE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
)
from providers.lora_provider import lora_names


# ---------------------------------------------------------------------------
# Hero / header HTML
# ---------------------------------------------------------------------------

HERO_HTML = """
<div id="hero-section">
  <h1>🔫 GUNS AI STUDIO</h1>
  <h3>Creative AI · Image · Video · Face</h3>
</div>
"""


# ---------------------------------------------------------------------------
# Helper: common prompt row
# ---------------------------------------------------------------------------

def _prompt_row(placeholder="Enter your prompt here..."):
    prompt = gr.Textbox(
        label="Prompt",
        placeholder=placeholder,
        lines=3,
    )
    neg = gr.Textbox(
        label="Negative Prompt",
        value=DEFAULT_NEGATIVE,
        lines=2,
    )
    return prompt, neg


def _size_seed_row():
    with gr.Row():
        width = gr.Slider(512, 2048, value=DEFAULT_WIDTH, step=64, label="Width")
        height = gr.Slider(512, 2048, value=DEFAULT_HEIGHT, step=64, label="Height")
    with gr.Row():
        steps = gr.Slider(1, 100, value=DEFAULT_STEPS, step=1, label="Steps")
        seed = gr.Number(value=DEFAULT_SEED, label="Seed (0 = random)", precision=0)
    return width, height, steps, seed


# ---------------------------------------------------------------------------
# Tab 1 — Text to Image
# ---------------------------------------------------------------------------

def build_txt2img_tab():
    with gr.Tab("🖼️ Text to Image"):
        gr.HTML("<div class='guns-tool-header'><h2>Text to Image</h2></div>")
        with gr.Row():
            with gr.Column():
                provider = gr.Dropdown(
                    choices=["Replicate", "HuggingFace", "RunPod", "Fal.ai", "Civitai"],
                    value="Replicate",
                    label="Provider",
                )
                model = gr.Textbox(
                    value=DEFAULT_IMAGE_MODEL,
                    label="Model (repo or Replicate slug)",
                )
                prompt, neg = _prompt_row()
                width, height, steps, seed = _size_seed_row()
                btn = gr.Button("✨ Generate", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Result", type="pil")
                status = gr.Textbox(label="Status", interactive=False)

        def _generate(provider, model, prompt, neg, w, h, steps, seed):
            from providers.replicate_provider import generate_with_replicate
            from providers.hf_provider import generate_with_hf
            from providers.runpod_provider import generate_with_runpod
            from providers.fal_provider import generate_image_with_fal
            from providers.civitai_provider import generate_with_civitai

            if provider == "Replicate":
                img = generate_with_replicate(prompt, neg, w, h, steps, seed, model=model)
                return img, "✅ Done (Replicate)"
            elif provider == "HuggingFace":
                img = generate_with_hf(prompt, neg, w, h, steps, seed, model=model)
                return img, "✅ Done (HuggingFace)"
            elif provider == "RunPod":
                img = generate_with_runpod(prompt, neg, w, h, steps, seed)
                return img, "✅ Done (RunPod)"
            elif provider == "Fal.ai":
                img = generate_image_with_fal(prompt, neg, w, h, steps, seed)
                return img, "✅ Done (Fal.ai)"
            elif provider == "Civitai":
                img = generate_with_civitai(prompt, neg, w, h, steps, seed)
                return img, "✅ Done (Civitai)"
            raise gr.Error(f"Unknown provider: {provider}")

        btn.click(
            _generate,
            inputs=[provider, model, prompt, neg, width, height, steps, seed],
            outputs=[output_image, status],
        )


# ---------------------------------------------------------------------------
# Tab 2 — Img2Img
# ---------------------------------------------------------------------------

def build_img2img_tab():
    with gr.Tab("🔄 Img2Img"):
        gr.HTML("<div class='guns-tool-header'><h2>Image to Image</h2></div>")
        with gr.Row():
            with gr.Column():
                provider = gr.Dropdown(
                    choices=["Local GPU", "RunPod"],
                    value="Local GPU",
                    label="Provider",
                )
                model = gr.Textbox(
                    value="runwayml/stable-diffusion-v1-5",
                    label="Model",
                )
                init_image = gr.Image(label="Input Image", type="pil")
                init_image_2 = gr.Image(label="Reference Image 2 (optional)", type="pil")
                prompt, neg = _prompt_row()
                with gr.Row():
                    strength = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Strength")
                    guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                width, height, steps, seed = _size_seed_row()
                with gr.Row():
                    preserve_face = gr.Checkbox(label="Preserve Face", value=False)
                    face_blend = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Face Blend")
                btn = gr.Button("🔄 Generate", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Result", type="pil")
                status = gr.Textbox(label="Status", interactive=False)

        def _generate(provider, prompt, neg_prompt, init_img, init_img2, model,
                      strength, guidance, steps, seed, preserve, blend):
            from providers.runpod_provider import generate_img2img_with_provider
            return generate_img2img_with_provider(
                provider if provider == "RunPod" else "Local",
                prompt, init_img, init_img2, model,
                strength, guidance, steps, seed, preserve, blend,
            )

        btn.click(
            _generate,
            inputs=[provider, prompt, neg, init_image, init_image_2, model,
                    strength, guidance, steps, seed, preserve_face, face_blend],
            outputs=[output_image, status],
        )


# ---------------------------------------------------------------------------
# Tab 3 — IP-Adapter FaceID
# ---------------------------------------------------------------------------

def build_ip_adapter_tab():
    with gr.Tab("👤 IP-Adapter FaceID"):
        gr.HTML("<div class='guns-tool-header'><h2>IP-Adapter FaceID</h2></div>")
        with gr.Row():
            with gr.Column():
                ref_image = gr.Image(label="Reference Face Image", type="pil")
                prompt, neg = _prompt_row("Describe the scene or style...")
                width, height, steps, seed = _size_seed_row()
                scale = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="IP-Adapter Scale")
                btn = gr.Button("🧠 Generate", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Result", type="pil")
                status = gr.Textbox(label="Status", interactive=False)

        def _generate(ref_img, prompt, neg, w, h, steps, seed, scale):
            from providers.ip_adapter_provider import generate_with_ip_adapter
            img = generate_with_ip_adapter(ref_img, prompt, neg, w, h, steps, seed, scale)
            return img, "✅ IP-Adapter FaceID complete."

        btn.click(
            _generate,
            inputs=[ref_image, prompt, neg, width, height, steps, seed, scale],
            outputs=[output_image, status],
        )


# ---------------------------------------------------------------------------
# Tab 4 — LoRA Generation + img2video
# ---------------------------------------------------------------------------

def build_lora_tab():
    with gr.Tab("🎨 LoRA Generator"):
        gr.HTML("<div class='guns-tool-header'><h2>LoRA Image + Video</h2></div>")
        with gr.Row():
            with gr.Column():
                lora_selector = gr.Dropdown(
                    choices=lora_names,
                    label="LoRA Style",
                    allow_custom_value=True,
                )
                prompt, neg = _prompt_row()
                width, height, steps, seed = _size_seed_row()
                lora_scale = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="LoRA Scale")
                btn_img = gr.Button("🎨 Generate Image", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
                # hidden URL store
                image_url_state = gr.State(value="")
                status_img = gr.Textbox(label="Image Status", interactive=False)

        with gr.Row():
            with gr.Column():
                video_prompt = gr.Textbox(label="Video Prompt (optional)", lines=2)
                video_seed = gr.Number(value=0, label="Video Seed", precision=0)
                video_duration = gr.Slider(2, 10, value=5, step=1, label="Duration (s)")
                btn_vid = gr.Button("🎬 Image → Video", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="Generated Video")
                status_vid = gr.Textbox(label="Video Status", interactive=False)

        def _gen_img(lora, prompt, neg, w, h, steps, seed, scale):
            from providers.lora_provider import generate_with_lora
            url = generate_with_lora(prompt, lora, neg, w, h, steps, seed, scale)
            return url, url, "✅ LoRA image generated."

        def _gen_vid(img_url, v_prompt, duration, v_seed):
            from providers.lora_provider import generate_lora_image_to_video
            url = generate_lora_image_to_video(img_url, v_prompt, duration, v_seed)
            return url, "✅ Video generated."

        btn_img.click(
            _gen_img,
            inputs=[lora_selector, prompt, neg, width, height, steps, seed, lora_scale],
            outputs=[output_image, image_url_state, status_img],
        )
        btn_vid.click(
            _gen_vid,
            inputs=[image_url_state, video_prompt, video_duration, video_seed],
            outputs=[output_video, status_vid],
        )


# ---------------------------------------------------------------------------
# Tab 5 — Local Video (LTX + CogVideoX)
# ---------------------------------------------------------------------------

def build_local_video_tab():
    with gr.Tab("🎞️ Local Video"):
        gr.HTML("<div class='guns-tool-header'><h2>Local HF Video Generation</h2></div>")

        with gr.Tabs():
            # LTX
            with gr.Tab("LTX Image-to-Video"):
                with gr.Row():
                    with gr.Column():
                        ltx_image = gr.Image(label="Input Image", type="pil")
                        ltx_prompt, ltx_neg = _prompt_row()
                        with gr.Row():
                            ltx_w = gr.Slider(256, 1280, value=768, step=64, label="Width")
                            ltx_h = gr.Slider(256, 1280, value=512, step=64, label="Height")
                        with gr.Row():
                            ltx_frames = gr.Slider(8, 120, value=25, step=1, label="Frames")
                            ltx_fps = gr.Slider(4, 30, value=8, step=1, label="FPS")
                        ltx_steps = gr.Slider(10, 100, value=50, step=1, label="Steps")
                        ltx_seed = gr.Number(value=0, label="Seed", precision=0)
                        ltx_btn = gr.Button("▶️ Generate LTX Video", variant="primary")
                    with gr.Column():
                        ltx_output = gr.Video(label="LTX Output")
                        ltx_status = gr.Textbox(label="Status", interactive=False)

                def _ltx(img, prompt, neg, w, h, frames, fps, steps, seed):
                    from providers.video_provider import generate_ltx_video
                    path = generate_ltx_video(img, prompt, neg, w, h, frames, fps, steps, seed)
                    return path, "✅ LTX video generated."

                ltx_btn.click(
                    _ltx,
                    inputs=[ltx_image, ltx_prompt, ltx_neg, ltx_w, ltx_h,
                            ltx_frames, ltx_fps, ltx_steps, ltx_seed],
                    outputs=[ltx_output, ltx_status],
                )

            # CogVideoX
            with gr.Tab("CogVideoX Text-to-Video"):
                with gr.Row():
                    with gr.Column():
                        cog_prompt, cog_neg = _prompt_row()
                        with gr.Row():
                            cog_w = gr.Slider(256, 1280, value=720, step=64, label="Width")
                            cog_h = gr.Slider(256, 1280, value=480, step=64, label="Height")
                        with gr.Row():
                            cog_frames = gr.Slider(8, 120, value=49, step=1, label="Frames")
                            cog_fps = gr.Slider(4, 30, value=8, step=1, label="FPS")
                        cog_steps = gr.Slider(10, 100, value=50, step=1, label="Steps")
                        cog_seed = gr.Number(value=0, label="Seed", precision=0)
                        cog_btn = gr.Button("▶️ Generate CogVideo", variant="primary")
                    with gr.Column():
                        cog_output = gr.Video(label="CogVideoX Output")
                        cog_status = gr.Textbox(label="Status", interactive=False)

                def _cog(prompt, neg, w, h, frames, fps, steps, seed):
                    from providers.video_provider import generate_cogvideo
                    path = generate_cogvideo(prompt, neg, w, h, frames, fps, steps, seed)
                    return path, "✅ CogVideoX video generated."

                cog_btn.click(
                    _cog,
                    inputs=[cog_prompt, cog_neg, cog_w, cog_h,
                            cog_frames, cog_fps, cog_steps, cog_seed],
                    outputs=[cog_output, cog_status],
                )


# ---------------------------------------------------------------------------
# Tab 6 — RunPod Image-to-Video + Video Extension
# ---------------------------------------------------------------------------

def build_runpod_video_tab():
    with gr.Tab("🚀 RunPod Video"):
        gr.HTML("<div class='guns-tool-header'><h2>RunPod Image-to-Video</h2></div>")

        with gr.Tabs():
            with gr.Tab("Image to Video"):
                with gr.Row():
                    with gr.Column():
                        rp_image = gr.Image(label="Input Image", type="pil")
                        rp_prompt, rp_neg = _prompt_row()
                        with gr.Row():
                            rp_steps = gr.Slider(10, 100, value=25, step=1, label="Steps")
                            rp_duration = gr.Slider(2, 30, value=5, step=1, label="Duration (s)")
                        rp_seed = gr.Number(value=0, label="Seed", precision=0)
                        rp_btn = gr.Button("🚀 Generate Video", variant="primary")
                    with gr.Column():
                        rp_output = gr.Video(label="RunPod Video Output")
                        rp_status = gr.Textbox(label="Status", interactive=False)

                def _rp_vid(img, prompt, neg, steps, duration, seed):
                    from providers.runpod_video_provider import generate_runpod_image_to_video
                    url = generate_runpod_image_to_video(img, prompt, neg, steps, seed, duration)
                    return url, "✅ RunPod video generated."

                rp_btn.click(
                    _rp_vid,
                    inputs=[rp_image, rp_prompt, rp_neg, rp_steps, rp_duration, rp_seed],
                    outputs=[rp_output, rp_status],
                )

            with gr.Tab("Extend Video"):
                with gr.Row():
                    with gr.Column():
                        ext_video = gr.Video(label="Input Video")
                        ext_prompt, ext_neg = _prompt_row("Continuation prompt...")
                        with gr.Row():
                            ext_steps = gr.Slider(10, 100, value=25, step=1, label="Steps")
                            ext_secs = gr.Slider(1, 30, value=4, step=1, label="Extra Seconds")
                        ext_seed = gr.Number(value=0, label="Seed", precision=0)
                        ext_btn = gr.Button("➕ Extend Video", variant="primary")
                    with gr.Column():
                        ext_output = gr.Video(label="Extended Video")
                        ext_status = gr.Textbox(label="Status", interactive=False)

                def _extend(video, prompt, neg, steps, secs, seed):
                    from providers.runpod_video_provider import extend_runpod_video
                    url = extend_runpod_video(video, prompt, neg, steps, seed, secs)
                    return url, "✅ Video extended."

                ext_btn.click(
                    _extend,
                    inputs=[ext_video, ext_prompt, ext_neg, ext_steps, ext_secs, ext_seed],
                    outputs=[ext_output, ext_status],
                )


# ---------------------------------------------------------------------------
# Tab 7 — Fal.ai Image & Video
# ---------------------------------------------------------------------------

def build_fal_tab():
    with gr.Tab("⚡ Fal.ai"):
        gr.HTML("<div class='guns-tool-header'><h2>Fal.ai Image & Video</h2></div>")

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        fal_model = gr.Textbox(value="fal-ai/flux/dev", label="Fal Model")
                        fal_prompt, fal_neg = _prompt_row()
                        fal_w, fal_h, fal_steps, fal_seed = _size_seed_row()
                        fal_btn = gr.Button("⚡ Generate Image", variant="primary")
                    with gr.Column():
                        fal_output = gr.Image(label="Result", type="pil")
                        fal_status = gr.Textbox(label="Status", interactive=False)

                def _fal_img(model, prompt, neg, w, h, steps, seed):
                    from providers.fal_provider import generate_image_with_fal
                    img = generate_image_with_fal(prompt, neg, w, h, steps, seed, model=model)
                    return img, "✅ Fal.ai image generated."

                fal_btn.click(
                    _fal_img,
                    inputs=[fal_model, fal_prompt, fal_neg, fal_w, fal_h, fal_steps, fal_seed],
                    outputs=[fal_output, fal_status],
                )

            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        falv_model = gr.Textbox(
                            value="fal-ai/kling-video/v1/standard/image-to-video",
                            label="Fal Video Model",
                        )
                        falv_image = gr.Image(label="Input Image (optional)", type="pil")
                        falv_prompt = gr.Textbox(label="Prompt", lines=2)
                        with gr.Row():
                            falv_dur = gr.Dropdown(["5", "10"], value="5", label="Duration (s)")
                            falv_ar = gr.Dropdown(
                                ["16:9", "9:16", "1:1"], value="16:9", label="Aspect Ratio"
                            )
                        falv_btn = gr.Button("⚡ Generate Video", variant="primary")
                    with gr.Column():
                        falv_output = gr.Video(label="Fal Video Output")
                        falv_status = gr.Textbox(label="Status", interactive=False)

                def _fal_vid(model, image, prompt, dur, ar):
                    from providers.fal_provider import generate_video_with_fal
                    url = generate_video_with_fal(prompt, image, model=model, duration=dur, aspect_ratio=ar)
                    return url, "✅ Fal.ai video generated."

                falv_btn.click(
                    _fal_vid,
                    inputs=[falv_model, falv_image, falv_prompt, falv_dur, falv_ar],
                    outputs=[falv_output, falv_status],
                )


# ---------------------------------------------------------------------------
# Tab 8 — Face Swap
# ---------------------------------------------------------------------------

def build_face_swap_tab():
    with gr.Tab("😎 Face Swap"):
        gr.HTML("<div class='guns-tool-header'><h2>Face Swap</h2></div>")

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        fs_source = gr.Image(label="Source Face Image", type="pil")
                        fs_target = gr.Image(label="Target Image", type="pil")
                        fs_btn = gr.Button("🔀 Swap Face", variant="primary")
                    with gr.Column():
                        fs_output = gr.Image(label="Result", type="pil")
                        fs_status = gr.Textbox(label="Status", interactive=False)

                def _swap_img(src, tgt):
                    from providers.faceswap_provider import face_swap_image
                    result = face_swap_image(src, tgt)
                    return result, "✅ Face swap complete."

                fs_btn.click(
                    _swap_img,
                    inputs=[fs_source, fs_target],
                    outputs=[fs_output, fs_status],
                )

            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        fsv_source = gr.Image(label="Source Face Image", type="pil")
                        fsv_video = gr.Video(label="Target Video")
                        fsv_btn = gr.Button("🔀 Swap Face in Video", variant="primary")
                    with gr.Column():
                        fsv_output = gr.Video(label="Result Video")
                        fsv_status = gr.Textbox(label="Status", interactive=False)

                def _swap_vid(src, vid):
                    from providers.faceswap_provider import face_swap_video
                    result = face_swap_video(src, vid)
                    return result, "✅ Video face swap complete."

                fsv_btn.click(
                    _swap_vid,
                    inputs=[fsv_source, fsv_video],
                    outputs=[fsv_output, fsv_status],
                )


# ---------------------------------------------------------------------------
# Tab 9 — Private Server
# ---------------------------------------------------------------------------

def build_private_server_tab():
    with gr.Tab("🔒 Private Server"):
        gr.HTML("<div class='guns-tool-header'><h2>Private Server Image-to-Video</h2></div>")
        with gr.Row():
            with gr.Column():
                ps_image = gr.Image(label="Input Image", type="pil")
                ps_prompt, ps_neg = _prompt_row()
                with gr.Row():
                    ps_steps = gr.Slider(10, 100, value=25, step=1, label="Steps")
                    ps_duration = gr.Slider(2, 30, value=5, step=1, label="Duration (s)")
                ps_seed = gr.Number(value=0, label="Seed", precision=0)
                ps_btn = gr.Button("🔒 Generate", variant="primary")
            with gr.Column():
                ps_output = gr.Video(label="Output Video")
                ps_status = gr.Textbox(label="Status", interactive=False)

        def _ps_vid(img, prompt, neg, steps, duration, seed):
            from providers.private_server_provider import generate_private_server_video
            url = generate_private_server_video(img, prompt, neg, steps, seed, duration)
            return url, "✅ Private server video generated."

        ps_btn.click(
            _ps_vid,
            inputs=[ps_image, ps_prompt, ps_neg, ps_steps, ps_duration, ps_seed],
            outputs=[ps_output, ps_status],
        )


# ---------------------------------------------------------------------------
# Main UI builder
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Guns AI Studio") as demo:
        gr.HTML(HERO_HTML)

        with gr.Tabs():
            build_txt2img_tab()
            build_img2img_tab()
            build_ip_adapter_tab()
            build_lora_tab()
            build_local_video_tab()
            build_runpod_video_tab()
            build_fal_tab()
            build_face_swap_tab()
            build_private_server_tab()

    return demo
