"""
app.py – Guns AI Studio entrypoint.

Run:
    python app.py

Environment variables (optional):
    HF_TOKEN – Hugging Face access token for gated models
    PORT – server port (default: 7860)
"""

import os
import socket
import datetime


# ---------------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------------

def _print_startup_summary(port: int) -> None:
    """Print a clear startup summary to aid SSR / port-conflict diagnosis."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ssr_backend_port = port + 1  # Gradio SSR uses main_port+1 for the Python backend
    print(f"\n===== Guns AI Studio Startup at {now} =====", flush=True)
    print(f"  Intended server port  : {port}", flush=True)
    print(f"  SSR Python backend    : {ssr_backend_port} (used by Node proxy if SSR is active)", flush=True)

    ssr_env_vars = [
        "GRADIO_SSR_MODE",
        "GRADIO_NODE_SERVER_NAME",
        "GRADIO_SERVER_PORT",
        "PORT",
        "HF_SPACES_RUNTIME_TASK_ENVIRONMENT",
        "SPACE_ID",
    ]
    print("  Relevant env vars:", flush=True)
    for var in ssr_env_vars:
        val = os.environ.get(var)
        print(f"    {var} = {val!r}", flush=True)

    _check_port_available(ssr_backend_port)
    print("==============================================\n", flush=True)


def _check_port_available(port: int) -> None:
    """Non-blocking check: warn if a port is already in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", port))
        if result == 0:
            print(
                f"  ⚠  WARNING: Port {port} appears to be OCCUPIED before launch "
                "(something may conflict with the SSR Python backend).",
                flush=True,
            )
        else:
            print(f"  ✓  Port {port} is free.", flush=True)
    except OSError as exc:
        print(f"  Port {port} check skipped ({exc})", flush=True)


# IMPORTANT: Import spaces BEFORE torch/diffusers or anything CUDA-related.
try:
    import spaces
except Exception as exc:
    print(f"Failed while importing spaces: {exc!r}", flush=True)
    raise

try:
    import gradio as gr
except Exception as exc:
    print(f"Failed while importing gradio: {exc!r}", flush=True)
    raise


def build_app() -> gr.Blocks:
    print("Importing UI modules...", flush=True)
    try:
        from ui import (
            text_tab,
            image_tab,
            audio_tab,
            video_tab,
            face_swap_tab,
            upscale_tab,
        )
    except Exception as exc:
        print(f"Failed while importing UI modules: {exc!r}", flush=True)
        raise

    with gr.Blocks(title="Guns AI Studio") as demo:
        gr.Markdown(
            """
# 🔫 Guns AI Studio

A multi-modal AI creative suite — text, images, audio, video,
face swap, and upscaling.
"""
        )

        tab_builders = (
            ("text", text_tab.build),
            ("image", image_tab.build),
            ("audio", audio_tab.build),
            ("video", video_tab.build),
            ("face swap", face_swap_tab.build),
            ("upscale", upscale_tab.build),
        )
        for tab_name, builder in tab_builders:
            print(f"Building {tab_name} tab...", flush=True)
            try:
                builder()
            except Exception as exc:
                print(f"Failed while building {tab_name} tab: {exc!r}", flush=True)
                raise

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    # Print a clear diagnostic summary before anything else starts.
    _print_startup_summary(port)

    print(f"Starting Guns AI Studio on port {port}...", flush=True)
    print("(This is the only Gradio launch — no secondary server will be started.)", flush=True)

    try:
        print("Building Gradio app...", flush=True)
        app = build_app()
        print("Gradio app built successfully.", flush=True)
    except Exception as exc:
        print(f"Startup failed during app construction: {exc!r}", flush=True)
        raise

    try:
        print("Launching Gradio app...", flush=True)
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,       # surface Python-side errors in the UI
            prevent_thread_lock=False,  # block until the server exits (keeps HF Space alive)
        )
        print("Gradio app has exited launch().", flush=True)
    except Exception as exc:
        print(f"Startup failed during app launch: {exc!r}", flush=True)
        raise