"""
app.py — Entrypoint for Guns AI Studio.

Run locally:
    python app.py

Deploy on Hugging Face Spaces with `gradio` as the SDK.
"""

import os
from pathlib import Path

# Ensure the outputs directory exists before any provider tries to write there
Path("outputs").mkdir(exist_ok=True)

from ui.tabs import build_ui  # noqa: E402 – import after outputs/ is created


def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )


if __name__ == "__main__":
    main()
