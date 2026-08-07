# 🔫 Guns AI Studio

A multi-modal AI creative suite built with Gradio and Hugging Face models.  
Generate text, images, audio, music, video, swap faces, and upscale images – all from a single web UI.

---

## Features

| Tab | Capability |
|-----|-----------|
| ✍️ Text Generation | Causal-LM inference (Mistral, Llama, etc.) |
| 🖼️ Image Generation | Stable Diffusion XL text-to-image |
| 🔊 Audio | Text-to-speech (MMS) + music generation (MusicGen) |
| 🎬 Video | Text-to-video (ModelScope) |
| 🎭 Face Swap | InsightFace inswapper |
| 🔍 Upscale | SD x4 upscaler |

## Project Structure

```
Guns_Studio/
├── app.py              # Gradio entrypoint – run this
├── config.py           # All model names, paths, and defaults
├── requirements.txt    # Python dependencies
├── providers/          # One module per capability
│   ├── text_provider.py
│   ├── image_provider.py
│   ├── audio_provider.py
│   ├── video_provider.py
│   ├── face_swap_provider.py
│   └── upscale_provider.py
├── ui/                 # Gradio tab builders
│   ├── text_tab.py
│   ├── image_tab.py
│   ├── audio_tab.py
│   ├── video_tab.py
│   ├── face_swap_tab.py
│   └── upscale_tab.py
├── utils/
│   └── helpers.py      # Shared utility functions
└── outputs/            # Generated files land here (git-ignored)
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/gunsnroses1000p-star/Guns_Studio.git
cd Guns_Studio

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set your Hugging Face token for gated models
export HF_TOKEN=hf_...

# 5. Launch
python app.py
```

The Gradio UI will be available at **http://localhost:7860**.

## Face Swap Setup

The face-swap tab requires `inswapper_128.onnx` from the InsightFace model zoo.  
Place the file at `models/inswapper_128.onnx` relative to the project root before using that tab.

## Configuration

Edit `config.py` to change default model names, generation parameters, or the output directory.

## License

See [LICENSE](LICENSE).