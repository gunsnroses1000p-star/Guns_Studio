# 🔫 Guns AI Studio

A premium AI creative studio built with Gradio, supporting multiple image and video generation providers.

## Features

| Tab | Description |
|-----|-------------|
| 🖼️ Text to Image | Generate images via Replicate, HuggingFace, RunPod, Fal.ai, or Civitai |
| 🔄 Img2Img | Transform existing images with Stable Diffusion (local GPU or RunPod) |
| 👤 IP-Adapter FaceID | Face-conditioned image generation with IP-Adapter FaceID Plus XL |
| 🎨 LoRA Generator | Generate images with custom LoRA weights + image-to-video |
| 🎞️ Local Video | LTX image-to-video and CogVideoX text-to-video (local GPU) |
| 🚀 RunPod Video | Image-to-video and video extension via RunPod |
| ⚡ Fal.ai | Image and video generation via Fal.ai (Kling, Flux, etc.) |
| 😎 Face Swap | Swap faces in images and videos using InsightFace |
| 🔒 Private Server | Image-to-video via a self-hosted private server |

## Project Structure

```
Guns_Studio/
├── app.py                          # Entrypoint
├── config.py                       # Env vars, constants, custom CSS
├── requirements.txt
├── outputs/                        # Generated files (git-ignored)
├── providers/
│   ├── replicate_provider.py       # Replicate image gen
│   ├── hf_provider.py              # Hugging Face image gen
│   ├── runpod_provider.py          # RunPod txt2img + img2img
│   ├── runpod_video_provider.py    # RunPod image-to-video + extension
│   ├── fal_provider.py             # Fal.ai image + video
│   ├── civitai_provider.py         # Civitai image gen
│   ├── lora_provider.py            # LoRA image gen + img2video
│   ├── ip_adapter_provider.py      # IP-Adapter FaceID
│   ├── video_provider.py           # Local LTX + CogVideoX
│   ├── faceswap_provider.py        # Face swap (image + video)
│   └── private_server_provider.py  # Private server video
├── utils/
│   ├── helpers.py                  # General helpers
│   ├── face.py                     # Face analyzer + preservation
│   └── runpod.py                   # RunPod job helpers
└── ui/
    └── tabs.py                     # Gradio UI assembly
```

## Setup

### Environment Variables (Hugging Face Secrets or `.env`)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face access token |
| `REPLICATE_API_TOKEN` | Replicate API token |
| `RUNPOD_API_KEY` | RunPod API key |
| `RUNPOD_ENDPOINT_ID` | RunPod endpoint ID |
| `FAL_KEY` | Fal.ai API key |
| `CIVITAI_API_KEY` | Civitai API key |
| `LORA_URL` | Base URL/path for LoRA weights |
| `PRIVATE_SERVER_URL` | URL of the private video generation server |

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

The app will be available at `http://localhost:7860`.

## Deploy on Hugging Face Spaces

1. Set SDK to `gradio` in your Space settings.
2. Add all required API keys as Space Secrets.
3. Push this repository — the Space will start automatically.

## License

See [LICENSE](LICENSE).