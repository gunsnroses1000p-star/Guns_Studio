""" providers/video_provider.py – Text-to-video generation via Diffusers. """
from __future__ import annotations
import os
import uuid
import spaces
from config import (
    HF_TOKEN, 
    VIDEO_MODEL_DEFAULT, 
    VIDEO_FRAMES_DEFAULT, 
    VIDEO_FPS_DEFAULT, 
    OUTPUT_DIR,
)

# Global variables to store the model and current state
_pipe = None
_loaded_model: str = ""

def _load(model_name: str) -> None:
    global _pipe, _loaded_model
    
    # 1. If the correct model is already loaded, do nothing
    if _loaded_model == model_name and _pipe is not None:
        return

    # 2. Lazy import torch and diffusers ONLY when needed
    import torch
    from diffusers import DiffusionPipeline
    
    print(f"Loading video model {model_name} into VRAM... please wait.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 3. Load the pipeline
    _pipe = DiffusionPipeline.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        token=HF_TOKEN or None,
    )
    _pipe = _pipe.to(device)
    
    # 4. Optional: Enable memory optimizations for RTX 6000
    if device == "cuda":
        _pipe.enable_model_cpu_offload() # Saves huge amounts of VRAM
        _pipe.enable_vae_slicing() # Prevents crashes during video encoding
        
    _loaded_model = model_name
@spaces.GPU
def generate(
    prompt: str, 
    model_name: str = VIDEO_MODEL_DEFAULT, 
    num_frames: int = VIDEO_FRAMES_DEFAULT, 
    fps: int = VIDEO_FPS_DEFAULT,
) -> str:
    """Generate a short video and return the saved .mp4 file path."""
    global _pipe
    
    # Ensure model is loaded
    _load(model_name)
    
    # Lazy import imageio
    import imageio
    
    # Run the pipeline
    result = _pipe(prompt, num_frames=num_frames)
    frames = result.frames[0] # list of PIL Images
    
    # Save the file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"video_{uuid.uuid4().hex[:8]}.mp4")
    imageio.mimsave(out_path, frames, fps=fps)
    
    return out_path