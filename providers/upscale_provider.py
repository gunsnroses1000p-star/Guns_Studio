""" providers/upscale_provider.py – Image upscaling via a Stable Diffusion x4 upscaler. """
from __future__ import annotations
import os
import uuid
import spaces
from PIL import Image
from config import (
    HF_TOKEN, 
    UPSCALE_MODEL_DEFAULT, 
    UPSCALE_STEPS_DEFAULT, 
    OUTPUT_DIR,
)

# Global variables to store the model and state
_pipe = None
_loaded_model: str = ""

def _load(model_name: str) -> None:
    global _pipe, _loaded_model
    
    # 1. If the correct model is already loaded, do nothing
    if _loaded_model == model_name and _pipe is not None:
        return

    # 2. Lazy import torch and diffusers ONLY when needed
    import torch
    from diffusers import StableDiffusionUpscalePipeline
    
    print(f"Loading Upscale model {model_name} into VRAM... please wait.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # RTX 6000 Optimization: Use float16 instead of float32
    # This reduces VRAM usage by 50% with no visible quality loss
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 3. Load the pipeline
    _pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        token=HF_TOKEN or None,
    )
    _pipe = _pipe.to(device)
    
    # 4. CRITICAL RTX 6000 Optimizations for Upscaling
    if device == "cuda":
        # Moves model parts to RAM when not in use
        _pipe.enable_model_cpu_offload() 
        # Essential for Upscaling: Processes the image in tiles to prevent OOM crashes
        _pipe.enable_vae_tiling() 
        
    _loaded_model = model_name
@spaces.GPU
def upscale(
    image_path: str, 
    prompt: str = "", 
    model_name: str = UPSCALE_MODEL_DEFAULT, 
    steps: int = UPSCALE_STEPS_DEFAULT,
) -> str:
    """Upscale *image_path* 4× and return the saved file path."""
    # Ensure model is loaded
    _load(model_name)
    
    # Process the image
    low_res = Image.open(image_path).convert("RGB")
    
    # Run the pipeline
    result_image: Image.Image = _pipe(
        prompt=prompt, 
        image=low_res, 
        num_inference_steps=steps,
    ).images[0]
    
    # Save the result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"upscaled_{uuid.uuid4().hex[:8]}.png")
    result_image.save(out_path)
    
    return out_path