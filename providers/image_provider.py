""" providers/image_provider.py – Text-to-image generation via Diffusers. """
from __future__ import annotations
import os
import spaces
from PIL import Image
from config import (
    HF_TOKEN, 
    IMAGE_MODEL_DEFAULT, 
    IMAGE_STEPS_DEFAULT, 
    IMAGE_GUIDANCE_DEFAULT, 
    IMAGE_WIDTH_DEFAULT, 
    IMAGE_HEIGHT_DEFAULT, 
    OUTPUT_DIR,
)

# Global variables to store the model and state
_pipe = None
_loaded_model = ""

def _load(model_name: str) -> None:
    global _pipe, _loaded_model
    
    # 1. If the correct model is already loaded, do nothing
    if _loaded_model == model_name and _pipe is not None:
        return

    # 2. Lazy import torch and diffusers ONLY when needed
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    
    print(f"Loading image model {model_name} into VRAM... please wait.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 3. Load the pipeline
    _pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        token=HF_TOKEN or None,
    )
    
    # Set the scheduler for better quality/speed
    _pipe.scheduler = DPMSolverMultistepScheduler.from_config(_pipe.scheduler.config)
    _pipe = _pipe.to(device)
    
    # 4. RTX 6000 Optimizations
    if device == "cuda":
        # Moves parts of the model to RAM when not in use, preventing crashes
        _pipe.enable_model_cpu_offload() 
        # Allows generating large images by processing them in tiles
        _pipe.enable_vae_tiling() 
        
    _loaded_model = model_name
@spaces.GPU
def generate(
    prompt: str, 
    negative_prompt: str = "", 
    model_name: str = IMAGE_MODEL_DEFAULT, 
    steps: int = IMAGE_STEPS_DEFAULT, 
    guidance_scale: float = IMAGE_GUIDANCE_DEFAULT, 
    width: int = IMAGE_WIDTH_DEFAULT, 
    height: int = IMAGE_HEIGHT_DEFAULT, 
    seed: int = -1,
) -> str:
    """Generate an image and return the saved file path."""
    import torch # Import locally for seed generation
    
    # Ensure model is loaded
    _load(model_name)
    
    generator = None
    if seed >= 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Run the pipeline
    image: Image.Image = _pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt or None, 
        num_inference_steps=steps, 
        guidance_scale=guidance_scale, 
        width=width, 
        height=height, 
        generator=generator,
    ).images[0]
    
    # Save the file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"image_{_unique_id()}.png")
    image.save(out_path)
    
    return out_path

def _unique_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]