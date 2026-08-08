"""
providers/ltx_providers.py

LTX-Video provider for extending an existing video.
"""

from __future__ import annotations
import spaces 
import os
from pathlib import Path

import torch
from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video


# ============================================================
# LTX CONFIGURATION
# ============================================================

LTX_MODEL = os.environ.get(
    "LTX_MODEL",
    "Lightricks/LTX-Video",
)

LTX_FPS = 24

LTX_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, jittery motion, "
    "flickering, distorted, warped anatomy, deformed face, "
    "identity drift, inconsistent motion"
)


_PIPELINE = None


# ============================================================
# LOAD PIPELINE
# ============================================================

def _get_pipeline():
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    if not torch.cuda.is_available():
        raise RuntimeError(
            "LTX requires a CUDA GPU. "
            "No CUDA GPU is currently available."
        )

    print(f"[LTX] Loading model: {LTX_MODEL}")

    _PIPELINE = LTXConditionPipeline.from_pretrained(
        LTX_MODEL,
        torch_dtype=torch.bfloat16,
    )

    _PIPELINE.to("cuda")

    try:
        _PIPELINE.vae.enable_tiling()
    except Exception:
        pass

    print("[LTX] Pipeline loaded successfully.")

    return _PIPELINE


# ============================================================
# EXTEND VIDEO
# ============================================================
@spaces.GPU(duration=300)
def extend_video(
    video_path,
    prompt: str,
    extension_frames: int = 81,
    seed: int = 0,
):
    """
    Extend the END of an existing video using LTX.

    The final 81 frames of the source video are used as
    conditioning frames. LTX then generates the continuation.

    Returns:
        Path to the generated MP4.
    """

    if not video_path:
        raise ValueError(
            "Please provide a video."
        )

    if not prompt or not prompt.strip():
        raise ValueError(
            "Please provide an extension prompt."
        )

    video_path = str(video_path)

    if not Path(video_path).exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    print(
        f"[LTX] Loading source video: {video_path}"
    )

    source_video = load_video(video_path)

    if not source_video:
        raise RuntimeError(
            "LTX could not read the input video."
        )

    source_count = len(source_video)

    print(
        f"[LTX] Source contains {source_count} frames."
    )

    # --------------------------------------------------------
    # LTX conditioning requirement:
    #
    # Input conditioning video must contain:
    #
    # 8n + 1 frames
    #
    # 81 frames is therefore a good starting point.
    # --------------------------------------------------------

    conditioning_frames = 81

    if source_count < conditioning_frames:
        raise ValueError(
            "The source video must contain at least "
            "81 frames for this first LTX extension test."
        )

    conditioning_video = source_video[-conditioning_frames:]

    # --------------------------------------------------------
    # Determine resolution from source.
    # --------------------------------------------------------

    first_frame = conditioning_video[0]

    width, height = first_frame.size

    # LTX works best with dimensions divisible by 32.
    width = width - (width % 32)
    height = height - (height % 32)

    if width < 256 or height < 256:
        raise ValueError(
            "The source video resolution is too small for LTX."
        )

    if first_frame.size != (width, height):
        conditioning_video = [
            frame.resize((width, height))
            for frame in conditioning_video
        ]

    print(
        f"[LTX] Resolution: {width}x{height}"
    )

    # --------------------------------------------------------
    # Convert requested extension length to a valid
    # LTX frame count.
    #
    # For the first test:
    #
    # 81 frames = 3.375 seconds @ 24 FPS
    # --------------------------------------------------------

    extension_frames = int(extension_frames)

    if extension_frames < 81:
        extension_frames = 81

    extension_frames = (
        ((extension_frames - 1) // 8) * 8
    ) + 1

    # The generated sequence contains the conditioning
    # segment plus the continuation.
    total_frames = conditioning_frames + extension_frames - 1

    # Target frame must be a multiple of 8.
    total_frames = (
        (total_frames // 8) * 8
    )

    print(
        f"[LTX] Conditioning frames: "
        f"{conditioning_frames}"
    )

    print(
        f"[LTX] Target output frames: "
        f"{total_frames}"
    )

    print(
        f"[LTX] Approximate output duration: "
        f"{total_frames / LTX_FPS:.2f} seconds"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = _get_pipeline()

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(int(seed))

    # The source video occupies frames 0 → 80.
    # LTX generates the continuation after it.
    condition = LTXVideoCondition(
        video=conditioning_video,
        frame_index=0,
    )

    print(
        "[LTX] Generating continuation..."
    )

    result = pipeline(
        conditions=[condition],
        prompt=prompt.strip(),
        negative_prompt=LTX_NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_frames=total_frames,
        frame_rate=LTX_FPS,
        num_inference_steps=30,
        guidance_scale=3.0,
        image_cond_noise_scale=0.15,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        generator=generator,
    )

    frames = result.frames[0]

    if not frames:
        raise RuntimeError(
            "LTX returned no frames."
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output_dir = Path("outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "ltx_extended_video.mp4"
    )

    export_to_video(
        frames,
        str(output_path),
        fps=LTX_FPS,
    )

    print(
        f"[LTX] Video saved to: {output_path}"
    )

    return str(output_path)
