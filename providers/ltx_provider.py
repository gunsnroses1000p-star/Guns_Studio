"""
providers/ltx_provider.py

LTX-Video provider for extending an existing video.
"""

from __future__ import annotations

import os
from pathlib import Path

import spaces
import torch

from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXVideoCondition,
)
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
    "worst quality, low quality, blurry, inconsistent motion, "
    "jittery, distorted, flickering, warped anatomy, "
    "deformed face, identity drift"
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

    print(
        f"[LTX] Loading model: {LTX_MODEL}"
    )

    _PIPELINE = LTXConditionPipeline.from_pretrained(
        LTX_MODEL,
        torch_dtype=torch.bfloat16,
    )

    # --------------------------------------------------------
    # FIX CURRENT DIFFUSERS / LTX SCHEDULER COMPATIBILITY
    #
    # Newer Diffusers versions can load the LTX scheduler
    # with dynamic shifting enabled. That scheduler requires
    # a `mu` value every time set_timesteps() is called.
    #
    # LTXConditionPipeline does not currently provide that
    # value in this code path, so generation fails with:
    #
    # ValueError:
    # `mu` must be passed when `use_dynamic_shifting`
    # is set to be `True`
    #
    # Disable dynamic shifting and let the scheduler use its
    # normal timestep calculation instead.
    # --------------------------------------------------------

    try:
        from diffusers import FlowMatchEulerDiscreteScheduler

        _PIPELINE.scheduler = (
            FlowMatchEulerDiscreteScheduler.from_config(
                _PIPELINE.scheduler.config,
                use_dynamic_shifting=False,
            )
        )

        print(
            "[LTX] Scheduler patched: "
            "dynamic shifting disabled."
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to configure the LTX scheduler: "
            f"{exc}"
        ) from exc

    # --------------------------------------------------------
    # MOVE TO GPU
    # --------------------------------------------------------

    _PIPELINE.to("cuda")

    # --------------------------------------------------------
    # VAE TILING
    # --------------------------------------------------------

    try:
        _PIPELINE.vae.enable_tiling()
        print(
            "[LTX] VAE tiling enabled."
        )
    except Exception:
        pass

    print(
        "[LTX] Pipeline loaded successfully."
    )

    return _PIPELINE



# ============================================================
# EXTEND VIDEO
# ============================================================
@spaces.GPU(duration=300)
def extend_video(
    video_path,
    prompt: str,
    extension_frames: int = 9,
    seed: int = 0,
):
    """
    Extend the END of an existing video using LTX.

    Controlled diagnostic test:
    - 81 conditioning frames
    - 9-frame extension request
    - 89 total output frames
    - 24 FPS
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

    # --------------------------------------------------------
    # LOAD SOURCE VIDEO
    # --------------------------------------------------------

    print(
        f"[LTX] Loading source video: "
        f"{video_path}"
    )

    source_video = load_video(
        video_path
    )

    if not source_video:
        raise RuntimeError(
            "LTX could not read the input video."
        )

    source_count = len(source_video)

    print(
        f"[LTX] Source contains "
        f"{source_count} frames."
    )

    # --------------------------------------------------------
    # CONDITIONING
    # --------------------------------------------------------

    conditioning_frames = 81

    if source_count < conditioning_frames:
        raise ValueError(
            "The source video must contain at least "
            "81 frames for LTX extension."
        )

    conditioning_video = (
        source_video[-conditioning_frames:]
    )

    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    first_frame = conditioning_video[0]

    width, height = first_frame.size

    # LTX works best with dimensions divisible by 32.
    width = width - (width % 32)
    height = height - (height % 32)

    if width < 256 or height < 256:
        raise ValueError(
            "The source video resolution is too small "
            "for LTX."
        )

    if first_frame.size != (width, height):
        conditioning_video = [
            frame.resize(
                (width, height)
            )
            for frame in conditioning_video
        ]

    print(
        f"[LTX] Resolution: "
        f"{width}x{height}"
    )

    # --------------------------------------------------------
    # CONTROLLED FRAME TEST
    #
    # 81 conditioning frames
    # + 9 extension frames
    # - 1 overlapping frame
    # = 89 total frames
    #
    # 89 = 8 * 11 + 1
    # --------------------------------------------------------

    conditioning_frames = 81
    extension_frames = 9

    total_frames = (
        conditioning_frames
        + extension_frames
        - 1
    )

    # Safety check for LTX 8n + 1 requirement.
    if (total_frames - 1) % 8 != 0:
        total_frames = (
            ((total_frames - 1) // 8) * 8
        ) + 1

    print(
        f"[LTX] Conditioning frames: "
        f"{conditioning_frames}"
    )

    print(
        f"[LTX] Extension frames: "
        f"{extension_frames}"
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
    # LOAD PIPELINE
    # --------------------------------------------------------

    pipeline = _get_pipeline()

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(
        int(seed)
    )

    # --------------------------------------------------------
    # CONDITION
    # --------------------------------------------------------

    condition = LTXVideoCondition(
        video=conditioning_video,
        frame_index=0,
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print(
        "[LTX] Using scheduler default "
        "timestep handling."
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
        guidance_scale=1.0,
        image_cond_noise_scale=0.025,
        decode_timestep=0.05,
        decode_noise_scale=0.025,
        generator=generator,
    )

    # --------------------------------------------------------
    # EXTRACT FRAMES
    # --------------------------------------------------------

    frames = result.frames[0]

    if not frames:
        raise RuntimeError(
            "LTX returned no frames."
        )

    print(
        f"[LTX] Generated "
        f"{len(frames)} frames."
    )

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "ltx_extended_video.mp4"
    )

    export_to_video(
        frames,
        str(output_path),
        fps=LTX_FPS,
    )

    print(
        f"[LTX] Video saved to: "
        f"{output_path}"
    )

    return str(output_path)
