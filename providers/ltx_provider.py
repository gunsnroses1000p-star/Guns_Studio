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
# LTX TIMESTEP SCHEDULE
# ============================================================

def _linear_quadratic_schedule(
    num_steps: int,
    threshold_noise: float = 0.025,
):
    """
    LTX's recommended linear/quadratic timestep schedule.

    This is used explicitly so newer Diffusers schedulers do not
    require a dynamic-shifting `mu` value.
    """

    linear_steps = num_steps // 2

    if num_steps < 2:
        return torch.tensor(
            [1.0],
            dtype=torch.float32,
        )

    linear_sigma_schedule = [
        i * threshold_noise / linear_steps
        for i in range(linear_steps)
    ]

    threshold_noise_step_diff = (
        linear_steps
        - threshold_noise * num_steps
    )

    quadratic_steps = (
        num_steps - linear_steps
    )

    quadratic_coef = (
        threshold_noise_step_diff
        / (linear_steps * quadratic_steps**2)
    )

    linear_coef = (
        threshold_noise / linear_steps
        - (
            2
            * threshold_noise_step_diff
            / quadratic_steps**2
        )
    )

    const = quadratic_coef * (
        linear_steps**2
    )

    quadratic_sigma_schedule = [
        quadratic_coef * (i**2)
        + linear_coef * i
        + const
        for i in range(
            linear_steps,
            num_steps,
        )
    ]

    sigma_schedule = (
        linear_sigma_schedule
        + quadratic_sigma_schedule
        + [1.0]
    )

    sigma_schedule = [
        1.0 - x
        for x in sigma_schedule
    ]

    return torch.tensor(
        sigma_schedule[:-1],
        dtype=torch.float32,
    )


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

    _PIPELINE = (
        LTXConditionPipeline.from_pretrained(
            LTX_MODEL,
            torch_dtype=torch.bfloat16,
        )
    )

    _PIPELINE.to("cuda")

    try:
        _PIPELINE.vae.enable_tiling()
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
    extension_frames: int = 81,
    seed: int = 0,
):

    """
    Extend the END of an existing video using LTX.

    The final 81 frames of the source video are used as
    conditioning context.

    Seed remains backend-only.
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
    #
    # LTX requires:
    #
    # 8n + 1 frames
    #
    # 81 frames = 8 * 10 + 1
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

    width = width - (
        width % 32
    )

    height = height - (
        height % 32
    )

    if width < 256 or height < 256:
        raise ValueError(
            "The source video resolution is too small "
            "for LTX."
        )

    if first_frame.size != (
        width,
        height,
    ):

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
    # EXTENSION LENGTH
    # --------------------------------------------------------

    extension_frames = int(
        extension_frames
    )

    if extension_frames < 81:
        extension_frames = 81

    extension_frames = (
        (
            (extension_frames - 1)
            // 8
        )
        * 8
    ) + 1

    total_frames = (
        conditioning_frames
        + extension_frames
        - 1
    )

    total_frames = (
        total_frames // 8
    ) * 8

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
    # PIPELINE
    # --------------------------------------------------------

    pipeline = _get_pipeline()

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(
        int(seed)
    )

    condition = LTXVideoCondition(
        video=conditioning_video,
        frame_index=0,
    )

    # --------------------------------------------------------
    # EXPLICIT LTX TIMESTEPS
    #
    # This avoids the dynamic-shifting `mu` error.
    # --------------------------------------------------------

    num_inference_steps = 30

    timesteps = (
        _linear_quadratic_schedule(
            num_inference_steps
        )
        * 1000.0
    )

    print(
        "[LTX] Using explicit LTX timestep schedule."
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
        num_inference_steps=num_inference_steps,
        timesteps=timesteps,
        guidance_scale=1.0,
        image_cond_noise_scale=0.025,
        decode_timestep=0.05,
        decode_noise_scale=0.025,
        generator=generator,
    )

    frames = result.frames[0]

    if not frames:
        raise RuntimeError(
            "LTX returned no frames."
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
