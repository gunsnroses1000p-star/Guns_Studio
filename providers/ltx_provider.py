"""
providers/ltx_provider.py

AnyFlow-FAR video continuation provider.

The filename is intentionally kept as ltx_provider.py for now
so the existing Guns AI Studio app import does not need to change.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import spaces
import torch

from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video, load_video


# ============================================================
# ANYFLOW CONFIGURATION
# ============================================================

ANYFLOW_MODEL = os.environ.get(
    "ANYFLOW_MODEL",
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers",
)

# Official examples use 16 FPS.
ANYFLOW_FPS = 16

# AnyFlow-FAR released checkpoint is built around 81 output frames.
ANYFLOW_OUTPUT_FRAMES = 81

# 33 = 4 * 8 + 1
#
# We give AnyFlow the last 33 frames of the existing video
# as V2V context.
#
# 33 context frames = about 2.06 seconds at 16 FPS.
#
# 81 output frames = about 5.06 seconds total.
#
# Therefore the new continuation is approximately:
#
# 81 - 33 = 48 frames
# 48 / 16 = 3 seconds
#
ANYFLOW_CONTEXT_FRAMES = 33

ANYFLOW_WIDTH = 832
ANYFLOW_HEIGHT = 480

ANYFLOW_STEPS = 4

ANYFLOW_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, "
    "flickering, jittery motion, unstable motion, "
    "warped anatomy, distorted face, deformed face, "
    "identity drift, changing person, duplicate person, "
    "extra limbs, unnatural movement, "
    "cartoon, anime, CGI, 3d render"
)


# ============================================================
# PIPELINE CACHE
# ============================================================

_PIPELINE = None


# ============================================================
# LOAD PIPELINE
# ============================================================

def _get_pipeline():
    """
    Load AnyFlow-FAR once and keep it cached.
    """

    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    if not torch.cuda.is_available():
        raise RuntimeError(
            "AnyFlow-FAR requires a CUDA GPU. "
            "No CUDA GPU is currently available."
        )

    print(
        f"[ANYFLOW] Loading model: {ANYFLOW_MODEL}"
    )

    print(
        "[ANYFLOW] Loading AnyFlowFARPipeline "
        "in bfloat16..."
    )

    _PIPELINE = AnyFlowFARPipeline.from_pretrained(
        ANYFLOW_MODEL,
        torch_dtype=torch.bfloat16,
    )

    # --------------------------------------------------------
    # MOVE TO GPU
    # --------------------------------------------------------

    print(
        "[ANYFLOW] Moving pipeline to CUDA..."
    )

    _PIPELINE.to("cuda")

    # --------------------------------------------------------
    # VAE TILING
    # --------------------------------------------------------

    try:
        _PIPELINE.vae.enable_tiling()

        print(
            "[ANYFLOW] VAE tiling enabled."
        )

    except Exception as exc:
        print(
            "[ANYFLOW] VAE tiling unavailable: "
            f"{exc}"
        )

    print(
        "[ANYFLOW] Pipeline loaded successfully."
    )

    return _PIPELINE


# ============================================================
# PREPARE VIDEO CONTEXT
# ============================================================

def _prepare_context(source_video):
    """
    Take the final 33 frames of the source video and convert
    them into the tensor format expected by AnyFlow-FAR.

    Required format:

        (B, T, C, H, W)

    Values:

        [0, 1]

    T must satisfy:

        T = 4n + 1

    33 satisfies this requirement.
    """

    if len(source_video) < ANYFLOW_CONTEXT_FRAMES:
        raise ValueError(
            "The source video must contain at least "
            f"{ANYFLOW_CONTEXT_FRAMES} frames."
        )

    context_frames = source_video[
        -ANYFLOW_CONTEXT_FRAMES:
    ]

    resized_frames = []

    for frame in context_frames:
        resized = frame.resize(
            (
                ANYFLOW_WIDTH,
                ANYFLOW_HEIGHT
            )
        )

        resized_frames.append(
            np.asarray(resized)
            .astype(np.float32)
            / 255.0
        )

    # --------------------------------------------------------
    # PIL / NumPy:
    #
    # (T, H, W, C)
    #
    # AnyFlow:
    #
    # (B, T, C, H, W)
    # --------------------------------------------------------

    video_array = np.stack(
        resized_frames,
        axis=0,
    )

    context = torch.from_numpy(
        video_array
    ).permute(
        0,
        3,
        1,
        2,
    )

    context = context.unsqueeze(0)

    context = context.to(
        device="cuda",
        dtype=torch.float32,
    )

    return context


# ============================================================
# EXTEND VIDEO
# ============================================================

@spaces.GPU(duration=300)
def extend_video(
    video_path,
    prompt: str,
    extension_frames: int = 48,
    seed: int = 0,
):
    """
    Extend the END of an existing video using AnyFlow-FAR.

    Workflow:

        Existing video
              |
              v
        Last 33 frames
              |
              v
        AnyFlow-FAR V2V
              |
              v
        81-frame continuation window
              |
              v
        Remove the 33-frame overlap
              |
              v
        Append ~48 new frames
              |
              v
        Full extended MP4

    NOTE:
    extension_frames is retained in the function signature so
    the existing app.py interface does not need to change.

    The released AnyFlow-FAR checkpoint generates 81-frame
    windows. We therefore use the first 33 frames as context
    and append the remaining 48 generated frames.
    """

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

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
        "=================================================="
    )

    print(
        "[ANYFLOW] Starting video extension."
    )

    print(
        f"[ANYFLOW] Input video: {video_path}"
    )

    print(
        f"[ANYFLOW] Prompt: {prompt.strip()}"
    )

    print(
        f"[ANYFLOW] Context frames: "
        f"{ANYFLOW_CONTEXT_FRAMES}"
    )

    print(
        f"[ANYFLOW] Output frames: "
        f"{ANYFLOW_OUTPUT_FRAMES}"
    )

    print(
        f"[ANYFLOW] Inference steps: "
        f"{ANYFLOW_STEPS}"
    )

    print(
        f"[ANYFLOW] FPS: {ANYFLOW_FPS}"
    )

    # --------------------------------------------------------
    # LOAD SOURCE VIDEO
    # --------------------------------------------------------

    print(
        "[ANYFLOW] Loading source video..."
    )

    source_video = load_video(
        video_path
    )

    if not source_video:
        raise RuntimeError(
            "AnyFlow could not read the input video."
        )

    source_count = len(source_video)

    print(
        f"[ANYFLOW] Source contains "
        f"{source_count} frames."
    )

    if source_count < ANYFLOW_CONTEXT_FRAMES:
        raise ValueError(
            "The source video is too short. "
            f"AnyFlow requires at least "
            f"{ANYFLOW_CONTEXT_FRAMES} frames."
        )

    # --------------------------------------------------------
    # LOAD PIPELINE
    # --------------------------------------------------------

    pipeline = _get_pipeline()

    # --------------------------------------------------------
    # PREPARE V2V CONTEXT
    # --------------------------------------------------------

    print(
        "[ANYFLOW] Preparing final "
        f"{ANYFLOW_CONTEXT_FRAMES} frames "
        "for V2V conditioning..."
    )

    context = _prepare_context(
        source_video
    )

    print(
        f"[ANYFLOW] Context tensor shape: "
        f"{tuple(context.shape)}"
    )

    # Expected:

    # (1, 33, 3, 480, 832)

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(
        int(seed)
    )

    print(
        f"[ANYFLOW] Seed: {int(seed)}"
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print(
        "[ANYFLOW] Generating continuation..."
    )

    print(
        "[ANYFLOW] Using V2V conditioning."
    )

    print(
        "[ANYFLOW] Generating 81-frame window..."
    )

    with torch.inference_mode():

        result = pipeline(
            prompt=prompt.strip(),
            video=context,
            negative_prompt=ANYFLOW_NEGATIVE_PROMPT,
            width=ANYFLOW_WIDTH,
            height=ANYFLOW_HEIGHT,
            num_frames=ANYFLOW_OUTPUT_FRAMES,
            num_inference_steps=ANYFLOW_STEPS,
            guidance_scale=1.0,
            use_mean_velocity=True,
            use_kv_cache=True,
            generator=generator,
        )

    # --------------------------------------------------------
    # EXTRACT GENERATED FRAMES
    # --------------------------------------------------------

    generated_frames = result.frames[0]

    if generated_frames is None:
        raise RuntimeError(
            "AnyFlow returned no video frames."
        )

    if len(generated_frames) == 0:
        raise RuntimeError(
            "AnyFlow returned an empty video."
        )

    print(
        f"[ANYFLOW] Generated "
        f"{len(generated_frames)} frames."
    )

    # --------------------------------------------------------
    # CALCULATE ACTUAL EXTENSION
    # --------------------------------------------------------

    new_frames = generated_frames[
        ANYFLOW_CONTEXT_FRAMES:
    ]

    if len(new_frames) == 0:
        raise RuntimeError(
            "AnyFlow produced no new frames after "
            "the conditioning section."
        )

    print(
        f"[ANYFLOW] New continuation frames: "
        f"{len(new_frames)}"
    )

    print(
        f"[ANYFLOW] New continuation duration: "
        f"{len(new_frames) / ANYFLOW_FPS:.2f} seconds"
    )

    # --------------------------------------------------------
    # PREPARE ORIGINAL VIDEO
    # --------------------------------------------------------

    original_frames = []

    for frame in source_video:
        frame = frame.resize(
            (
                ANYFLOW_WIDTH,
                ANYFLOW_HEIGHT,
            )
        )

        original_frames.append(
            frame
        )

    # --------------------------------------------------------
    # COMBINE
    #
    # Keep the ENTIRE original video.
    #
    # Append only the newly generated frames.
    #
    # This prevents the 33-frame conditioning section
    # from being duplicated in the final MP4.
    # --------------------------------------------------------

    final_frames = (
        original_frames
        + list(new_frames)
    )

    print(
        f"[ANYFLOW] Final frame count: "
        f"{len(final_frames)}"
    )

    print(
        f"[ANYFLOW] Final duration: "
        f"{len(final_frames) / ANYFLOW_FPS:.2f} seconds"
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
        / "anyflow_extended_video.mp4"
    )

    print(
        "[ANYFLOW] Exporting MP4..."
    )

    export_to_video(
        final_frames,
        str(output_path),
        fps=ANYFLOW_FPS,
    )

    print(
        f"[ANYFLOW] Video saved to: "
        f"{output_path}"
    )

    print(
        "=================================================="
    )

    return str(output_path)
