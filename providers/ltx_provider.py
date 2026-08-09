"""
providers/ltx_provider.py

AnyFlow-FAR video continuation provider.

The filename is intentionally retained so app.py does not need
to change yet.
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
# CONFIG
# ============================================================

ANYFLOW_MODEL = os.environ.get(
    "ANYFLOW_MODEL",
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers",
)

ANYFLOW_FPS = 16

# Canonical released AnyFlow-FAR configuration.
ANYFLOW_OUTPUT_FRAMES = 81

# 33 = 4n + 1
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
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    if not torch.cuda.is_available():
        raise RuntimeError(
            "AnyFlow-FAR requires CUDA, but no CUDA GPU "
            "is currently available."
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

    print(
        "[ANYFLOW] Moving pipeline to CUDA..."
    )

    _PIPELINE.to("cuda")

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
    Convert the final 33 source frames into:

        (1, 33, 3, 480, 832)

    AnyFlow V2V requires T = 4n + 1.
    """

    if len(source_video) < ANYFLOW_CONTEXT_FRAMES:
        raise ValueError(
            "The source video must contain at least "
            f"{ANYFLOW_CONTEXT_FRAMES} frames."
        )

    context_frames = source_video[
        -ANYFLOW_CONTEXT_FRAMES:
    ]

    frames = []

    for frame in context_frames:

        frame = frame.resize(
            (
                ANYFLOW_WIDTH,
                ANYFLOW_HEIGHT,
            )
        )

        arr = (
            np.asarray(frame)
            .astype(np.float32)
            / 255.0
        )

        frames.append(arr)

    video_array = np.stack(
        frames,
        axis=0,
    )

    # T,H,W,C -> T,C,H,W
    context = torch.from_numpy(
        video_array
    ).permute(
        0,
        3,
        1,
        2,
    )

    # T,C,H,W -> B,T,C,H,W
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
    extension_frames: int = 81,
    seed: int = 0,
):
    """
    Extend an existing video using AnyFlow-FAR.

    AnyFlow-FAR generates an 81-frame window using the final
    33 frames of the current video as conditioning context.

    Each generation therefore contributes:

        81 generated frames
        - 33 conditioning frames
        = 48 new frames

    For longer requested extensions, multiple continuation passes
    are performed. Each pass uses the newly extended video's final
    33 frames as the next conditioning context.

    The requested extension_frames value is therefore treated as
    the approximate number of NEW frames to append.
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

    requested_extension = int(extension_frames)

    if requested_extension <= 0:
        raise ValueError(
            "Extension length must be greater than zero."
        )

    print("=" * 58)
    print("[ANYFLOW] Starting video extension.")
    print(
        f"[ANYFLOW] Input video: {video_path}"
    )
    print(
        f"[ANYFLOW] Requested NEW frames: "
        f"{requested_extension}"
    )
    print(
        f"[ANYFLOW] Prompt: {prompt.strip()}"
    )
    print(
        f"[ANYFLOW] Context frames: "
        f"{ANYFLOW_CONTEXT_FRAMES}"
    )
    print(
        f"[ANYFLOW] Generation window: "
        f"{ANYFLOW_OUTPUT_FRAMES}"
    )
    print(
        f"[ANYFLOW] New frames per pass: "
        f"{ANYFLOW_OUTPUT_FRAMES - ANYFLOW_CONTEXT_FRAMES}"
    )
    print(
        f"[ANYFLOW] Inference steps: "
        f"{ANYFLOW_STEPS}"
    )
    print(
        f"[ANYFLOW] FPS: "
        f"{ANYFLOW_FPS}"
    )

    # --------------------------------------------------------
    # LOAD INITIAL VIDEO
    # --------------------------------------------------------

    print(
        "[ANYFLOW] Loading source video..."
    )

    current_video = load_video(
        video_path
    )

    if not current_video:
        raise RuntimeError(
            "AnyFlow could not read the input video."
        )

    print(
        f"[ANYFLOW] Initial source contains "
        f"{len(current_video)} frames."
    )

    if len(current_video) < ANYFLOW_CONTEXT_FRAMES:
        raise ValueError(
            "The source video is too short. "
            f"At least {ANYFLOW_CONTEXT_FRAMES} "
            "frames are required."
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    pipeline = _get_pipeline()

    # --------------------------------------------------------
    # CALCULATE PASSES
    # --------------------------------------------------------

    new_frames_per_pass = (
        ANYFLOW_OUTPUT_FRAMES
        - ANYFLOW_CONTEXT_FRAMES
    )

    if new_frames_per_pass <= 0:
        raise RuntimeError(
            "AnyFlow configuration is invalid: "
            "output frames must be greater than "
            "context frames."
        )

    required_passes = int(
        np.ceil(
            requested_extension
            / new_frames_per_pass
        )
    )

    print(
        f"[ANYFLOW] Required continuation passes: "
        f"{required_passes}"
    )

    # --------------------------------------------------------
    # GENERATOR
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
    # CONTINUATION LOOP
    # --------------------------------------------------------

    total_new_frames = 0

    for pass_index in range(
        required_passes
    ):

        print("-" * 58)

        print(
            f"[ANYFLOW] Continuation pass "
            f"{pass_index + 1}/{required_passes}"
        )

        print(
            f"[ANYFLOW] Current video frames: "
            f"{len(current_video)}"
        )

        # ----------------------------------------------------
        # PREPARE CURRENT CONTEXT
        # ----------------------------------------------------

        context_frames = current_video[
            -ANYFLOW_CONTEXT_FRAMES:
        ]

        frames = []

        for frame in context_frames:

            frame = frame.resize(
                (
                    ANYFLOW_WIDTH,
                    ANYFLOW_HEIGHT,
                )
            )

            arr = (
                np.asarray(frame)
                .astype(np.float32)
                / 255.0
            )

            frames.append(arr)

        video_array = np.stack(
            frames,
            axis=0,
        )

        # T,H,W,C -> T,C,H,W

        context = torch.from_numpy(
            video_array
        ).permute(
            0,
            3,
            1,
            2,
        )

        # T,C,H,W -> B,T,C,H,W

        context = context.unsqueeze(0)

        context = context.to(
            device="cuda",
            dtype=torch.float32,
        )

        print(
            f"[ANYFLOW] Context tensor shape: "
            f"{tuple(context.shape)}"
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        print(
            "[ANYFLOW] Starting AnyFlow-FAR..."
        )

        print(
            "[ANYFLOW] Using KV cache."
        )

        print(
            "[ANYFLOW] Using mean velocity."
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
                use_mean_velocity=True,
                use_kv_cache=True,
                generator=generator,
            )

        # ----------------------------------------------------
        # GET GENERATED FRAMES
        # ----------------------------------------------------

        generated_frames = result.frames[0]

        if generated_frames is None:
            raise RuntimeError(
                "AnyFlow returned no frames."
            )

        if len(generated_frames) == 0:
            raise RuntimeError(
                "AnyFlow returned an empty video."
            )

        print(
            f"[ANYFLOW] Generated window: "
            f"{len(generated_frames)} frames."
        )

        # ----------------------------------------------------
        # REMOVE CONDITIONING OVERLAP
        # ----------------------------------------------------

        new_frames = generated_frames[
            ANYFLOW_CONTEXT_FRAMES:
        ]

        if not new_frames:
            raise RuntimeError(
                "No new frames remained after "
                "removing the conditioning overlap."
            )

        # ----------------------------------------------------
        # ONLY TAKE WHAT WE STILL NEED
        # ----------------------------------------------------

        remaining_frames = (
            requested_extension
            - total_new_frames
        )

        frames_to_add = min(
            len(new_frames),
            remaining_frames,
        )

        new_frames = list(
            new_frames[:frames_to_add]
        )

        print(
            f"[ANYFLOW] New frames from pass: "
            f"{len(new_frames)}"
        )

        # ----------------------------------------------------
        # APPEND TO CURRENT VIDEO
        # ----------------------------------------------------

        current_video = (
            list(current_video)
            + new_frames
        )

        total_new_frames += len(
            new_frames
        )

        print(
            f"[ANYFLOW] Total new frames so far: "
            f"{total_new_frames}"
        )

        print(
            f"[ANYFLOW] Current total frames: "
            f"{len(current_video)}"
        )

        # ----------------------------------------------------
        # DONE?
        # ----------------------------------------------------

        if total_new_frames >= requested_extension:

            print(
                "[ANYFLOW] Requested extension "
                "length reached."
            )

            break

    # --------------------------------------------------------
    # FINAL VIDEO
    # --------------------------------------------------------

    print("-" * 58)

    print(
        f"[ANYFLOW] Final source + extension frames: "
        f"{len(current_video)}"
    )

    print(
        f"[ANYFLOW] Actual new frames added: "
        f"{total_new_frames}"
    )

    print(
        f"[ANYFLOW] Added duration: "
        f"{total_new_frames / ANYFLOW_FPS:.2f} seconds"
    )

    print(
        f"[ANYFLOW] Final duration: "
        f"{len(current_video) / ANYFLOW_FPS:.2f} seconds"
    )

    # --------------------------------------------------------
    # RESIZE FINAL VIDEO
    # --------------------------------------------------------

    final_frames = []

    for frame in current_video:

        frame = frame.resize(
            (
                ANYFLOW_WIDTH,
                ANYFLOW_HEIGHT,
            )
        )

        final_frames.append(
            frame
        )

    # --------------------------------------------------------
    # EXPORT
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
        f"[ANYFLOW] Video saved: "
        f"{output_path}"
    )

    print("=" * 58)

    return str(output_path)
