"""
providers/face_swap_provider.py – Face-swap using InsightFace / inswapper.

Requires `inswapper_128.onnx` to be placed in a `models/` directory next to
this project.  Download from the InsightFace model zoo or Buffalo_L pack.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from config import FACE_SWAP_MODEL, OUTPUT_DIR

_app = None
_swapper = None


def _load() -> None:
    global _app, _swapper
    if _app is not None:
        return

    import insightface
    from insightface.app import FaceAnalysis

    _app = FaceAnalysis(name="buffalo_l")
    _app.prepare(ctx_id=0, det_size=(640, 640))

    model_path = Path(__file__).parent.parent / "models" / FACE_SWAP_MODEL
    if not model_path.exists():
        raise FileNotFoundError(
            f"Face-swap model not found at {model_path}. "
            "Download inswapper_128.onnx and place it in the models/ directory."
        )
    _swapper = insightface.model_zoo.get_model(str(model_path), providers=["CPUExecutionProvider"])


def swap(source_image_path: str, target_image_path: str) -> str:
    """Swap the face from *source* onto *target* and return the output path."""
    _load()

    source = cv2.imread(source_image_path)
    target = cv2.imread(target_image_path)
    if source is None:
        raise ValueError(f"Could not read source image: {source_image_path}")
    if target is None:
        raise ValueError(f"Could not read target image: {target_image_path}")

    source_faces = _app.get(source)
    target_faces = _app.get(target)

    if not source_faces:
        raise ValueError("No face detected in the source image.")
    if not target_faces:
        raise ValueError("No face detected in the target image.")

    result = target.copy()
    source_face = source_faces[0]
    for face in target_faces:
        result = _swapper.get(result, face, source_face, paste_back=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"faceswap_{uuid.uuid4().hex[:8]}.jpg")
    cv2.imwrite(out_path, result)
    return out_path
