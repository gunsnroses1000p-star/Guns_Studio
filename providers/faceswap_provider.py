"""
providers/faceswap_provider.py — Face swap for images and videos.
"""

import io
import base64
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from PIL import Image

from utils.face import load_face_analyzer


def face_swap_image(
    source_image: Image.Image,
    target_image: Image.Image,
) -> Image.Image:
    """
    Swap the face detected in *source_image* onto *target_image*.

    Uses InsightFace's ``get`` method to detect source face embedding, then
    an ``inswapper`` model to perform the swap.
    """
    if source_image is None:
        raise gr.Error("Please upload a source (face) image.")
    if target_image is None:
        raise gr.Error("Please upload a target image.")

    try:
        import insightface
        from insightface.model_zoo import get_model

        analyzer = load_face_analyzer()
        src_arr = np.asarray(source_image.convert("RGB"))
        tgt_arr = np.asarray(target_image.convert("RGB"))

        src_faces = analyzer.get(src_arr)
        tgt_faces = analyzer.get(tgt_arr)

        if not src_faces:
            raise gr.Error("No face detected in the source image.")
        if not tgt_faces:
            raise gr.Error("No face detected in the target image.")

        # Load inswapper model (will be downloaded to ~/.insightface on first call)
        swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])

        result_arr = tgt_arr.copy()
        for tgt_face in tgt_faces:
            result_arr = swapper.get(result_arr, tgt_face, src_faces[0], paste_back=True)

        return Image.fromarray(result_arr)

    except gr.Error:
        raise
    except Exception as err:
        raise gr.Error(f"Face swap failed: {err}")


def face_swap_video(
    source_image: Image.Image,
    video_path: str,
) -> str:
    """
    Swap the face from *source_image* into every frame of *video_path*.
    Returns the path to the processed video.
    """
    if source_image is None:
        raise gr.Error("Please upload a source (face) image.")
    if not video_path:
        raise gr.Error("Please upload a video.")

    try:
        import cv2
        import insightface
        from insightface.model_zoo import get_model

        analyzer = load_face_analyzer()
        swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])

        src_arr = np.asarray(source_image.convert("RGB"))
        src_faces = analyzer.get(src_arr)
        if not src_faces:
            raise gr.Error("No face detected in the source image.")
        source_face = src_faces[0]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = str(Path("outputs") / "face_swap_video.mp4")
        Path("outputs").mkdir(exist_ok=True)
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tgt_faces = analyzer.get(frame_rgb)
            result = frame_rgb.copy()
            for face in tgt_faces:
                result = swapper.get(result, face, source_face, paste_back=True)
            writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        cap.release()
        writer.release()
        return out_path

    except gr.Error:
        raise
    except Exception as err:
        raise gr.Error(f"Video face swap failed: {err}")
