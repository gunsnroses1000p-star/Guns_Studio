"""
utils/face.py — Face analysis, preservation, and eye-safe restoration.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

_face_analyzer = None


def load_face_analyzer():
    """Lazy-load the InsightFace FaceAnalysis model (CPU)."""
    global _face_analyzer
    if _face_analyzer is None:
        import insightface  # optional heavy dependency

        _face_analyzer = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _face_analyzer


def preserve_original_face(
    original_image: Image.Image,
    edited_image: Image.Image,
    strength: float = 0.65,
) -> Image.Image:
    """
    Blend the original face region back into *edited_image* to prevent
    facial distortion introduced by the generation step.
    """
    original_image = original_image.convert("RGB")
    edited_image = edited_image.convert("RGB").resize(
        original_image.size, Image.LANCZOS
    )

    try:
        analyzer = load_face_analyzer()
        faces = analyzer.get(np.asarray(original_image))
        if not faces:
            print("Face preservation: no face detected.", flush=True)
            return edited_image

        face = max(
            faces,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        )
        x1, y1, x2, y2 = face.bbox.astype(int)
        fw = x2 - x1
        fh = y2 - y1
        pad_x = int(fw * 0.20)
        pad_top = int(fh * 0.20)
        pad_bot = int(fh * 0.12)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_top)
        x2 = min(original_image.width, x2 + pad_x)
        y2 = min(original_image.height, y2 + pad_bot)

        original_face = original_image.crop((x1, y1, x2, y2))
        edited_face = edited_image.crop((x1, y1, x2, y2))

        strength = max(0.0, min(1.0, float(strength)))
        blended_face = Image.blend(edited_face, original_face, strength)

        mask = Image.new("L", original_face.size, 0)
        draw = ImageDraw.Draw(mask)
        mw, mh = mask.size
        mx = int(mw * 0.08)
        my = int(mh * 0.05)
        draw.ellipse((mx, my, mw - mx, mh - my), fill=255)
        blur_r = max(8, int(min(mw, mh) * 0.12))
        mask = mask.filter(ImageFilter.GaussianBlur(blur_r))

        result = edited_image.copy()
        result.paste(blended_face, (x1, y1), mask)
        return result

    except Exception as err:
        print(f"Face preservation failed: {err}", flush=True)
        return edited_image


def restore_face_eye_safe(image_path: str | None) -> str | None:
    """
    Apply a very light sharpening pass to a face image (eye-safe strength)
    and return the path to the saved result.
    """
    if image_path is None:
        return None
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.filter(
            ImageFilter.UnsharpMask(radius=0.4, percent=25, threshold=6)
        )
        image = ImageEnhance.Contrast(image).enhance(1.02)
        image = ImageEnhance.Sharpness(image).enhance(1.02)
        output_path = "outputs/face_eye_safe.png"
        image.save(output_path)
        return output_path
    except Exception:
        return image_path
