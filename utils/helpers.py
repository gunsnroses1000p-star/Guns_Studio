"""
utils/helpers.py – Shared utility functions.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def unique_filename(prefix: str, ext: str, directory: str) -> str:
    """Return a unique file path inside *directory*."""
    os.makedirs(directory, exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.{ext.lstrip('.')}"
    return os.path.join(directory, name)


def ensure_dir(path: str) -> str:
    """Create *path* if it doesn't exist and return it."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def pil_to_gradio(image) -> str:
    """Save a PIL Image to a temp file and return its path for Gradio."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name)
        return f.name
