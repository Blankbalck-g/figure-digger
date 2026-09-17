"""Unicode-safe image I/O.

OpenCV's `cv2.imread` / `cv2.imwrite` go through the ANSI code page on Windows, so a
path containing non-ASCII characters (a Chinese paper title, a folder name, a user
name) silently fails: `imread` returns None and the whole figure is skipped without any
error. Round-tripping through NumPy bytes avoids the code page entirely.
"""

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that also works for non-ASCII paths (returns None if unreadable)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    if img is None:
        # fall back to the plain call: covers exotic cases and keeps old behaviour
        img = cv2.imread(str(path), flags)
    return img


def imwrite(path, img):
    """cv2.imwrite that also works for non-ASCII paths."""
    path = str(path)
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
        return True
    return bool(cv2.imwrite(path, img))
