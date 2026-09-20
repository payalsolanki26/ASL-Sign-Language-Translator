"""
Privacy-by-design utilities  (Data Privacy)

Hand images are personal data: hand geometry is biometric, and frames can capture a
face, background or room. Controls implemented here:

  1. pseudonymize_id   - replace filenames (which embed capture timestamps) with keyed-hash
                         IDs. The training manifest keeps only relative order (frame_idx,
                         take_id), never absolute time.
  2. strip_metadata    - re-encode the pixels only: EXIF (GPS, device, time) is dropped.
  3. blur_faces        - detect frontal faces and Gaussian-blur them before data leaves the
                         ingest boundary. Haar cascades miss partial / side faces, so treat
                         this as defence-in-depth, not a guarantee; review flagged samples.
  4. minimise          - drop columns the model does not need before sharing a manifest.
"""
from __future__ import annotations
import hashlib
import hmac
import io
import os

import cv2
import numpy as np
from PIL import Image

PSEUDO_ENV = "ASL_PSEUDO_SALT"


def pseudonymize_id(original_name: str, salt: str | None = None, length: int = 16) -> str:
    """Keyed hash (HMAC-SHA256). Without the salt an ID cannot be linked back to the file name."""
    salt = salt or os.environ.get(PSEUDO_ENV)
    if not salt:
        raise RuntimeError(f"Set {PSEUDO_ENV} to pseudonymize identifiers.")
    return "img_" + hmac.new(salt.encode(), original_name.encode(), hashlib.sha256).hexdigest()[:length]


_HARMLESS_INFO = {"jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi", "progressive", "progression"}


def strip_metadata(img_bytes: bytes, fmt: str = "JPEG", quality: int = 95) -> bytes:
    """Return the image without EXIF / ICC / comments.

    Images that carry no such metadata are returned byte-for-byte (no lossy re-encode).
    Otherwise the pixels are re-encoded on their own."""
    with Image.open(io.BytesIO(img_bytes)) as im:
        if len(im.getexif()) == 0 and not (set(im.info) - _HARMLESS_INFO):
            return img_bytes
        clean = Image.fromarray(np.asarray(im))
        buf = io.BytesIO()
        clean.save(buf, format=fmt, quality=quality)
        return buf.getvalue()


_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if _CASCADE.empty():
            raise RuntimeError("OpenCV face cascade not found")
    return _CASCADE


def detect_faces(rgb: np.ndarray, scale: float = 1.1, neighbors: int = 4, min_size: int = 30):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return _cascade().detectMultiScale(grey, scale, neighbors, minSize=(min_size, min_size))


def blur_faces(rgb: np.ndarray, **kw) -> tuple[np.ndarray, int]:
    """Blur detected faces. Returns (image, n_faces_blurred)."""
    out, faces = rgb.copy(), detect_faces(rgb, **kw)
    for x, y, w, h in faces:
        k = max(31, (w // 2) | 1)
        out[y:y + h, x:x + w] = cv2.GaussianBlur(out[y:y + h, x:x + w], (k, k), 0)
    return out, int(len(faces))
