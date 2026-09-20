"""
Live-frame preprocessing  (Deep Learning + Data Engineering: avoids TRAIN/SERVE SKEW)

The training images are not raw camera frames: they are hand crops with the MediaPipe
skeleton drawn on top, a magenta box, letter-boxed on a white 300x300 canvas. A model
trained on that will misbehave on a plain webcam frame, so inference must recreate it:

    raw frame -> HandLandmarker (21 landmarks) -> draw skeleton on the frame -> crop with margin
              -> draw magenta box -> letterbox on a 300x300 white canvas

Uses the MediaPipe *Tasks* API (needs the `hand_landmarker.task` model file, see config).
NOTE: not executed during authoring (no model file / camera in the build sandbox). Before
trusting results, compare one output of `frame_to_canvas` side by side with a training image
and adjust DOT_RADIUS / LINE_THICKNESS / MARGIN until they look the same.
"""
from __future__ import annotations
import cv2
import numpy as np

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
DOT_RADIUS, LINE_THICKNESS, MARGIN, CANVAS = 5, 2, 20, 300
MAGENTA = (255, 0, 255)


class HandCanvasMaker:
    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=model_path),
                                            num_hands=1, running_mode=vision.RunningMode.IMAGE)
        self._det = vision.HandLandmarker.create_from_options(opts)

    def frame_to_canvas(self, rgb: np.ndarray):
        """rgb uint8 (H,W,3) -> 300x300 RGB canvas in the training format, or None if no hand."""
        h, w = rgb.shape[:2]
        res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.hand_landmarks:
            return None
        pts = np.array([(lm.x * w, lm.y * h) for lm in res.hand_landmarks[0]], dtype=np.float32)
        img = rgb.copy()
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (255, 255, 255), LINE_THICKNESS)
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), DOT_RADIUS, (255, 0, 0), -1)          # RGB red
        x0, y0 = np.floor(pts.min(0)).astype(int) - MARGIN
        x1, y1 = np.ceil(pts.max(0)).astype(int) + MARGIN
        cv2.rectangle(img, (x0, y0), (x1, y1), MAGENTA, 4)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        s = CANVAS / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
        canvas = np.full((CANVAS, CANVAS, 3), 255, np.uint8)
        oy, ox = (CANVAS - crop.shape[0]) // 2, (CANVAS - crop.shape[1]) // 2
        canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        return canvas
