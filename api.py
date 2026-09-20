"""
Inference API  (Deep Learning + Data Security & Privacy)

    export ASL_API_KEY=...  ASL_SIGNING_KEY=...  ASL_LOG_SALT=...
    uvicorn asl.serve.api:app --host 127.0.0.1 --port 8000        (run from ./src)

Security / privacy controls
    * model file signature verified at startup; torch.load(weights_only=True)
    * API-key auth, constant-time comparison
    * upload size cap and pixel cap (decompression-bomb guard); only JPEG/PNG accepted
    * per-client sliding-window rate limit (client id = keyed hash of the IP, not the IP)
    * images are processed IN MEMORY and never written to disk or logged
    * logs contain: request id, latency, predicted label, confidence bucket - nothing else
    * strict CORS allow-list; security headers; low-confidence answers say "uncertain"
Terminate TLS in front of this service (reverse proxy) - do not expose plain HTTP publicly.
NOTE: not executed during authoring (FastAPI / PyTorch / MediaPipe model unavailable there).
"""
from __future__ import annotations
import hashlib
import hmac
import io
import logging
import os
import time
import uuid
from collections import defaultdict, deque

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from asl.config import load_config, resolve
from asl.data.dataset import build_transforms, mask_magenta
from asl.models.asl_net import build_model
from asl.security import integrity
from asl.serve.preprocess_live import HandCanvasMaker

cfg = load_config()
S = cfg["serve"]
Image.MAX_IMAGE_PIXELS = S["max_pixels"]
log = logging.getLogger("asl.api"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

app = FastAPI(title="ASL classifier", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=S["allowed_origins"], allow_methods=["POST", "GET"], allow_headers=["X-API-Key"])

_state: dict = {}
_hits: dict[str, deque] = defaultdict(deque)


@app.on_event("startup")
def _load():
    import torch
    ckpt = resolve(cfg, "artifacts_dir") / "asl_model.pt"
    if not integrity.verify_file(ckpt):
        raise RuntimeError("model signature invalid or missing - refusing to start")
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = build_model(cfg); model.load_state_dict(ck["state_dict"]); model.eval()
    _state.update(model=model, classes=ck["meta"]["classes"], mask=ck["meta"]["mask_magenta_box"],
                  tf=build_transforms(ck["meta"]["image_size"], False, cfg["train"]["augment"]),
                  hands=HandCanvasMaker(str(resolve(cfg, "artifacts_dir") / "hand_landmarker.task")))


@app.middleware("http")
async def _headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.update({"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
                         "Referrer-Policy": "no-referrer", "Content-Security-Policy": "default-src 'none'"})
    return resp


def _auth(request: Request):
    key = os.environ.get("ASL_API_KEY", "")
    if not key or not hmac.compare_digest(request.headers.get("X-API-Key", ""), key):
        raise HTTPException(401, "invalid API key")


def _rate_limit(request: Request):
    salt = os.environ.get("ASL_LOG_SALT", "dev")
    cid = hmac.new(salt.encode(), (request.client.host if request.client else "?").encode(), hashlib.sha256).hexdigest()[:12]
    now, q = time.time(), _hits[cid]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= S["rate_limit_per_minute"]:
        raise HTTPException(429, "rate limit exceeded")
    q.append(now)
    request.state.client_id = cid


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", dependencies=[Depends(_auth), Depends(_rate_limit)])
async def predict(request: Request, file: UploadFile = File(...)):
    import torch
    t0, rid = time.time(), uuid.uuid4().hex[:10]
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "only JPEG or PNG images are accepted")
    data = await file.read(S["max_upload_bytes"] + 1)
    if len(data) > S["max_upload_bytes"]:
        raise HTTPException(413, "image too large")
    try:
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(400, "could not decode image")
    del data                                                    # nothing is persisted
    canvas = _state["hands"].frame_to_canvas(rgb)
    if canvas is None:
        return {"request_id": rid, "status": "no_hand_detected"}
    img = Image.fromarray(canvas)
    img = mask_magenta(img) if _state["mask"] else img
    with torch.no_grad():
        p = torch.softmax(_state["model"](_state["tf"](img).unsqueeze(0)), 1)[0]
    top = torch.topk(p, S["top_k"])
    preds = [{"label": _state["classes"][i], "confidence": round(float(c), 4)} for c, i in zip(top.values, top.indices)]
    status = "ok" if preds[0]["confidence"] >= S["min_confidence"] else "uncertain"
    log.info("rid=%s client=%s status=%s top1=%s conf_bucket=%.1f ms=%d", rid, request.state.client_id, status,
             preds[0]["label"], int(preds[0]["confidence"] * 10) / 10, (time.time() - t0) * 1000)
    return {"request_id": rid, "status": status, "predictions": preds}
