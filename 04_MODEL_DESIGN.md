# 4. Model design (deep learning)

![model](../design/02_model_architecture.png)

## Task
27-way image classification (A-Z, `space`) from a 300x300 crop that already contains the hand skeleton.

## Input pipeline
Load -> paint magenta box white -> resize to 128x128 -> augment (train only) -> ImageNet normalise.

Augmentations: rotation +-12 deg, scale 0.9-1.1, shift 6% (white fill), colour jitter, random erasing.
**No horizontal flip:** a mirrored hand is a left-handed version of the sign, not the same picture.
Add it deliberately only if you also want to support left-handed signers.

## Networks (`configs/config.yaml -> model.name`)
| Name | Notes |
|---|---|
| `asl_net` | Residual CNN with squeeze-excite, about 2.8 M parameters (hand estimate: run `python -m asl.models.asl_net` for the exact count). Trains on CPU, fast on any GPU, exports cleanly. |
| `mobilenet_v3_small` | torchvision backbone, new 27-way head. `pretrained: true` uses ImageNet weights (needs internet). |

## Training recipe
Cross-entropy with label smoothing 0.1 and inverse-frequency class weights; AdamW (lr 2e-3, wd 1e-4);
OneCycle schedule; mixed precision on CUDA; batch 64; up to 40 epochs; early stop after 8 epochs with no gain in
**validation macro-F1**. The best checkpoint is saved with its metadata (classes, image size) and HMAC-signed.
The test block is used once, in `evaluate`.

## Evaluation protocol
1. Test-block accuracy, **macro-F1**, per-class F1, top confusions, calibration (ECE + reliability diagram).
2. Grad-CAM on correct and wrong samples: heat should sit on the hand and skeleton, not on the white canvas or background.
3. **New-signer hold-out** (recommended): the only test that speaks to real-world use.
4. Compare against the k-NN baseline in `reports/leakage_demo.json`; a deep model that cannot beat ~97% on the
   leak-safe test block is not adding value yet.

## Things to expect (hypotheses to check, not results)
* Confusions among closed-fist letters (A, E, M, N, S, T) and among U/V/R, K/P, G/Q.
* J and Z are motion signs; one frame captures only part of the gesture.
* The `space` class is a custom gesture and should be easy.
* High test scores on this dataset will not by themselves prove generalisation (see `02_DATA_ENGINEERING.md`).

## Train/serve skew: the important trap
The network is trained on skeleton-overlay images. A live app must therefore run MediaPipe, draw the
skeleton the same way, crop with a margin, draw the box, and letter-box on a white 300x300 canvas
(`serve/preprocess_live.py`). Before trusting live predictions, put one live canvas next to a training image and
tune `DOT_RADIUS`, `LINE_THICKNESS` and `MARGIN` until they match. Feeding a plain webcam frame will fail.

## Phase 2 options
| Option | Why |
|---|---|
| Landmark model: 21x3 keypoints, wrist-relative, scale-normalised, small MLP | Tiny, fast, robust to background and skin tone, and **no images are stored**, a privacy win. Needs landmarks re-extracted from raw recordings. |
| Temporal model: GRU or 1-D conv over a 16-frame landmark window | Handles J and Z properly, and supports whole-word spelling. |
| Export: ONNX / TFLite | On-device deployment. |
| Self-training / more signers | The biggest accuracy-for-effort gain in practice is more diverse data. |
