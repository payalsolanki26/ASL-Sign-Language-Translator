# Live ASL Fingerspelling Translator & Classifier

A real-time American Sign Language (ASL) fingerspelling recognition and translation system treating **data engineering, deep learning, security/privacy, and accessible human interface design** as a unified product.

![pipeline map](design/01_pipeline_map.png)

---

## Key Capabilities

1. **Live Camera Fingerspelling**: Real-time webcam capture with MediaPipe hand landmark extraction.
2. **Train/Serve Skew Mitigation**: Web frames are processed to match the training data representation (21 landmark keypoints, white skeleton connections, red landmark dots, letterboxed onto a 300×300 white canvas, with magenta shortcut box masking).
3. **Temporal Stabilization & Debounce**: Prevents character spam (e.g. `AAAA`). Commits a character only after being stably held for ~0.8s with a smooth circular progress indicator, followed by a debounce cooldown.
4. **Persistent Word Builder**: Word accumulation with interactive controls: **[Add Space]**, **[Backspace]**, **[Clear All]**, and **[Copy Text]**.
5. **Confidence Gating**: Configurable confidence threshold (default 55%). Predictions below threshold are marked `⚠ Not sure — adjust hand` and are never committed.
6. **Multi-Hand & No-Hand Safety**: Detects when multiple hands are in view (`Please show one hand`) or when no hand is present (`Place one hand inside frame`), cleanly pausing recognition.
7. **J and Z Motion Notice**: Clearly flags `J` and `Z` as motion-dependent signs requiring motion gestures.
8. **Security & Privacy by Design**: HMAC-SHA256 model signing, zero disk frame storage, in-memory processing only, client-side/local execution by default.

---

## Directory Structure

```
HAND SIGN/
├── configs/
│   └── config.yaml            # Single source of truth for all pipeline & model parameters
├── data/                      # Manifests and data directories
├── docs/                      # Technical design documents (architecture, security, model card, data card)
├── design/                    # Architectural diagrams and visual designs
├── artifacts/                 # Signed model checkpoint (asl_model.pt), MediaPipe task, audit log
├── reports/                   # Figures, metrics.json, data quality, and dashboard.html
├── src/
│   └── asl/
│       ├── config.py          # Configuration loader and path resolver
│       ├── data/              # Ingestion, validation, and temporal-block splitting
│       ├── models/            # AslNet architecture (~2.8M params with Squeeze-and-Excitation)
│       ├── security/          # Crypto, integrity signing, audit chaining, privacy
│       ├── serve/             # FastAPI backend, preprocessing, and static web UI
│       │   ├── api.py
│       │   ├── preprocess_live.py
│       │   └── static/
│       │       └── index.html # Full-featured interactive translator web app
│       ├── training/          # Training loop, early stopping, and evaluation
│       └── viz/               # EDA plots, Grad-CAM, and HTML dashboard
├── scripts/
│   ├── run_pipeline.py        # CLI for every stage: ingest, split, train, evaluate, dashboard
│   └── make_diagrams.py
├── tests/
│   ├── test_core.py           # Core integrity, audit, crypto, privacy, and split tests
│   └── test_live.py           # Live preprocessing, stabilization, word builder, and API tests
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Download MediaPipe Hand Landmarker Model
```powershell
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'artifacts/hand_landmarker.task')"
```

### 3. Run the Test Suite (All 31 Tests)
```powershell
python -m unittest discover -s tests -v
```

### 4. Launch the Live Web Application

**Option A (One-Click Launchers in Project Folder):**
- On **Windows**: Double-click [`run_app.bat`](file:///c:/Users/pratham/Downloads/HAND%20SIGN/run_app.bat)
- On **PowerShell**: Run `.\run_app.ps1`
- On **Bash / Git Bash / Linux**: Run `./run_app.sh`

**Option B (Manual Terminal Command):**
```powershell
python -m uvicorn asl.serve.api:app --app-dir src --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser. Click **Start Camera** to begin real-time translation.

---

## Dataset & Leak-Safe Temporal Splitting

* **Dataset Size**: 9,676 valid 300×300 images across 27 classes (`A`–`Z` plus `space`).
* **Source Frame Capture**: Captured from continuous 30 FPS video recordings. Adjacent frames differ by milliseconds and are virtually identical.
* **Why Random Splitting Fails**: Naive random train/test splitting causes severe data leakage where nearly identical frames land in both train and test sets, artificially inflating accuracy scores (a simple 32×32 k-NN gets 100% on a random split).
* **Temporal-Block Split with Embargo**:
  * Each class recording is partitioned into 6 contiguous blocks.
  * Blocks are assigned cleanly to Train, Validation, and Test.
  * An embargo of 10 frames on either side of validation/test boundaries is dropped to eliminate transition leakage.
  * **Resulting Split**: 5,373 train / 1,613 validation / 1,610 test / 1,080 embargo frames.

---

## Model Architecture & Verified Performance

### Architecture (`AslNet`)
* **Type**: Residual Convolutional Neural Network with Squeeze-and-Excitation (SE) channel attention.
* **Parameters**: 2,804,923 (~2.8M parameters).
* **Input Size**: 128×128 RGB, ImageNet normalized.
* **Output**: 27 class logits (A–Z, space).
* **Loss & Optimizer**: CrossEntropyLoss with Label Smoothing (0.1), Inverse-Frequency Class Weights, AdamW, and OneCycle learning rate schedule.

### Measured Evaluation Results (Leak-Safe Test Block)
*Evaluated strictly on the untouched test block of 1,610 samples (no data leakage):*

| Metric | Score |
|---|---|
| **Test Accuracy** | **92.24%** (1,485 / 1,610 correct) |
| **Macro-F1** | **0.9084** |
| **Expected Calibration Error (ECE)** | **0.0928** |
| **Inference Latency** | **~35 ms (GPU) / ~300 ms (CPU)** |
| **Checkpoint Integrity** | **HMAC-SHA256 Verified** |

Evaluation reports, confusion matrix, per-class F1, reliability diagrams, and Grad-CAM visualizations are saved in `reports/figures/` and aggregated in `reports/dashboard.html`.

---

## Live Translation UI & Stabilization Specification

1. **Dark Navy Camera Stage**: High contrast stage with live video and SVG overlay for the 21-point hand skeleton.
2. **Recognition HUD**:
   * Prominent letter display (clamp 72px–110px).
   * Confidence percentage readout.
   * Visual badge with icon and explicit text: `✓ Confident` (green), `⚠ Not sure — adjust hand` (amber), `✋ No hand detected` (muted), `👥 Please show one hand` (warning).
3. **Word Builder**:
   * Individual letter tiles for translated text.
   * Active pending tile with animated progress ring filling over 0.8s.
   * Debounce cooldown preventing repeated characters while holding.
   * `[ ⎵ Add Space ]`, `[ ⌫ Backspace ]`, `[ 🗑 Clear All ]`, `[ 📋 Copy Text ]`.
4. **Top 3 Guesses**: Real-time alternative prediction bars showing relative probabilities (e.g. M vs N near-misses).
5. **Interactive Controls & Tuning**:
   * Adjustable Confidence Threshold slider (40%–90%).
   * Adjustable Hold Duration slider (0.4s–1.5s).
   * Mode switch: **Live Camera** vs **Demo Mode** (simulated sequence for testing without camera).

---

## Known Limitations & Design Boundaries

1. **J and Z Motion Gestures**: J and Z in ASL are dynamic gestures involving movement (tracing a 'J' or 'Z' shape). A single static video frame cannot capture the full trajectory. The UI explicitly marks J and Z with a warning (`⚠ Motion sign — hold steady or check twice`). A Phase 2 temporal model (e.g., GRU over a 16-frame landmark sequence) is recommended for continuous sign dynamics.
2. **Single Hand Operation**: Designed for one-handed fingerspelling. Showing multiple hands triggers a safety warning.
3. **Generalization Across New Signers**: The current dataset originates from a limited number of recording sessions. While high accuracy is achieved on the leak-safe split, testing with a diverse hold-out group of new signers (different skin tones, hand sizes, lighting, and cameras) is essential for real-world deployment.
