"""
Sign Reader — Streamlit Web Interface for ASL Fingerspelling Classifier
Root-level entry point compatible with local execution and Streamlit Community Cloud.
"""
import io
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch

# Ensure both root directory and src directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Robust import strategy to handle both structured (src/asl/...) and flattened repo layouts
try:
    from asl.config import load_config, resolve
    from asl.data.dataset import build_transforms, mask_magenta
    from asl.models.asl_net import build_model
    from asl.serve.preprocess_live import HandCanvasMaker
except ImportError:
    try:
        from config import load_config, resolve
        from dataset import build_transforms, mask_magenta
        from asl_net import build_model
        from preprocess_live import HandCanvasMaker
    except ImportError as e:
        st.error(f"Failed to import project modules: {e}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Sign Reader — Live ASL Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #16202E;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #5D6878;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: #F8FAFC;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-letter {
        font-size: 5rem;
        font-weight: 800;
        color: #1E8A63;
        line-height: 1;
        margin: 10px 0;
    }
    .prediction-letter.unsure {
        color: #B77A08;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-ok { background: #E6F4EA; color: #1E8A63; }
    .status-warn { background: #FEF3D6; color: #B77A08; }
    .status-err { background: #FCE8E6; color: #C8553D; }
</style>
""", unsafe_allow_html=True)


def get_checkpoint_path() -> Path:
    """Locate the trained model checkpoint across common project locations."""
    candidates = [
        ROOT_DIR / "artifacts" / "asl_model.pt",
        ROOT_DIR / "asl_model.pt",
        SRC_DIR / "asl" / "artifacts" / "asl_model.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return ROOT_DIR / "artifacts" / "asl_model.pt"


def get_landmarker_path() -> Path:
    """Locate or download the MediaPipe Hand Landmarker model."""
    candidates = [
        ROOT_DIR / "artifacts" / "hand_landmarker.task",
        ROOT_DIR / "hand_landmarker.task",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Target path for auto-download (essential for Streamlit Community Cloud)
    target = ROOT_DIR / "artifacts" / "hand_landmarker.task"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        with st.spinner("Downloading MediaPipe Hand Landmarker model..."):
            urllib.request.urlretrieve(url, str(target))
        return target
    except Exception as e:
        st.warning(f"Could not automatically download MediaPipe task model: {e}")
        return target


@st.cache_resource(show_spinner=False)
def load_asl_pipeline():
    """Load configuration, trained PyTorch model checkpoint, and HandCanvasMaker."""
    cfg = load_config()

    # 1. Model Checkpoint
    ckpt_path = get_checkpoint_path()
    if not ckpt_path.exists():
        return None, None, None, f"Model checkpoint not found at: `{ckpt_path}`. Please ensure `asl_model.pt` is committed or placed in `artifacts/`."

    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        return None, None, None, f"Failed to load checkpoint `{ckpt_path}`: {e}"

    meta = ck.get("meta", {})
    classes = meta.get("classes", [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space"
    ])
    image_size = meta.get("image_size", 128)
    mask_magenta_box = meta.get("mask_magenta_box", True)

    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    tf = build_transforms(image_size, False, cfg["train"]["augment"])

    # 2. MediaPipe Hand Canvas Maker
    task_file = get_landmarker_path()
    if not task_file.exists():
        return None, None, None, f"MediaPipe model not found at `{task_file}`. Please download `hand_landmarker.task` into `artifacts/`."

    try:
        maker = HandCanvasMaker(str(task_file))
    except Exception as e:
        return None, None, None, f"Failed to initialize MediaPipe HandLandmarker: {e}"

    pipeline = {
        "model": model,
        "classes": classes,
        "transform": tf,
        "mask_magenta": mask_magenta_box,
        "maker": maker,
        "checkpoint_path": ckpt_path,
        "min_confidence": cfg.get("serve", {}).get("min_confidence", 0.38)
    }
    return pipeline, None, None, None


def main():
    st.markdown('<div class="main-title">🤟 Sign Reader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Live American Sign Language (ASL) Fingerspelling Translator</div>', unsafe_allow_html=True)

    pipeline, _, _, err = load_asl_pipeline()
    if err:
        st.error(err)
        st.info("💡 **Deployment Note**: Make sure `artifacts/asl_model.pt` and `artifacts/hand_landmarker.task` are present in your Git repository or storage.")
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        min_conf = st.slider("Confidence Threshold (%)", min_value=25, max_value=85, value=int(pipeline["min_confidence"] * 100), step=1) / 100.0
        st.markdown("---")
        st.markdown("### 📋 Model Info")
        st.markdown(f"**Architecture**: `AslNet` (~2.8M params)")
        st.markdown(f"**Classes**: 27 (`A–Z` + `space`)")
        st.markdown(f"**Checkpoint**: `{pipeline['checkpoint_path'].name}`")
        st.markdown("---")
        st.markdown("### 🛡 Privacy")
        st.caption("All images are processed strictly in memory and are never persisted to disk.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📷 Camera / Image Input")
        input_mode = st.radio("Choose Input Method", ["Webcam Snapshot", "Upload Image"], horizontal=True)

        input_img = None
        if input_mode == "Webcam Snapshot":
            camera_file = st.camera_input("Take a snapshot of your hand sign")
            if camera_file is not None:
                input_img = Image.open(camera_file).convert("RGB")
        else:
            upload_file = st.file_uploader("Upload an ASL hand image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if upload_file is not None:
                input_img = Image.open(upload_file).convert("RGB")

    with col2:
        st.subheader("🔍 Prediction Results")

        if input_img is None:
            st.info("👉 Take a snapshot or upload an image to view the ASL translation.")
            return

        rgb = np.asarray(input_img)

        # Run MediaPipe Hand Preprocessing (matches training data format)
        canvas, hand_status, info = pipeline["maker"].process_frame(rgb)

        # Fallback for preprocessed dataset canvases
        if canvas is None and rgb.shape[:2] == (300, 300):
            red_pts = (rgb[..., 0] > 160) & (rgb[..., 1] < 70) & (rgb[..., 2] < 70)
            if red_pts.sum() >= 15:
                canvas = rgb
                hand_status = "ok"

        if hand_status == "no_hand_detected":
            st.warning("✋ **No hand detected**: Please hold one hand clearly facing the camera.")
            return

        if hand_status == "multiple_hands":
            st.warning("👥 **Multiple hands detected**: Please show only one hand at a time.")
            return

        # Prepare image for model inference
        proc_pil = Image.fromarray(canvas)
        if pipeline["mask_magenta"]:
            proc_pil = mask_magenta(proc_pil)

        inp_tensor = pipeline["transform"](proc_pil).unsqueeze(0)

        with torch.no_grad():
            logits = pipeline["model"](inp_tensor)[0]
            probs = torch.softmax(logits, dim=0)

        topk = torch.topk(probs, 5)
        top_classes = [pipeline["classes"][idx.item()] for idx in topk.indices]
        top_confs = [float(v.item()) for v in topk.values]

        top_label = top_classes[0]
        top_conf = top_confs[0]
        is_confident = top_conf >= min_conf
        display_letter = "␣ (space)" if top_label == "space" else top_label

        # Prediction Card
        card_class = "status-ok" if is_confident else "status-warn"
        status_text = "✓ Confident" if is_confident else "⚠ Adjusting / Lower Confidence"

        st.markdown(f"""
        <div class="prediction-card">
            <div class="status-pill {card_class}">{status_text}</div>
            <div class="prediction-letter {'unsure' if not is_confident else ''}">{display_letter}</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #16202E;">Confidence: {top_conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if top_label in ("J", "Z"):
            st.info("ℹ️ **Motion Sign**: J and Z are motion-dependent gestures in ASL.")

        # Top Alternatives
        st.markdown("#### Top Predictions")
        for lbl, conf in zip(top_classes[:3], top_confs[:3]):
            disp = "space" if lbl == "space" else lbl
            st.write(f"**{disp}** — `{conf*100:.1f}%`")
            st.progress(conf)

        with st.expander("🖼 View Preprocessed Model Input"):
            st.image(canvas, caption="300×300 Letterboxed Canvas with Hand Landmarks", use_container_width=True)


if __name__ == "__main__":
    main()
