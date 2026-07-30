"""
Steel Defect Classifier — Streamlit Application.

Interactive frontend for steel defect classification. Browse images,
run inference, view Grad-CAM visualizations, and track inspection history.

Run with:
    streamlit run steel_defect/app.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steel_defect.inference import SteelPredictor
from steel_defect.gradcam import GradCAM
from steel_defect.preprocessing import overlay_gradcam
from steel_defect.utils import (
    setup_logging,
    DATA_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
)

logger = setup_logging("steel_defect.app")

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Defect Classification",
    page_icon="🔬",
    layout="wide",
)

# ── Session State ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False


# ── Helpers ────────────────────────────────────────────────────
@st.cache_resource
def load_predictor() -> SteelPredictor:
    """Load model once and cache across reruns."""
    predictor = SteelPredictor()
    predictor.load_model()
    return predictor


def get_available_categories() -> list[str]:
    """List class subdirectories that exist in the data folder."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name in CLASS_NAMES
    )


def get_images_in_category(category: str) -> list[Path]:
    """List all image files in a category directory."""
    cat_dir = DATA_DIR / category
    if not cat_dir.exists():
        return []
    return sorted(
        p for p in cat_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )


def generate_gradcam(predictor: SteelPredictor, image_tensor: torch.Tensor) -> np.ndarray:
    """Generate Grad-CAM heatmap from the last conv layer."""
    try:
        # Get the last convolutional layer from model.features
        target_layer = None
        for layer in reversed(list(predictor.model.features.children())):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
                break

        if target_layer is None:
            return np.zeros(IMAGE_SIZE, dtype=np.float32)

        cam = GradCAM(predictor.model, target_layer)
        heatmap = cam.generate(image_tensor.to(predictor.device))
        cam.remove_hooks()
        return heatmap
    except Exception as e:
        logger.warning("Grad-CAM failed: %s", e)
        return np.zeros(IMAGE_SIZE, dtype=np.float32)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Steel Defect Classifier")
    st.caption("CNN-based Steel Defect Classification")
    st.divider()

    st.subheader("Settings")
    gradcam_alpha = st.slider(
        "Grad-CAM opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
    )

    st.divider()

    st.subheader("Image Source")
    categories = get_available_categories()

    if not categories:
        st.error(
            "⚠️ No dataset found.\n\n"
            "Place images in subdirectories:\n"
            "`data/steel_defect/{no_defect,defect_1,...}/`"
        )
        st.stop()

    selected_category = st.selectbox(
        "Browse category",
        categories,
        help="Select a class folder to browse images from.",
        disabled=st.session_state.running,
    )

    st.divider()

    st.subheader("Auto-Inspect Speed")
    inspect_interval = st.slider(
        "Interval (sec)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.5,
    )

    st.divider()

    st.subheader("Inspection History")
    if st.session_state.history:
        st.metric("Inspected", len(st.session_state.history))
        class_counts = {}
        for h in st.session_state.history:
            lbl = h["label"]
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        cols = st.columns(min(len(class_counts), 3))
        for i, (cls_name, count) in enumerate(sorted(class_counts.items())):
            cols[i % len(cols)].metric(cls_name, count)

        if not st.session_state.running:
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
    else:
        st.caption("No inspections yet.")


# ── Main Content ───────────────────────────────────────────────
st.header("Steel Surface Inspection")

# Load images for selected category
image_paths = get_images_in_category(selected_category)
if not image_paths:
    st.warning(f"No images found in {selected_category}/")
    st.stop()

# Load model
try:
    predictor = load_predictor()
except (FileNotFoundError, NotImplementedError) as e:
    st.error(
        f"⚠️ Cannot load model.\n\n"
        f"Train first: `python -m steel_defect.train`\n\n"
        f"Error: {e}"
    )
    st.stop()

# ── Control Buttons ────────────────────────────────────────────
col_auto, col_manual, col_spacer = st.columns([1, 1, 2])

with col_auto:
    if st.session_state.running:
        if st.button("⏹  Stop", type="primary", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    else:
        if st.button("▶  Auto Inspect", type="primary", use_container_width=True):
            st.session_state.running = True
            st.rerun()

with col_manual:
    manual_btn = st.button(
        "🔬 Single Inspect",
        use_container_width=True,
        disabled=st.session_state.running,
    )

# ── Display Placeholders ──────────────────────────────────────
st.divider()
status_placeholder = st.empty()
col_img, col_cam, col_stats = st.columns([1, 1, 0.8])

with col_img:
    img_header = st.empty()
    img_display = st.empty()
    img_caption = st.empty()

with col_cam:
    cam_header = st.empty()
    cam_display = st.empty()
    cam_caption = st.empty()

with col_stats:
    stats_header = st.empty()
    stats_label = st.empty()
    stats_confidence = st.empty()
    stats_latency = st.empty()
    stats_scores = st.empty()

history_divider = st.empty()
history_header = st.empty()
history_cols_placeholder = st.empty()


def display_result(result: dict, image: np.ndarray, source: str):
    """
    APP-1: Display the prediction result in the Streamlit placeholders.

    This function receives a prediction result dict and the original image,
    and should fill in the UI placeholders defined above.

    The result dict contains (see inference.py for details):
        - result["label"]:        str, e.g., "defect_2"
        - result["confidence"]:   float, 0-1
        - result["class_scores"]: dict mapping class names to probabilities
        - result["predicted_idx"]:int, class index
        - result["latency_ms"]:   float

    Your task — fill the stats column:
        1. stats_header.subheader("Prediction")

        2. Display the predicted label with color:
           - If label is "no_defect": use stats_label.success(f"✅ **{label}**")
           - Otherwise: use stats_label.error(f"🔴 **{label}**")

        3. stats_confidence.metric("Confidence", f"{confidence:.1%}")

        4. stats_latency.metric("Latency", f"{latency_ms:.0f} ms")

        5. Display per-class scores in stats_scores using a container:
           with stats_scores.container():
               for class_name, score in result["class_scores"].items():
                   st.progress(score, text=f"{class_name}: {score:.1%}")

    The image display and Grad-CAM overlay are handled below (scaffold).

    Hint:
        label = result["label"]
        confidence = result["confidence"]
        latency_ms = result["latency_ms"]
    """
    # ── Scaffold: image display ───────────────────────────
    img_header.subheader("Input Image")
    display_img = cv2.resize(image, IMAGE_SIZE)
    img_display.image(display_img, use_container_width=True)
    img_caption.caption(f"Source: {source}")

    # ── Scaffold: Grad-CAM overlay ────────────────────────
    cam_header.subheader("Grad-CAM")
    try:
        from steel_defect.preprocessing import build_val_transforms
        transform = build_val_transforms()
        tensor = transform(image=cv2.resize(image, IMAGE_SIZE))["image"]
        tensor = tensor.unsqueeze(0)
        heatmap = generate_gradcam(predictor, tensor)
        blended = overlay_gradcam(display_img, heatmap, alpha=gradcam_alpha)
        cam_display.image(blended, use_container_width=True)
        cam_caption.caption("Red = high activation")
    except Exception:
        cam_display.info("Grad-CAM unavailable")

    # ┌──────────────────────────────────────────────────────┐
    # │  APP-1: Write your code below                        │
    # │  Fill stats_header, stats_label, stats_confidence,   │
    # │  stats_latency, and stats_scores using the result    │
    # │  dict. See docstring above for details.              │
    # └──────────────────────────────────────────────────────┘
    
    label = result["label"]
    confidence = result["confidence"]
    latency_ms = result["latency_ms"]

    stats_header.subheader("Prediction")

    if label is "no_defect": 
        stats_label.success(f"✅ **{label}**")
    else:
        stats_label.error(f"🔴 **{label}**")

    stats_confidence.metric("Confidence", f"{confidence:.1%}")

    stats_latency.metric("Latency", f"{latency_ms:.0f} ms")

    with stats_scores.container():
        for class_name, score in result["class_scores"].items():
            st.progress(score, text=f"{class_name}: {score:.1%}")


def display_history():
    """Render recent inspection history."""
    if not st.session_state.history:
        return

    history_divider.divider()
    history_header.subheader("Recent Inspections")

    recent = list(reversed(st.session_state.history[-10:]))
    with history_cols_placeholder.container():
        cols = st.columns(min(len(recent), 5))
        for i, item in enumerate(recent[:5]):
            with cols[i]:
                icon = "✅" if item["label"] == "no_defect" else "🔴"
                st.markdown(f"**{icon} {item['label']}**")
                st.caption(f"Conf: {item['confidence']:.1%}")
                st.caption(f"Source: {item.get('source', 'N/A')}")


# ── Inspection Logic ───────────────────────────────────────────
current_image_idx = 0

if manual_btn and not st.session_state.running:
    image_path = image_paths[0]
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    result = predictor.predict(image)
    result["source"] = image_path.name
    st.session_state.history.append(result)
    display_result(result, image, image_path.name)
    display_history()

if st.session_state.running:
    total = len(image_paths)
    status_placeholder.info(
        f"🔄 **Auto-inspecting** — {selected_category} "
        f"({total} images) — every {inspect_interval:.1f}s"
    )

    inspected = 0
    while st.session_state.running:
        idx = inspected % total
        image_path = image_paths[idx]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

        result = predictor.predict(image)
        result["source"] = image_path.name
        st.session_state.history.append(result)

        display_result(result, image, image_path.name)
        display_history()

        inspected += 1
        time.sleep(inspect_interval)

        if inspected >= total:
            inspected = 0

if not st.session_state.running and not manual_btn:
    display_history()
