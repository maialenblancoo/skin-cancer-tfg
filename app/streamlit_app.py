"""
Streamlit application for multimodal skin lesion classification.
Model: EfficientNet-B0 + clinical metadata (age + anatomical location)
Dataset: HAM10000 | TFG - Maialen Blanco Ibarra, Universidad de Deusto
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Hugging Face model download ───────────────────────────────────────────────
from huggingface_hub import hf_hub_download

# ── Path setup (works both locally and on Streamlit Cloud) ────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import SkinLesionModel
from src.xai import (
    run_gradcam,
    run_smoothgrad,
    overlay_heatmap,
    run_shap_metadata,
    METADATA_FEATURE_NAMES,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma ⚠",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}
MEL_IDX         = 4
MEL_THRESHOLD   = 0.31   # selected on validation set (no data leakage)
UNCERTAINTY_THR = 0.70   # below this → refer to specialist
METADATA_DIM    = 16

# Anatomical locations in strict alphabetical order (must match training)
LOCATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face",
    "foot", "genital", "hand", "lower extremity",
    "neck", "scalp", "trunk", "unknown", "upper extremity",
]

# Locations with high melanoma prevalence but underrepresented in training
# (from shortcut-learning analysis: prev_mel > 15% AND n_total < 100)
HIGH_RISK_UNDERREPRESENTED = {
    "ear":  31.7,
    "face": 18.0,
    "neck": 16.7,
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    """Download weights from HF Hub and load E09 model."""
    weights_path = hf_hub_download(
        repo_id="maialenblancoo/skin-cancer-pfg",
        filename="model.pth",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SkinLesionModel(metadata_dim=METADATA_DIM)
    state  = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def color_constancy(img_array: np.ndarray, power: float = 6.0) -> np.ndarray:
    """Shades-of-Gray color constancy normalization (power=6.0, as in training)."""
    img = img_array.astype(np.float32)
    norm = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    scale = norm.mean() / (norm + 1e-6)
    img = np.clip(img * scale, 0, 255).astype(np.uint8)
    return img

def build_metadata_vector(age: float, location: str) -> torch.Tensor:
    """Build the 16-dim metadata vector (age normalised + location one-hot)."""
    vec = np.zeros(METADATA_DIM, dtype=np.float32)
    vec[0] = age / 90.0                          # age normalised by max (90)
    if location in LOCATIONS:
        vec[1 + LOCATIONS.index(location)] = 1.0
    return torch.tensor(vec).unsqueeze(0)        # shape (1, 16)


def get_tta_transforms():
    """6 geometric TTA transforms used during training (no colour jitter —
    colour constancy already normalises illumination variability)."""
    base = [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ]
    return [
        T.Compose(base),                                              # 0 original
        T.Compose([T.RandomHorizontalFlip(p=1.0)] + base),           # 1 hflip
        T.Compose([T.RandomVerticalFlip(p=1.0)]   + base),           # 2 vflip
        T.Compose([T.RandomHorizontalFlip(p=1.0),
                   T.RandomVerticalFlip(p=1.0)]   + base),           # 3 hv flip
        T.Compose([T.RandomRotation((90, 90))]    + base),           # 4 rot 90
        T.Compose([T.RandomRotation((270, 270))]  + base),           # 5 rot 270
    ]


@torch.no_grad()
def predict_tta(model, device, pil_img: Image.Image,
                meta_tensor: torch.Tensor) -> np.ndarray:
    """Run TTA inference; return averaged softmax probabilities (7,)."""
    transforms = get_tta_transforms()
    meta = meta_tensor.to(device)
    probs_list = []
    for tfm in transforms:
        img_t = tfm(pil_img).unsqueeze(0).to(device)
        logits = model(img_t, meta)
        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.mean(probs_list, axis=0)[0]   # shape (7,)


def apply_threshold(probs: np.ndarray) -> tuple[int, str, float]:
    """
    Apply melanoma threshold (0.31) on top of argmax.
    Returns (pred_idx, pred_label, confidence).
    """
    if probs[MEL_IDX] > MEL_THRESHOLD:
        pred_idx = MEL_IDX
    else:
        pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, CLASS_NAMES[pred_idx], confidence

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def render_probability_bar(class_name: str, label: str,
                           prob: float, is_pred: bool, is_mel: bool):
    """Render a single probability bar."""
    bar_color = "#e53e3e" if is_mel else ("#3182ce" if is_pred else "#a0aec0")
    bar_width = max(prob * 100, 0.5)
    st.markdown(f"""
    <div style="margin-bottom:6px">
      <div style="display:flex; justify-content:space-between;
                  font-size:13px; margin-bottom:2px;">
        <span style="font-weight:{'700' if is_pred else '400'}">
          {label}
        </span>
        <span style="font-weight:{'700' if is_pred else '400'}">
          {prob:.1%}
        </span>
      </div>
      <div style="background:#e2e8f0; border-radius:4px; height:10px;">
        <div style="background:{bar_color}; width:{bar_width}%;
                    height:10px; border-radius:4px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_shap_plot(shap_vals: np.ndarray, age: float) -> plt.Figure:
    """Horizontal bar chart for SHAP metadata values."""
    #meta_np = meta_tensor.numpy()[0]
    indices = np.argsort(np.abs(shap_vals))[::-1][:10]   # top 10

    features = []
    values   = []
    for i in indices:
        name = METADATA_FEATURE_NAMES[i]
        if name == "age":
            name = f"age ({age:.0f} yrs)"
        features.append(name)
        values.append(float(shap_vals[i]))

    features = features[::-1]
    values   = values[::-1]
    colors   = ["#e53e3e" if v > 0 else "#3182ce" for v in values]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(features, values, color=colors, height=0.6)
    ax.axvline(0, color="#4a5568", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title("Metadata contribution (SHAP)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    for ticklabel, feature in zip(ax.get_yticklabels(), features):
        fname = feature.replace(f" ({age:.0f} yrs)", "")
        is_active = (fname == "age" or fname == f"loc: {location}")
        if is_active:
            ticklabel.set_fontweight("bold")
            ticklabel.set_bbox(dict(facecolor="#fefcbf", edgecolor="none", pad=2))
        ticklabel.set_fontsize(9)
    
    ax.axvline(0, color="#4a5568", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title("Metadata contribution (SHAP)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    return fig

############################################ PROVISIONAL - EXPERIMENTAL
# COMP
def compute_image_vs_metadata_contrib(model, device, img_t, meta_t, background):
    """Ablación marginal: contribución imagen vs metadatos a p(pred_class)."""
    with torch.no_grad():
        # Predicción real
        logits_real = model(img_t, meta_t.to(device))
        probs_real = F.softmax(logits_real, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs_real))

        # Features de imagen fijas
        img_features = model.image_branch(img_t)

        # p(img real, meta background) — promedio sobre background
        bg_tensor = torch.tensor(background, dtype=torch.float32).to(device)
        meta_feats = model.metadata_branch(bg_tensor)
        fused = torch.cat([img_features.expand(len(background), -1), meta_feats], dim=1)
        probs_img_only = F.softmax(model.classifier(fused), dim=1).cpu().numpy()[:, pred_class].mean()

        # p(img background, meta real) — promedio sobre background
        img_bg_feats = model.image_branch(
            torch.zeros_like(img_t).to(device)
        )
        meta_feat_real = model.metadata_branch(meta_t.to(device))
        fused2 = torch.cat([img_bg_feats.expand(1, -1), meta_feat_real], dim=1)
        probs_meta_only = F.softmax(model.classifier(fused2), dim=1).cpu().numpy()[0, pred_class]

    contrib_img  = abs(float(probs_real[pred_class]) - float(probs_meta_only))
    contrib_meta = abs(float(probs_real[pred_class]) - float(probs_img_only))
    return contrib_img, contrib_meta, pred_class

def render_contrib_plot(contrib_img, contrib_meta):
    """Bar chart imagen vs metadatos."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(
        ["Metadata", "Image"],
        [contrib_meta, contrib_img],
        color=["#e53e3e", "#3182ce"],
        height=0.5,
    )
    ax.set_xlabel("Contribution to p(predicted class)", fontsize=10)
    ax.set_title("Image vs Metadata influence", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, [contrib_meta, contrib_img]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(contrib_img, contrib_meta) * 1.3)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
  .main-title {
    font-size: 2.1rem; font-weight: 800;
    color: #1a202c; margin-bottom: 0;
  }
  .subtitle {
    font-size: 1rem; color: #718096; margin-top: 4px;
  }
  .result-box {
    border-radius: 12px; padding: 20px 24px;
    margin-bottom: 16px;
  }
  .result-mel {
    background: #fff5f5; border: 2px solid #feb2b2;
  }
  .result-ok {
    background: #f0fff4; border: 2px solid #9ae6b4;
  }
  .warning-box {
    background: #fffbeb; border: 2px solid #f6e05e;
    border-radius: 10px; padding: 14px 18px;
    font-size: 14px; color: #744210; margin-top: 10px;
  }
  .flag-box {
    background: #fff5f5; border: 2px solid #fc8181;
    border-radius: 10px; padding: 14px 18px;
    font-size: 14px; color: #742a2a; margin-top: 10px;
  }
  .section-title {
    font-size: 1rem; font-weight: 700;
    color: #2d3748; margin-bottom: 8px; margin-top: 16px;
  }
  .disclaimer {
    font-size: 11px; color: #a0aec0;
    margin-top: 24px; text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🔬 Skin Lesion Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multimodal deep learning system · EfficientNet-B0 + clinical metadata · '
    'HAM10000 · Explainable AI (Grad-CAM + SHAP)</p>',
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Patient data")

    uploaded_file = st.file_uploader(
        "Dermoscopic image",
        type=["jpg", "jpeg", "png"],
        help="Upload a dermoscopic image of the skin lesion.",
    )

    age = st.slider(
        "Patient age",
        min_value=1, max_value=90, value=45, step=1,
    )

    location = st.selectbox(
        "Anatomical location",
        options=LOCATIONS,
        index=LOCATIONS.index("back"),
    )

    run_shap = st.checkbox(
        "Compute SHAP (slower, ~30 s)",
        value=False,
        help="SHAP KernelExplainer on the metadata branch. "
             "Disable for faster inference.",
    )

    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    st.divider()
    st.caption(
        "**Model:** E09 — Color Constancy · age + location · seed 42  \n"
        "**Inference:** TTA ×6 geometric transforms  \n"
        "**Melanoma threshold:** 0.31 (val-selected)  \n"
        "**Melanoma Recall:** 0.9102 · **ROC-AUC:** 0.9727"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

if not uploaded_file:
    st.info("👈 Upload a dermoscopic image and fill in the patient data to begin.")
    st.stop()

# Load image
# Load image — resize a 224×224 antes de color constancy
# Grad-CAM & SmoothGrad always compatibles
pil_img  = Image.open(uploaded_file).convert("RGB")
pil_224  = pil_img.resize((224, 224), Image.LANCZOS)
img_cc   = color_constancy(np.array(pil_224))
pil_cc   = Image.fromarray(img_cc)
meta_t   = build_metadata_vector(age, location)

if analyze_btn or "last_result" in st.session_state:

    if analyze_btn:
        # Load model
        model, device = load_model()

        with st.spinner("Running TTA inference (6 transforms)…"):
            probs = predict_tta(model, device, pil_cc, meta_t)

        pred_idx, pred_name, confidence = apply_threshold(probs)

        img_t = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(pil_cc).unsqueeze(0).to(device)

        with st.spinner("Computing Grad-CAM..."):
            heatmap_gc = run_gradcam(model, img_t, meta_t.to(device), pred_idx)
            overlay    = overlay_heatmap(img_cc, heatmap_gc)

        with st.spinner("Computing SmoothGrad (25 samples)..."):
            heatmap_sg = run_smoothgrad(model, img_t, meta_t.to(device), pred_idx)
            import cv2
            saliency = overlay_heatmap(img_cc, heatmap_sg, colormap=cv2.COLORMAP_HOT)

        contrib_img  = None
        contrib_meta = None
        shap_vals = None
        if run_shap:
            with st.spinner("Computing SHAP values (~30 s)..."):
                # Background uniforme: una muestra por localización con edad media
                # Garantiza contraste SHAP para cualquier localización de entrada
                background = np.zeros((15, METADATA_DIM), dtype=np.float32)
                background[:, 0] = 0.45  # edad media ~40 años normalizada por 90
                for i in range(15):
                    background[i, 1 + i] = 1.0  # una muestra por cada localización
                shap_vals = run_shap_metadata(
                    model, img_t, meta_t.to(device),
                    background_meta=background,
                    target_class=pred_idx, n_background=50
                )
                # PROB
                contrib_img, contrib_meta, _ = compute_image_vs_metadata_contrib(
                    model, device, img_t, meta_t, background
                )

        st.session_state["last_result"] = {
            "probs": probs, "pred_idx": pred_idx,
            "pred_name": pred_name, "confidence": confidence,
            "overlay": overlay, "saliency": saliency, "shap_vals": shap_vals,
            # PROB
            "contrib_img": contrib_img, "contrib_meta": contrib_meta,
        }

    # Retrieve cached result
    res        = st.session_state["last_result"]
    probs      = res["probs"]
    pred_idx   = res["pred_idx"]
    pred_name  = res["pred_name"]
    confidence = res["confidence"]
    overlay    = res["overlay"]
    saliency   = res.get("saliency", None)
    shap_vals  = res["shap_vals"]
    contrib_img  = res.get("contrib_img", None)
    contrib_meta = res.get("contrib_meta", None)

    # ── LAYOUT ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT — images
    with col1:
        st.markdown('<p class="section-title">Input image</p>', unsafe_allow_html=True)
        # Row 1 — original ancho completo
        st.image(pil_img, caption="Original", use_container_width=True)
        # Row 2 — color constancy + Grad-CAM + SmoothGrad
        cc_col, cam_col, sal_col = st.columns(3)
        with cc_col:
            st.image(pil_cc, caption="Color Constancy", use_container_width=True)
        with cam_col:
            st.image(overlay, caption="Grad-CAM", use_container_width=True)
        with sal_col:
            if saliency is not None:
                st.image(saliency, caption="SmoothGrad", use_container_width=True)

        if shap_vals is not None:
            st.markdown('<p class="section-title">Metadata influence (SHAP)</p>',
                        unsafe_allow_html=True)
            fig = render_shap_plot(shap_vals, age)
            st.pyplot(fig)
            plt.close(fig)
            # PROB
        if contrib_img is not None:
            fig2 = render_contrib_plot(contrib_img, contrib_meta)
            st.pyplot(fig2)
            plt.close(fig2)

            # Contextual note based on image/metadata ratio
            ratio = contrib_img / (contrib_meta + 1e-8)
            if contrib_meta < 0.01:
                st.info(
                    "**Image-driven prediction.** "
                    "Clinical metadata did not modify the result. "
                    "The lesion presents sufficiently distinctive visual characteristics."
                )
            elif confidence >= UNCERTAINTY_THR:
                st.info(
                    f"**Clinical metadata contributed to this prediction** "
                    f"(image/metadata ratio: {ratio:.0f}×). "
                    f"Age and/or anatomical location influenced the result alongside the image."
                )
            else:
                st.warning(
                    "⚠️ **Ambiguous image with clinical metadata influence.** "
                    "The model is relying on age and location to reach a decision. "
                    "This case requires review by a dermatology specialist."
                )

    # RIGHT — results
    with col2:
        is_mel  = pred_idx == MEL_IDX
        box_cls = "result-mel" if is_mel else "result-ok"
        icon    = "🔴" if is_mel else "🟢"

        st.markdown(
            f'<div class="result-box {box_cls}">'
            f'<div style="font-size:1.5rem; font-weight:800;">'
            f'{icon} {CLASS_LABELS[pred_name]}</div>'
            f'<div style="font-size:0.9rem; color:#4a5568; margin-top:4px;">'
            f'Confidence: <b>{confidence:.1%}</b> &nbsp;|&nbsp; '
            f'Melanoma probability: <b>{probs[MEL_IDX]:.1%}</b> '
            f'(threshold: {MEL_THRESHOLD})'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Uncertainty warning
        if confidence < UNCERTAINTY_THR:
            st.markdown(
                f'<div class="warning-box">⚠️ <b>High uncertainty</b> — '
                f'model confidence ({confidence:.1%}) is below {UNCERTAINTY_THR:.0%}. '
                f'Please refer to a dermatology specialist.</div>',
                unsafe_allow_html=True,
            )

        # High-risk underrepresented location flag (shortcut-learning mitigation)
        if location in HIGH_RISK_UNDERREPRESENTED:
            prev = HIGH_RISK_UNDERREPRESENTED[location]
            st.markdown(
                f'<div class="flag-box">🚩 <b>High-risk location notice</b> — '
                f'<i>{location}</i> has a real-world melanoma prevalence of '
                f'<b>{prev}%</b> but is underrepresented in the training set '
                f'(n&lt;100). The model may underweight this location. '
                f'<b>Consider specialist referral regardless of prediction.</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Probability bars
        st.markdown('<p class="section-title">Class probabilities</p>',
                    unsafe_allow_html=True)
        sorted_idx = np.argsort(probs)[::-1]
        for i in sorted_idx:
            render_probability_bar(
                CLASS_NAMES[i],
                CLASS_LABELS[CLASS_NAMES[i]],
                float(probs[i]),
                i == pred_idx,
                i == MEL_IDX,
            )

        # Clinical info
        st.markdown('<p class="section-title">Patient input summary</p>',
                    unsafe_allow_html=True)
        st.markdown(f"- **Age:** {age} years  \n"
                    f"- **Anatomical location:** {location}")

    st.markdown(
        '<p class="disclaimer">'
        'This tool is intended for research and educational purposes only. '
        'It does not constitute a medical diagnosis. '
        'Always consult a qualified dermatologist for clinical decisions.'
        '</p>',
        unsafe_allow_html=True,
    )