"""
Streamlit application for multimodal skin lesion classification.
Model: EfficientNet-B0 + clinical metadata (age + anatomical location)
Dataset: HAM10000 | TFG - Maialen Blanco Ibarra, Universidad de Deusto
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM as PytorchGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import SkinLesionModel

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}
MEL_IDX         = 4
MEL_THRESHOLD   = 0.31
UNCERTAINTY_THR = 0.70
METADATA_DIM    = 16

LOCATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face",
    "foot", "genital", "hand", "lower extremity",
    "neck", "scalp", "trunk", "unknown", "upper extremity",
]

HIGH_RISK_UNDERREPRESENTED = {
    "ear":  31.7,
    "face": 18.0,
    "neck": 16.7,
}

METADATA_FEATURE_NAMES = [
    "age",
    "loc: abdomen", "loc: acral",   "loc: back",    "loc: chest",
    "loc: ear",     "loc: face",    "loc: foot",    "loc: genital",
    "loc: hand",    "loc: lower extremity", "loc: neck",
    "loc: scalp",   "loc: trunk",   "loc: unknown", "loc: upper extremity",
]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model():
    weights_path = hf_hub_download(
        repo_id="maialenblancoo/skin-cancer-pfg",
        filename="model.pth",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SkinLesionModel(metadata_dim=METADATA_DIM)
    state  = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def color_constancy(img_array: np.ndarray, power: float = 6.0) -> np.ndarray:
    """Shades-of-Gray color constancy normalization (power=6.0, as in training)."""
    img  = img_array.astype(np.float32)
    norm = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    scale = norm.mean() / (norm + 1e-6)
    return np.clip(img * scale, 0, 255).astype(np.uint8)


def build_metadata_vector(age: float, location: str) -> torch.Tensor:
    """Build the 16-dim metadata vector (age normalised + location one-hot)."""
    vec = np.zeros(METADATA_DIM, dtype=np.float32)
    vec[0] = age / 90.0
    if location in LOCATIONS:
        vec[1 + LOCATIONS.index(location)] = 1.0
    return torch.tensor(vec).unsqueeze(0)


def get_img_tensor(pil_img: Image.Image, device) -> torch.Tensor:
    """Standard val/test transform to tensor on device."""
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(pil_img).unsqueeze(0).to(device)


def get_tta_transforms():
    """6 geometric TTA transforms used during training."""
    base = [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return [
        T.Compose(base),
        T.Compose([T.RandomHorizontalFlip(p=1.0)] + base),
        T.Compose([T.RandomVerticalFlip(p=1.0)]   + base),
        T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0)] + base),
        T.Compose([T.RandomRotation((90, 90))]    + base),
        T.Compose([T.RandomRotation((270, 270))]  + base),
    ]


@torch.no_grad()
def predict_tta(model, device, pil_img: Image.Image,
                meta_tensor: torch.Tensor) -> np.ndarray:
    """Run TTA inference; return averaged softmax probabilities (7,)."""
    meta = meta_tensor.to(device)
    probs_list = []
    for tfm in get_tta_transforms():
        img_t  = tfm(pil_img).unsqueeze(0).to(device)
        logits = model(img_t, meta)
        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.mean(probs_list, axis=0)[0]


def apply_threshold(probs: np.ndarray):
    """Apply melanoma threshold (0.31). Returns (pred_idx, pred_label, confidence)."""
    pred_idx   = MEL_IDX if probs[MEL_IDX] > MEL_THRESHOLD else int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, CLASS_NAMES[pred_idx], confidence


# ══════════════════════════════════════════════════════════════════════════════
# XAI — IMAGE BRANCH
# ══════════════════════════════════════════════════════════════════════════════

class _ImageBranchWrapper(nn.Module):
    """
    Wraps the full multimodal model as an image-only model for pytorch-grad-cam,
    fixing the metadata tensor (consistent with xai.py).
    """
    def __init__(self, model: SkinLesionModel, meta_tensor: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("fixed_meta", meta_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.fixed_meta.expand(x.shape[0], -1))


def run_gradcam(model, device, pil_img: Image.Image,
                meta_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """
    Grad-CAM using pytorch-grad-cam library with _ImageBranchWrapper,
    consistent with xai.py. Returns heatmap (224, 224) in [0, 1].
    """
    meta    = meta_tensor.to(device)
    wrapped = _ImageBranchWrapper(model, meta)
    target_layer = [model.image_branch.backbone.blocks[-1]]
    img_t   = get_img_tensor(pil_img, device)
    targets = [ClassifierOutputTarget(target_class)]
    with PytorchGradCAM(model=wrapped, target_layers=target_layer) as cam:
        heatmap = cam(input_tensor=img_t, targets=targets)
    return heatmap[0]  # (224, 224)


def run_smoothgrad(model, device, pil_img: Image.Image,
                   meta_tensor: torch.Tensor, target_class: int,
                   n_samples: int = 25, noise_level: float = 0.15) -> np.ndarray:
    """
    SmoothGrad saliency map, consistent with xai.py implementation.
    Accumulates per-channel gradients across noisy samples, then collapses
    channels by max at the end. Returns heatmap (224, 224) in [0, 1].
    """
    img_t = get_img_tensor(pil_img, device)
    meta  = meta_tensor.to(device)
    std   = noise_level * (img_t.max() - img_t.min()).item()

    accumulated = torch.zeros(3, 224, 224)  # always on CPU

    for _ in range(n_samples):
        noise  = torch.randn_like(img_t) * std
        noisy  = (img_t + noise).clone().requires_grad_(True)
        model.zero_grad()
        logits = model(noisy, meta)
        logits[0, target_class].backward()
        grad   = noisy.grad.data.abs()[0]   # (3, 224, 224)
        accumulated += grad.detach().cpu()

    smoothgrad = (accumulated / n_samples)
    smoothgrad, _ = smoothgrad.max(dim=0)    # (224, 224)
    smoothgrad = smoothgrad.numpy()
    smoothgrad = (smoothgrad - smoothgrad.min()) / (smoothgrad.max() - smoothgrad.min() + 1e-8)
    return smoothgrad


def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45, cmap: str = "jet") -> np.ndarray:
    """
    Overlay a heatmap on an RGB image using matplotlib colormap.
    cmap='jet' for Grad-CAM, cmap='hot' for SmoothGrad (consistent with xai.py).
    """
    heatmap_clipped = np.clip(heatmap, 0.0, 1.0)
    colormap        = plt.get_cmap(cmap)
    heatmap_colored = (colormap(heatmap_clipped)[:, :, :3] * 255).astype(np.uint8)
    img_resized     = np.array(Image.fromarray(img_rgb).resize((224, 224)))
    return (alpha * heatmap_colored + (1 - alpha) * img_resized).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# XAI — METADATA BRANCH (SHAP)
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_metadata(model, device, pil_img: Image.Image,
                          meta_tensor: torch.Tensor,
                          target_class: int,
                          n_background: int = 50) -> np.ndarray:
    """
    SHAP KernelExplainer on the metadata branch.
    Image features are fixed; only metadata is perturbed.
    Consistent with run_shap_metadata() in xai.py.
    Returns shap_values array of shape (16,).
    """
    try:
        import shap
    except ImportError:
        return np.zeros(METADATA_DIM)

    img_t = get_img_tensor(pil_img, device)

    with torch.no_grad():
        img_features = model.image_branch(img_t)  # (1, 256)

    rng        = np.random.default_rng(42)
    background = np.zeros((n_background, METADATA_DIM), dtype=np.float32)
    background[:, 0] = rng.uniform(0, 1, n_background)

    def predict_fn(meta_array: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            batch_size = meta_array.shape[0]
            meta_t     = torch.tensor(meta_array, dtype=torch.float32).to(device)
            img_feat   = img_features.expand(batch_size, -1)
            meta_feat  = model.metadata_branch(meta_t)
            fused      = torch.cat([img_feat, meta_feat], dim=1)
            logits     = model.classifier(fused)
            probs      = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(meta_tensor.numpy(), nsamples=200, silent=True)
    return shap_values[target_class][0]


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_probability_bar(label: str, prob: float, is_pred: bool, is_mel: bool):
    bar_color = "#e53e3e" if is_mel else ("#3182ce" if is_pred else "#a0aec0")
    bar_width = max(prob * 100, 0.5)
    st.markdown(f"""
    <div style="margin-bottom:6px">
      <div style="display:flex; justify-content:space-between;
                  font-size:13px; margin-bottom:2px;">
        <span style="font-weight:{'700' if is_pred else '400'}">{label}</span>
        <span style="font-weight:{'700' if is_pred else '400'}">{prob:.1%}</span>
      </div>
      <div style="background:#e2e8f0; border-radius:4px; height:10px;">
        <div style="background:{bar_color}; width:{bar_width}%;
                    height:10px; border-radius:4px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_shap_plot(shap_vals: np.ndarray, age: float) -> plt.Figure:
    """Horizontal bar chart for SHAP metadata values (top 10), consistent with xai.py."""
    indices  = np.argsort(np.abs(shap_vals))[::-1][:10]
    features = []
    values   = []
    for i in indices:
        name = METADATA_FEATURE_NAMES[i]
        if name == "age":
            name = f"age = {age:.0f} yrs"
        features.append(name)
        values.append(float(shap_vals[i]))

    features = features[::-1]
    values   = values[::-1]
    colors   = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(features, values, color=colors, height=0.6, edgecolor="white")
    ax.axvline(0, color="#4a5568", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title("Metadata contribution (SHAP)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
  .main-title    { font-size:2.1rem; font-weight:800; color:#1a202c; margin-bottom:0; }
  .subtitle      { font-size:1rem; color:#718096; margin-top:4px; }
  .result-box    { border-radius:12px; padding:20px 24px; margin-bottom:16px; }
  .result-mel    { background:#fff5f5; border:2px solid #feb2b2; }
  .result-ok     { background:#f0fff4; border:2px solid #9ae6b4; }
  .warning-box   { background:#fffbeb; border:2px solid #f6e05e; border-radius:10px;
                   padding:14px 18px; font-size:14px; color:#744210; margin-top:10px; }
  .flag-box      { background:#fff5f5; border:2px solid #fc8181; border-radius:10px;
                   padding:14px 18px; font-size:14px; color:#742a2a; margin-top:10px; }
  .section-title { font-size:1rem; font-weight:700; color:#2d3748;
                   margin-bottom:8px; margin-top:16px; }
  .disclaimer    { font-size:11px; color:#a0aec0; margin-top:24px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Skin Lesion Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multimodal deep learning · EfficientNet-B0 + clinical metadata · '
    'HAM10000 · Explainable AI (Grad-CAM + SmoothGrad + SHAP)</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Patient data")

    uploaded_file = st.file_uploader(
        "Dermoscopic image",
        type=["jpg", "jpeg", "png"],
        help="Upload a dermoscopic image of the skin lesion.",
    )

    age = st.slider("Patient age", min_value=1, max_value=90, value=45, step=1)

    location = st.selectbox(
        "Anatomical location",
        options=LOCATIONS,
        index=LOCATIONS.index("back"),
    )

    run_shap = st.checkbox(
        "Compute SHAP (slower, ~30 s)",
        value=False,
        help="SHAP KernelExplainer on the metadata branch.",
    )

    analyze_btn = st.button("Analyze", use_container_width=True, type="primary")

    st.divider()
    st.caption(
        "**Model:** E09 — Color Constancy · age + location · seed 42  \n"
        "**Inference:** TTA x6 geometric transforms  \n"
        "**Melanoma threshold:** 0.31 (val-selected)  \n"
        "**Melanoma Recall:** 0.9102 · **ROC-AUC:** 0.9727"
    )

# ── MAIN ─────────────────────────────────────────────────────────────────────

if not uploaded_file:
    st.info("Upload a dermoscopic image and fill in the patient data to begin.")
    st.stop()

pil_img = Image.open(uploaded_file).convert("RGB")
img_cc  = color_constancy(np.array(pil_img))
pil_cc  = Image.fromarray(img_cc)
meta_t  = build_metadata_vector(age, location)

if analyze_btn or "last_result" in st.session_state:

    if analyze_btn:
        model, device = load_model()

        with st.spinner("Running TTA inference (6 transforms)..."):
            probs = predict_tta(model, device, pil_cc, meta_t)

        pred_idx, pred_name, confidence = apply_threshold(probs)

        with st.spinner("Computing Grad-CAM..."):
            heatmap_gc = run_gradcam(model, device, pil_cc, meta_t, pred_idx)
            overlay_gc = overlay_heatmap(img_cc, heatmap_gc, cmap="jet")

        with st.spinner("Computing SmoothGrad (25 samples)..."):
            heatmap_sg = run_smoothgrad(model, device, pil_cc, meta_t, pred_idx)
            overlay_sg = overlay_heatmap(img_cc, heatmap_sg, cmap="hot")

        shap_vals = None
        if run_shap:
            with st.spinner("Computing SHAP values (~30 s)..."):
                shap_vals = compute_shap_metadata(
                    model, device, pil_cc, meta_t, pred_idx
                )

        st.session_state["last_result"] = {
            "probs": probs, "pred_idx": pred_idx,
            "pred_name": pred_name, "confidence": confidence,
            "overlay_gc": overlay_gc, "overlay_sg": overlay_sg,
            "shap_vals": shap_vals,
        }

    res        = st.session_state["last_result"]
    probs      = res["probs"]
    pred_idx   = res["pred_idx"]
    pred_name  = res["pred_name"]
    confidence = res["confidence"]
    overlay_gc = res.get("overlay_gc", None)
    overlay_sg = res.get("overlay_sg", None)
    shap_vals  = res["shap_vals"]

    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT — images + XAI
    with col1:
        st.markdown('<p class="section-title">Visual explanations</p>',
                    unsafe_allow_html=True)
        st.image(pil_img, caption="Original", use_container_width=True)
        cam_col, sg_col = st.columns(2)
        with cam_col:
            if overlay_gc is not None:
                st.image(overlay_gc, caption="Grad-CAM", use_container_width=True)
        with sg_col:
            if overlay_sg is not None:
                st.image(overlay_sg, caption="SmoothGrad", use_container_width=True)

        if shap_vals is not None:
            st.markdown('<p class="section-title">Metadata influence (SHAP)</p>',
                        unsafe_allow_html=True)
            fig = render_shap_plot(shap_vals, age)
            st.pyplot(fig)
            plt.close(fig)

    # RIGHT — results
    with col2:
        is_mel  = pred_idx == MEL_IDX
        box_cls = "result-mel" if is_mel else "result-ok"
        label   = "MELANOMA DETECTED" if is_mel else CLASS_LABELS[pred_name]

        st.markdown(
            f'<div class="result-box {box_cls}">'
            f'<div style="font-size:1.5rem; font-weight:800;">{label}</div>'
            f'<div style="font-size:0.9rem; color:#4a5568; margin-top:4px;">'
            f'Confidence: <b>{confidence:.1%}</b> &nbsp;|&nbsp; '
            f'Melanoma probability: <b>{probs[MEL_IDX]:.1%}</b> '
            f'(threshold: {MEL_THRESHOLD})'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if confidence < UNCERTAINTY_THR:
            st.markdown(
                f'<div class="warning-box"><b>High uncertainty</b> — '
                f'model confidence ({confidence:.1%}) is below {UNCERTAINTY_THR:.0%}. '
                f'Please refer to a dermatology specialist.</div>',
                unsafe_allow_html=True,
            )

        if location in HIGH_RISK_UNDERREPRESENTED:
            prev = HIGH_RISK_UNDERREPRESENTED[location]
            st.markdown(
                f'<div class="flag-box"><b>High-risk location notice</b> — '
                f'<i>{location}</i> has a real-world melanoma prevalence of '
                f'<b>{prev}%</b> but is underrepresented in the training set '
                f'(n&lt;100). The model may underweight this location. '
                f'<b>Consider specialist referral regardless of prediction.</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<p class="section-title">Class probabilities</p>',
                    unsafe_allow_html=True)
        for i in np.argsort(probs)[::-1]:
            render_probability_bar(
                CLASS_LABELS[CLASS_NAMES[i]],
                float(probs[i]),
                i == pred_idx,
                i == MEL_IDX,
            )

        st.markdown('<p class="section-title">Patient input summary</p>',
                    unsafe_allow_html=True)
        st.markdown(f"- **Age:** {age} years  \n- **Anatomical location:** {location}")

    st.markdown(
        '<p class="disclaimer">'
        'This tool is intended for research and educational purposes only. '
        'It does not constitute a medical diagnosis. '
        'Always consult a qualified dermatologist for clinical decisions.'
        '</p>',
        unsafe_allow_html=True,
    )