"""
src/xai.py
==========
Explainability (XAI) module for the multimodal skin lesion classification system.

Implements four visual explanation methods for the image branch and SHAP analysis
for the metadata branch of the final model E09 (Color Constancy + age + localization,
seed=42).

Visual methods (image branch):
    - Grad-CAM       : class activation map via gradients at last conv block
    - Grad-CAM++     : improved Grad-CAM with better multi-instance localization
    - Vanilla Saliency : raw input gradients (high-res, noisy)
    - SmoothGrad     : averaged input gradients over noisy samples (high-res, clean)

Metadata method:
    - SHAP KernelExplainer : contribution of each clinical variable (age + 15
                             localization categories) with image features fixed.

Usage (from notebook 08):
    from src.xai import (
        load_model_for_xai,
        run_gradcam,
        run_gradcam_plus,
        run_vanilla_saliency,
        run_smoothgrad,
        run_shap_metadata,
        select_representative_cases,
        overlay_heatmap,
        plot_visual_explanations,
        plot_shap_metadata,
        plot_combined_explanation,
    )
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shap

from src.config import CLASSES, NUM_CLASSES
from src.model import SkinLesionModel
from src.dataset import SkinLesionDataset, find_image_path
from src.transforms import get_transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_FEATURE_NAMES = [
    "age",
    "loc: abdomen",
    "loc: acral",
    "loc: back",
    "loc: chest",
    "loc: ear",
    "loc: face",
    "loc: foot",
    "loc: genital",
    "loc: hand",
    "loc: lower extremity",
    "loc: neck",
    "loc: scalp",
    "loc: trunk",
    "loc: unknown",
    "loc: upper extremity",
]

# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_model_for_xai(weights_path: str, metadata_dim: int = 16,
                        device: torch.device = None) -> SkinLesionModel:
    """
    Load the final model E09 from a .pth weights file and set it to eval mode.

    Args:
        weights_path : Path to the .pth file (best epoch weights).
        metadata_dim : Dimension of the metadata input vector. Default 16
                       (age=1 + localization=15) for E09.
        device       : torch.device. If None, auto-detected (CUDA if available).

    Returns:
        model        : SkinLesionModel in eval mode on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkinLesionModel(metadata_dim=metadata_dim)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 2. Image preprocessing helpers
# ---------------------------------------------------------------------------

def load_and_preprocess_image(image_id: str, preprocess: str = "colorconstancy",
                               device: torch.device = None) -> tuple:
    """
    Load a single image by image_id, apply the val/test transform pipeline,
    and return both the tensor (for model input) and the original RGB array
    (for overlay visualization).

    Args:
        image_id   : HAM10000 image identifier (e.g. 'ISIC_0024306').
        preprocess : Preprocessing mode. Default 'colorconstancy' (E09).
        device     : torch.device. If None, auto-detected.

    Returns:
        img_tensor : Float tensor of shape (1, 3, 224, 224) on device.
        img_rgb    : np.ndarray of shape (224, 224, 3) uint8, for visualization.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path, already_preprocessed = find_image_path(image_id, preprocess)

    # Load original image for visualization (resize to 224x224, keep RGB)
    pil_img = Image.open(path).convert("RGB").resize((224, 224))
    img_rgb = np.array(pil_img)

    # Apply val/test transform (resize + normalize)
    transform = get_transforms("test")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    return img_tensor, img_rgb


def get_metadata_tensor(row, dataset_instance, device: torch.device = None) -> torch.Tensor:
    """
    Build the metadata tensor for a single row from a DataFrame using the
    dataset's internal _get_metadata method (ensures consistent encoding).

    Args:
        row              : pandas Series — one row from test.csv.
        dataset_instance : SkinLesionDataset instance (used for encoding logic).
        device           : torch.device. If None, auto-detected.

    Returns:
        meta_tensor : Float tensor of shape (1, metadata_dim) on device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_vector = dataset_instance._get_metadata(row)
    meta_tensor = torch.tensor(meta_vector, dtype=torch.float32).unsqueeze(0).to(device)
    return meta_tensor


# ---------------------------------------------------------------------------
# 3. Grad-CAM
# ---------------------------------------------------------------------------

def _get_target_layer(model: SkinLesionModel):
    """
    Return the target convolutional layer for Grad-CAM / Grad-CAM++.
    This is the last block of EfficientNet-B0 before global average pooling.

    The pytorch-grad-cam library requires a list with one element.
    """
    # EfficientNet-B0 last convolutional block via timm backbone
    target_layer = model.image_branch.backbone.blocks[-1]
    return [target_layer]


class _ImageBranchWrapper(nn.Module):
    """
    Wrapper that makes the full SkinLesionModel behave as an image-only model
    for Grad-CAM by fixing the metadata tensor.

    The metadata tensor is registered as a buffer so it moves with the model
    to any device automatically.
    """

    def __init__(self, model: SkinLesionModel, meta_tensor: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("fixed_meta", meta_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.fixed_meta.expand(x.shape[0], -1))


def run_gradcam(model: SkinLesionModel, img_tensor: torch.Tensor,
                meta_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image.

    Args:
        model        : SkinLesionModel in eval mode.
        img_tensor   : Float tensor of shape (1, 3, 224, 224).
        meta_tensor  : Float tensor of shape (1, metadata_dim).
        target_class : Class index to explain. If None, uses the predicted class.

    Returns:
        heatmap : np.ndarray of shape (224, 224), values in [0, 1].
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    wrapped = _ImageBranchWrapper(model, meta_tensor)
    target_layers = _get_target_layer(model)

    if target_class is None:
        with torch.no_grad():
            logits = model(img_tensor, meta_tensor)
            target_class = logits.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=wrapped, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    return grayscale_cam[0]  # shape (224, 224)


def run_gradcam_plus(model: SkinLesionModel, img_tensor: torch.Tensor,
                     meta_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
    """
    Compute Grad-CAM++ heatmap for a single image.

    Args:
        model        : SkinLesionModel in eval mode.
        img_tensor   : Float tensor of shape (1, 3, 224, 224).
        meta_tensor  : Float tensor of shape (1, metadata_dim).
        target_class : Class index to explain. If None, uses the predicted class.

    Returns:
        heatmap : np.ndarray of shape (224, 224), values in [0, 1].
    """
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    wrapped = _ImageBranchWrapper(model, meta_tensor)
    target_layers = _get_target_layer(model)

    if target_class is None:
        with torch.no_grad():
            logits = model(img_tensor, meta_tensor)
            target_class = logits.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(target_class)]

    with GradCAMPlusPlus(model=wrapped, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    return grayscale_cam[0]  # shape (224, 224)


# ---------------------------------------------------------------------------
# 4. Vanilla Saliency & SmoothGrad
# ---------------------------------------------------------------------------

def run_vanilla_saliency(model: SkinLesionModel, img_tensor: torch.Tensor,
                          meta_tensor: torch.Tensor,
                          target_class: int = None) -> np.ndarray:
    """
    Compute Vanilla Saliency map (input gradient) for a single image.

    The saliency is computed as the absolute value of the gradient of the
    target class score with respect to the input image pixels. The three
    colour channels are collapsed by taking the maximum across channels.

    Args:
        model        : SkinLesionModel in eval mode.
        img_tensor   : Float tensor of shape (1, 3, 224, 224).
        meta_tensor  : Float tensor of shape (1, metadata_dim).
        target_class : Class index to explain. If None, uses the predicted class.

    Returns:
        saliency : np.ndarray of shape (224, 224), values in [0, 1].
    """
    img_tensor = img_tensor.clone().requires_grad_(True)

    logits = model(img_tensor, meta_tensor)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # Backpropagate the score of the target class
    model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    # Gradient shape: (1, 3, 224, 224) → take abs → max over channels
    saliency = img_tensor.grad.data.abs()          # (1, 3, 224, 224)
    saliency, _ = saliency[0].max(dim=0)           # (224, 224)
    saliency = saliency.cpu().numpy()

    # Normalise to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    assert saliency.shape == (224, 224), f"Unexpected saliency shape: {saliency.shape}"
    return saliency


def run_smoothgrad(model: SkinLesionModel, img_tensor: torch.Tensor,
                   meta_tensor: torch.Tensor, target_class: int = None,
                   n_samples: int = 50, noise_level: float = 0.15) -> np.ndarray:
    """
    Compute SmoothGrad saliency map for a single image.

    Adds Gaussian noise to the image n_samples times, computes the vanilla
    saliency for each noisy version, and averages the results. This cancels
    out noise artefacts and produces a cleaner, more stable saliency map.

    Args:
        model        : SkinLesionModel in eval mode.
        img_tensor   : Float tensor of shape (1, 3, 224, 224).
        meta_tensor  : Float tensor of shape (1, metadata_dim).
        target_class : Class index to explain. If None, determined from clean image.
        n_samples    : Number of noisy samples to average. Default 50.
        noise_level  : Std of Gaussian noise relative to signal range. Default 0.15.

    Returns:
        smoothgrad : np.ndarray of shape (224, 224), values in [0, 1].
    """
    # Determine target class from clean image
    if target_class is None:
        with torch.no_grad():
            logits = model(img_tensor, meta_tensor)
            target_class = logits.argmax(dim=1).item()

    # Noise std based on the range of the input tensor
    std = noise_level * (img_tensor.max() - img_tensor.min()).item()

    accumulated_grads = torch.zeros(3, 224, 224)  # (3, 224, 224) — always on CPU

    for _ in range(n_samples):
        noise = torch.randn_like(img_tensor) * std
        noisy_img = (img_tensor + noise).clone().requires_grad_(True)

        logits = model(noisy_img, meta_tensor)
        model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        grad = noisy_img.grad.data.abs()[0]          # (3, 224, 224)
        grad_max, _ = grad.max(dim=0)                # (224, 224)
        accumulated_grads += grad_max.detach().cpu()

    smoothgrad = (accumulated_grads / n_samples)
    smoothgrad, _ = smoothgrad.max(dim=0)   # collapse channels → (224, 224)
    smoothgrad = smoothgrad.numpy()

    # Normalise to [0, 1]
    smoothgrad = (smoothgrad - smoothgrad.min()) / (smoothgrad.max() - smoothgrad.min() + 1e-8)
    return smoothgrad


# ---------------------------------------------------------------------------
# 5. SHAP for metadata branch
# ---------------------------------------------------------------------------

def run_shap_metadata(model: SkinLesionModel, img_tensor: torch.Tensor,
                       meta_tensor: torch.Tensor,
                       background_meta: np.ndarray,
                       target_class: int = None,
                       n_background: int = 100) -> np.ndarray:
    """
    Compute SHAP values for the metadata branch using KernelExplainer.

    The image features are fixed (frozen) for this specific case. SHAP
    perturbs only the 16-dimensional metadata vector to measure each
    clinical variable's contribution to the prediction.

    Args:
        model           : SkinLesionModel in eval mode.
        img_tensor      : Float tensor of shape (1, 3, 224, 224) — the case image.
        meta_tensor     : Float tensor of shape (1, 16) — the case metadata.
        background_meta : np.ndarray of shape (N, 16) — background dataset
                          (training set metadata vectors) for SHAP baseline.
        target_class    : Class index to explain. If None, uses predicted class.
        n_background    : Number of background samples to use. Default 100.

    Returns:
        shap_values : np.ndarray of shape (16,) — SHAP value per feature
                      for the target class.
    """
    device = img_tensor.device

    # Fix image features for this case
    with torch.no_grad():
        fixed_img_features = model.image_branch(img_tensor)  # (1, 256)

    if target_class is None:
        with torch.no_grad():
            logits = model(img_tensor, meta_tensor)
            target_class = logits.argmax(dim=1).item()

    # Define prediction function: metadata array → softmax probabilities
    def predict_fn(meta_array: np.ndarray) -> np.ndarray:
        """
        Takes a batch of metadata vectors (np.ndarray shape [B, 16]),
        runs them through the metadata branch and classifier with fixed
        image features, and returns softmax probabilities (shape [B, 7]).
        """
        meta_t = torch.tensor(meta_array, dtype=torch.float32).to(device)
        batch_size = meta_t.shape[0]

        with torch.no_grad():
            # Expand fixed image features to match batch size
            img_feat = fixed_img_features.expand(batch_size, -1)  # (B, 256)
            meta_feat = model.metadata_branch(meta_t)             # (B, 64)
            fused = torch.cat([img_feat, meta_feat], dim=1)       # (B, 320)
            logits = model.classifier(fused)                      # (B, 7)
            probs = F.softmax(logits, dim=1)                      # (B, 7)

        return probs.cpu().numpy()

    # Sample background for SHAP baseline
    if background_meta.shape[0] > n_background:
        idx = np.random.choice(background_meta.shape[0], n_background, replace=False)
        background = background_meta[idx]
    else:
        background = background_meta

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(
        meta_tensor.cpu().numpy(), nsamples=200, silent=True
    )
    # shap_values is a list of 7 arrays (one per class), each shape (1, 16)
    return shap_values[target_class][0]  # shape (16,)


# ---------------------------------------------------------------------------
# 6. Case selection utilities
# ---------------------------------------------------------------------------

def select_representative_cases(probs: np.ndarray, labels: np.ndarray,
                                  test_df, n_per_type: int = 3) -> dict:
    """
    Select representative cases from the test set for XAI analysis.

    Four case types are selected:
        - 'correct_melanoma'   : true melanoma, predicted as melanoma, prob_mel > 0.90
        - 'critical_error'     : true melanoma, predicted as nv (most dangerous failure)
        - 'high_uncertainty'   : max probability across all classes < 0.70
        - 'correct_non_mel'    : true nv or bkl, predicted correctly, high confidence

    Args:
        probs       : np.ndarray of shape (1503, 7) — softmax probabilities.
        labels      : np.ndarray of shape (1503,)  — ground truth class indices.
        test_df     : pandas DataFrame — test.csv with image metadata.
        n_per_type  : Number of cases to select per type. Default 3.

    Returns:
        cases : dict with keys ['correct_melanoma', 'critical_error',
                'high_uncertainty', 'correct_non_mel'], each containing a list
                of dicts with keys ['idx', 'image_id', 'true_class',
                'pred_class', 'confidence', 'row'].
    """
    mel_idx = CLASSES.index("mel")   # 4
    nv_idx  = CLASSES.index("nv")    # 5
    bkl_idx = CLASSES.index("bkl")  # 2

    preds = probs.argmax(axis=1)
    max_conf = probs.max(axis=1)

    def _build_case(i):
        return {
            "idx": i,
            "image_id": test_df.iloc[i]["image_id"],
            "true_class": CLASSES[labels[i]],
            "pred_class": CLASSES[preds[i]],
            "confidence": float(max_conf[i]),
            "prob_mel": float(probs[i, mel_idx]),
            "row": test_df.iloc[i],
        }

    cases = {}

    # --- Correct melanoma (high confidence) ---
    mask = (labels == mel_idx) & (preds == mel_idx) & (probs[:, mel_idx] > 0.90)
    idxs = np.where(mask)[0]
    # Sort by descending melanoma probability
    idxs = idxs[np.argsort(probs[idxs, mel_idx])[::-1]][:n_per_type]
    cases["correct_melanoma"] = [_build_case(i) for i in idxs]

    # --- Critical error: true melanoma predicted as nv ---
    mask = (labels == mel_idx) & (preds == nv_idx)
    idxs = np.where(mask)[0]
    # Sort by descending nv probability (most confidently wrong first)
    idxs = idxs[np.argsort(probs[idxs, nv_idx])[::-1]][:n_per_type]
    cases["critical_error"] = [_build_case(i) for i in idxs]

    # --- High uncertainty ---
    mask = max_conf < 0.70
    idxs = np.where(mask)[0]
    # Sort by ascending max confidence (most uncertain first)
    idxs = idxs[np.argsort(max_conf[idxs])][:n_per_type]
    cases["high_uncertainty"] = [_build_case(i) for i in idxs]

    # --- Correct non-melanoma (nv or bkl, high confidence) ---
    mask = (
        ((labels == nv_idx) | (labels == bkl_idx)) &
        (preds == labels) &
        (max_conf > 0.85)
    )
    idxs = np.where(mask)[0]
    # Sort by descending confidence
    idxs = idxs[np.argsort(max_conf[idxs])[::-1]][:n_per_type]
    cases["correct_non_mel"] = [_build_case(i) for i in idxs]

    # Summary
    for case_type, case_list in cases.items():
        print(f"  {case_type}: {len(case_list)} cases selected")

    return cases


# ---------------------------------------------------------------------------
# 7. Visualization helpers
# ---------------------------------------------------------------------------

def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45, colormap=None) -> np.ndarray:
    import cv2
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    """
    Overlay a heatmap on an RGB image using a colourmap.

    Args:
        img_rgb  : np.ndarray (224, 224, 3) uint8 — original image.
        heatmap  : np.ndarray (224, 224) float in [0, 1] — activation map.
        alpha    : Blend weight for the heatmap overlay. Default 0.45.
        colormap : OpenCV colormap constant. Default COLORMAP_JET.

    Returns:
        overlay : np.ndarray (224, 224, 3) uint8 — blended image.
    """
    import cv2

    heatmap_clipped = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
    heatmap_uint8 = np.uint8(255 * heatmap_clipped)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    h, w          = heatmap_rgb.shape[:2]
    img_resized   = cv2.resize(img_rgb, (w, h))
    overlay       = cv2.addWeighted(img_resized, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def plot_visual_explanations(img_rgb: np.ndarray,
                              gradcam: np.ndarray,
                              gradcam_plus: np.ndarray,
                              vanilla: np.ndarray,
                              smoothgrad: np.ndarray,
                              title: str = "",
                              save_path: str = None) -> plt.Figure:
    """
    Plot a 1×5 figure: original image + 4 visual explanation maps side by side.

    Layout:
        [Original] [Grad-CAM] [Grad-CAM++] [Vanilla Saliency] [SmoothGrad]

    Args:
        img_rgb      : np.ndarray (224, 224, 3) uint8.
        gradcam      : np.ndarray (224, 224) in [0, 1].
        gradcam_plus : np.ndarray (224, 224) in [0, 1].
        vanilla      : np.ndarray (224, 224) in [0, 1].
        smoothgrad   : np.ndarray (224, 224) in [0, 1].
        title        : Figure suptitle string.
        save_path    : If provided, saves the figure to this path (dpi=150).

    Returns:
        fig : matplotlib Figure object.
    """
    import cv2

    gradcam_overlay    = overlay_heatmap(img_rgb, gradcam)
    gradcam_p_overlay  = overlay_heatmap(img_rgb, gradcam_plus)
    vanilla_overlay    = overlay_heatmap(img_rgb, vanilla, colormap=cv2.COLORMAP_HOT)
    smoothgrad_overlay = overlay_heatmap(img_rgb, smoothgrad, colormap=cv2.COLORMAP_HOT)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    data = [
        (img_rgb,          "Original"),
        (gradcam_overlay,  "Grad-CAM"),
        (gradcam_p_overlay,"Grad-CAM++"),
        (vanilla_overlay,  "Vanilla Saliency"),
        (smoothgrad_overlay,"SmoothGrad"),
    ]

    for ax, (img, label) in zip(axes, data):
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig


def plot_shap_metadata(shap_values: np.ndarray,
                        meta_vector: np.ndarray,
                        target_class: str,
                        title: str = "",
                        save_path: str = None) -> plt.Figure:
    """
    Plot a horizontal bar chart of SHAP values for the 16 metadata features.

    Positive values (red) push the prediction towards the target class.
    Negative values (blue) push away from it.

    Args:
        shap_values  : np.ndarray (16,) — SHAP value per feature.
        meta_vector  : np.ndarray (16,) — actual feature values for this case.
        target_class : String name of the class being explained (e.g. 'mel').
        title        : Figure title.
        save_path    : If provided, saves the figure to this path (dpi=150).

    Returns:
        fig : matplotlib Figure object.
    """
    # Build display labels showing feature name and actual value
    labels = []
    for i, name in enumerate(METADATA_FEATURE_NAMES):
        val = meta_vector[i]
        if i == 0:
            # age: denormalise for readability (val * 90)
            labels.append(f"{name} = {val * 90:.0f} yrs")
        else:
            # localization: show 1/0
            labels.append(f"{name} = {int(val)}")

    # Sort by absolute SHAP value descending
    order = np.argsort(np.abs(shap_values))[::-1]
    sorted_shap   = shap_values[order]
    sorted_labels = [labels[i] for i in order]

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sorted_shap]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(sorted_shap)), sorted_shap, color=colors, edgecolor="white")
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title(title or f"SHAP — Metadata contribution\nTarget class: {target_class}",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()  # Most important feature at top

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig


def plot_combined_explanation(img_rgb: np.ndarray,
                               gradcam: np.ndarray,
                               gradcam_plus: np.ndarray,
                               vanilla: np.ndarray,
                               smoothgrad: np.ndarray,
                               shap_values: np.ndarray,
                               meta_vector: np.ndarray,
                               case_info: dict,
                               save_path: str = None) -> plt.Figure:
    """
    Generate a combined figure with visual explanations (top row) and SHAP
    metadata analysis (bottom panel) for a single case.

    Layout:
        Top row    : [Original] [Grad-CAM] [Grad-CAM++] [Vanilla] [SmoothGrad]
        Bottom     : SHAP horizontal bar chart

    Args:
        img_rgb     : np.ndarray (224, 224, 3) uint8.
        gradcam     : np.ndarray (224, 224) in [0, 1].
        gradcam_plus: np.ndarray (224, 224) in [0, 1].
        vanilla     : np.ndarray (224, 224) in [0, 1].
        smoothgrad  : np.ndarray (224, 224) in [0, 1].
        shap_values : np.ndarray (16,).
        meta_vector : np.ndarray (16,).
        case_info   : dict from select_representative_cases (contains image_id,
                      true_class, pred_class, confidence, prob_mel).
        save_path   : If provided, saves the figure to this path (dpi=150).

    Returns:
        fig : matplotlib Figure object.
    """
    import cv2
    
    gradcam_overlay    = overlay_heatmap(img_rgb, gradcam)
    gradcam_p_overlay  = overlay_heatmap(img_rgb, gradcam_plus)
    vanilla_overlay    = overlay_heatmap(img_rgb, vanilla, colormap=cv2.COLORMAP_HOT)
    smoothgrad_overlay = overlay_heatmap(img_rgb, smoothgrad, colormap=cv2.COLORMAP_HOT)

    fig = plt.figure(figsize=(22, 12))

    # --- Top row: 5 image panels ---
    top_images = [
        (img_rgb,           "Original"),
        (gradcam_overlay,   "Grad-CAM"),
        (gradcam_p_overlay, "Grad-CAM++"),
        (vanilla_overlay,   "Vanilla Saliency"),
        (smoothgrad_overlay,"SmoothGrad"),
    ]

    for col, (img, label) in enumerate(top_images):
        ax = fig.add_subplot(2, 5, col + 1)
        ax.imshow(img)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")

    # --- Bottom: SHAP bar chart spanning full width ---
    ax_shap = fig.add_subplot(2, 1, 2)

    order = np.argsort(np.abs(shap_values))[::-1]
    sorted_shap   = shap_values[order]
    sorted_labels = []
    for i in order:
        val = meta_vector[i]
        name = METADATA_FEATURE_NAMES[i]
        if i == 0:
            sorted_labels.append(f"{name} = {val * 90:.0f} yrs")
        else:
            sorted_labels.append(f"{name} = {int(val)}")

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sorted_shap]
    ax_shap.barh(range(len(sorted_shap)), sorted_shap, color=colors, edgecolor="white")
    ax_shap.set_yticks(range(len(sorted_labels)))
    ax_shap.set_yticklabels(sorted_labels, fontsize=9)
    ax_shap.axvline(0, color="black", linewidth=0.8)
    ax_shap.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax_shap.set_title(
        f"Metadata contribution (SHAP) — Target class: {case_info['pred_class']}",
        fontsize=11, fontweight="bold"
    )
    ax_shap.invert_yaxis()

    # --- Suptitle with case info ---
    suptitle = (
        f"Image ID: {case_info['image_id']}   |   "
        f"True: {case_info['true_class']}   |   "
        f"Predicted: {case_info['pred_class']}   |   "
        f"Confidence: {case_info['confidence']:.1%}   |   "
        f"P(mel): {case_info['prob_mel']:.3f}"
    )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 8. Background dataset builder (for SHAP)
# ---------------------------------------------------------------------------

def build_background_metadata(train_df, dataset_instance) -> np.ndarray:
    """
    Build the background metadata matrix for SHAP KernelExplainer by
    extracting metadata vectors from all training samples.

    Args:
        train_df         : pandas DataFrame — train.csv.
        dataset_instance : SkinLesionDataset instance (for _get_metadata).

    Returns:
        background : np.ndarray of shape (N_train, 16).
    """
    vectors = []
    for _, row in train_df.iterrows():
        vectors.append(dataset_instance._get_metadata(row))
    return np.array(vectors, dtype=np.float32)