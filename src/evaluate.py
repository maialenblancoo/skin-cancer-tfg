import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, recall_score,
    precision_score, average_precision_score,
    brier_score_loss
)
from sklearn.preprocessing import label_binarize
from src.config import (
    BATCH_SIZE, NUM_WORKERS, CLASSES,
    FIGURES_DIR, RESULTS_CSV, OUTPUTS_DIR
)


def evaluate_model(model, test_df, dataset_class, transform_fn,
                   preprocess, metadata, experiment_name, device,
                   results_csv=None):
    """
    Evalúa el modelo sobre el split de test y guarda resultados.

    Métricas globales: Macro F1, Macro Recall, Macro Precision,
                       ROC-AUC, mAP, Brier Score, Melanoma Recall.
    Métricas por clase: Precision, Recall, F1, AUC para cada una
                        de las 7 clases.

    Guarda:
      - Matriz de confusión en outputs/figures/
      - Probabilidades del test en outputs/probs/
      - Métricas en results_csv (por defecto outputs/results.csv)

    Devuelve un diccionario con todas las métricas.
    """

    # CSV donde guardar resultados
    if results_csv is None:
        results_csv = RESULTS_CSV

    # ── Inferencia ────────────────────────────────────────────────────────────
    test_dataset = dataset_class(
        df=test_df,
        preprocess=preprocess,
        metadata=metadata,
        transform=transform_fn("test")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in test_loader:
            if len(metadata) > 0:
                images, meta, labels = batch
                meta = meta.to(device)
            else:
                images, labels = batch
                meta = None

            images  = images.to(device)
            outputs = model(images, meta)
            probs   = torch.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    # ── Binarizar etiquetas ───────────────────────────────────────────────────
    n_classes  = len(CLASSES)
    labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    # ── Métricas globales ─────────────────────────────────────────────────────
    macro_f1        = f1_score(all_labels, all_preds, average="macro")
    macro_recall    = recall_score(all_labels, all_preds, average="macro")
    macro_precision = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )

    try:
        auc = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average="macro"
        )
    except Exception:
        auc = 0.0

    try:
        map_score = average_precision_score(
            labels_bin, all_probs, average="macro"
        )
    except Exception:
        map_score = 0.0

    brier_per_class = [
        brier_score_loss(labels_bin[:, c], all_probs[:, c])
        for c in range(n_classes)
    ]
    brier_score = float(np.mean(brier_per_class))

    mel_idx    = CLASSES.index("mel")
    mel_recall = recall_score(
        all_labels, all_preds, labels=[mel_idx], average="micro"
    )

    # ── Métricas por clase ────────────────────────────────────────────────────
    per_class = {}
    for c, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": round(float(precision_score(
                all_labels, all_preds, labels=[c],
                average="micro", zero_division=0
            )), 4),
            "recall": round(float(recall_score(
                all_labels, all_preds, labels=[c], average="micro"
            )), 4),
            "f1": round(float(f1_score(
                all_labels, all_preds, labels=[c],
                average="micro", zero_division=0
            )), 4),
            "auc": round(float(roc_auc_score(
                labels_bin[:, c], all_probs[:, c]
            )), 4),
        }

    # ── Print resumen ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Resultados: {experiment_name}")
    print(f"{'='*60}")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Macro Recall    : {macro_recall:.4f}")
    print(f"  Macro Precision : {macro_precision:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  mAP             : {map_score:.4f}")
    print(f"  Brier Score     : {brier_score:.4f}  (0=perfecto, 1=peor)")
    print(f"  Melanoma Recall : {mel_recall:.4f}")
    print(f"\n  Métricas por clase:")
    print(f"  {'Clase':<14} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*50}")
    for cls, m in per_class.items():
        print(f"  {cls:<14} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['auc']:>8.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASSES)}")

    # ── Matriz de confusión ───────────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.title(f"Confusion Matrix - {experiment_name}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, f"{experiment_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Matriz de confusión guardada en: {cm_path}")

    # ── Guardar probabilidades ────────────────────────────────────────────────
    probs_dir = os.path.join(OUTPUTS_DIR, "probs")
    os.makedirs(probs_dir, exist_ok=True)
    np.save(os.path.join(probs_dir, f"{experiment_name}_probs.npy"),  all_probs)
    np.save(os.path.join(probs_dir, f"{experiment_name}_labels.npy"), all_labels)
    print(f"  Probabilidades guardadas en:     {probs_dir}/{experiment_name}_probs.npy")

    # ── Guardar en CSV ────────────────────────────────────────────────────────
    metrics = {
        "experiment"      : experiment_name,
        "preprocess"      : preprocess,
        "metadata"        : "+".join(metadata) if metadata else "none",
        "macro_f1"        : round(macro_f1, 4),
        "macro_recall"    : round(macro_recall, 4),
        "macro_precision" : round(macro_precision, 4),
        "roc_auc"         : round(auc, 4),
        "map"             : round(map_score, 4),
        "brier_score"     : round(brier_score, 4),
        "melanoma_recall" : round(mel_recall, 4),
    }

    for cls, m in per_class.items():
        for metric_name, value in m.items():
            metrics[f"{cls}_{metric_name}"] = value

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
    else:
        df_results = pd.DataFrame()

    df_results = pd.concat(
        [df_results, pd.DataFrame([metrics])],
        ignore_index=True
    )
    df_results.to_csv(results_csv, index=False)
    print(f"  Resultados guardados en:         {results_csv}")

    return metrics