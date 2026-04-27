import argparse
import random
import numpy as np
import torch
import pandas as pd

from src.config import SPLITS_DIR, MODELS_DIR, RESULTS_CSV
from src.dataset import SkinLesionDataset
from src.transforms import get_transforms
from src.model import SkinLesionModel
from src.train import train_model
from src.evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Lanzar un experimento")

    parser.add_argument(
        "--preprocess",
        type=str,
        default="none",
        choices=["none", "dullrazor", "colorconstancy", "both"],
        help="Tipo de preprocesado"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        default=[],
        help="Metadatos a usar: age sex localization (separados por espacio)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Nombre del experimento (opcional, se genera automáticamente)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)"
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="Ruta al CSV de resultados (por defecto usa RESULTS_CSV de config)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Reproducibilidad ──────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ── Nombre automático del experimento ─────────────────────────────────────
    if args.experiment_name is None:
        meta_str = "+".join(args.metadata) if args.metadata else "none"
        args.experiment_name = f"prep_{args.preprocess}__meta_{meta_str}__seed_{args.seed}"

    # ── CSV de resultados ─────────────────────────────────────────────────────
    results_csv = args.results_csv if args.results_csv is not None else RESULTS_CSV

    print(f"\nExperimento  : {args.experiment_name}")
    print(f"Preprocesado : {args.preprocess}")
    print(f"Metadatos    : {args.metadata if args.metadata else 'ninguno'}")
    print(f"Seed         : {args.seed}")
    print(f"Results CSV  : {results_csv}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")

    # ── Cargar splits ─────────────────────────────────────────────────────────
    train_df = pd.read_csv(f"{SPLITS_DIR}/train.csv")
    val_df   = pd.read_csv(f"{SPLITS_DIR}/val.csv")
    test_df  = pd.read_csv(f"{SPLITS_DIR}/test.csv")

    # ── Modelo ────────────────────────────────────────────────────────────────
    dummy_dataset = SkinLesionDataset(
        df=train_df.head(1),
        preprocess="none",
        metadata=args.metadata
    )
    metadata_dim = dummy_dataset.metadata_dim
    model = SkinLesionModel(metadata_dim=metadata_dim).to(device)

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    best_model_path = train_model(
        model           = model,
        train_df        = train_df,
        val_df          = val_df,
        dataset_class   = SkinLesionDataset,
        transform_fn    = get_transforms,
        preprocess      = args.preprocess,
        metadata        = args.metadata,
        experiment_name = args.experiment_name,
        device          = device,
        seed            = args.seed,
    )

    # ── Evaluación ────────────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )

    metrics = evaluate_model(
        model           = model,
        test_df         = test_df,
        dataset_class   = SkinLesionDataset,
        transform_fn    = get_transforms,
        preprocess      = args.preprocess,
        metadata        = args.metadata,
        experiment_name = args.experiment_name,
        device          = device,
        results_csv     = results_csv,
    )

    print(f"\nExperimento finalizado: {args.experiment_name}")
    print(f"Métricas finales: {metrics}")


if __name__ == "__main__":
    main()