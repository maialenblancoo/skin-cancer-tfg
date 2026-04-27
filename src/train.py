import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from src.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    NUM_WORKERS, MODELS_DIR, CLASSES
)


def get_class_weights(df, device):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.
    Esto penaliza más los fallos en clases minoritarias (ej: melanoma).
    """
    counts = df["dx"].value_counts()
    weights = []
    for cls in CLASSES:
        weights.append(1.0 / counts.get(cls, 1))
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(CLASSES)
    return weights.to(device)


def train_model(model, train_df, val_df, dataset_class, transform_fn,
                preprocess, metadata, experiment_name, device, seed=42):
    """
    Loop de entrenamiento completo con mixed precision (torch.amp).

    Parámetros:
        model           : SkinLesionModel
        train_df        : DataFrame del split de entrenamiento
        val_df          : DataFrame del split de validación
        dataset_class   : clase SkinLesionDataset
        transform_fn    : función get_transforms
        preprocess      : "none" | "dullrazor" | "colorconstancy" | "both"
        metadata        : lista de metadatos, ej: ["age", "sex"]
        experiment_name : nombre para guardar el modelo
        device          : torch.device
        seed            : semilla para reproducibilidad (default: 42)
    """
    # La semilla ya está fijada en run.py, pero la fijamos también aquí
    # para garantizar reproducibilidad si train_model se llama directamente
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = GradScaler("cuda", enabled=use_amp)

    # ── Datasets y DataLoaders ────────────────────────────────────────────────
    train_dataset = dataset_class(
        df=train_df,
        preprocess=preprocess,
        metadata=metadata,
        transform=transform_fn("train")
    )
    val_dataset = dataset_class(
        df=val_df,
        preprocess=preprocess,
        metadata=metadata,
        transform=transform_fn("val")
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Loss con pesos de clase ───────────────────────────────────────────────
    class_weights = get_class_weights(train_df, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizador ───────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    best_val_loss       = float("inf")
    patience_counter    = 0
    early_stop_patience = 7

    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, f"{experiment_name}_best.pth")

    print(f"\n{'='*60}")
    print(f"Experimento: {experiment_name}")
    print(f"Device: {device}  |  Mixed precision: {use_amp}  |  Seed: {seed}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        # ── Fase de entrenamiento ─────────────────────────────────────────────
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for batch in train_loader:
            if len(metadata) > 0:
                images, meta, labels = batch
                meta = meta.to(device)
            else:
                images, labels = batch
                meta = None

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast("cuda", enabled=use_amp):
                outputs = model(images, meta)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Fase de validación ────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(metadata) > 0:
                    images, meta, labels = batch
                    meta = meta.to(device)
                else:
                    images, labels = batch
                    meta = None

                images = images.to(device)
                labels = labels.to(device)

                with autocast("cuda", enabled=use_amp):
                    outputs = model(images, meta)
                    loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # ── Guardar mejor modelo ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Mejor modelo guardado (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping en epoch {epoch+1}")
                break

    print(f"\nEntrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")
    print(f"Modelo guardado en: {best_model_path}")

    return best_model_path