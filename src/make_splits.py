import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    METADATA_CSV, SPLITS_DIR,
    SPLIT_SEED, TEST_SIZE, VAL_SIZE
)


def make_splits():
    print("Leyendo metadata...")
    df = pd.read_csv(METADATA_CSV)

    print(f"Total imágenes: {len(df)}")
    print(f"Distribución de clases:\n{df['dx'].value_counts()}\n")

    # ── Train / Test ──────────────────────────────────────────────────────────
    # Stratify para mantener la distribución de clases en todos los splits
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=df["dx"]
    )

    # ── Train / Val ───────────────────────────────────────────────────────────
    # VAL_SIZE es sobre el total, ajustamos para que sea correcto
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=SPLIT_SEED,
        stratify=train_val["dx"]
    )

    # ── Guardar ───────────────────────────────────────────────────────────────
    os.makedirs(SPLITS_DIR, exist_ok=True)

    train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(SPLITS_DIR, "val.csv"),   index=False)
    test.to_csv(os.path.join(SPLITS_DIR, "test.csv"),  index=False)

    print(f"Splits guardados en {SPLITS_DIR}")
    print(f"  Train : {len(train):5d} imágenes")
    print(f"  Val   : {len(val):5d} imágenes")
    print(f"  Test  : {len(test):5d} imágenes")
    print(f"\nDistribución train:\n{train['dx'].value_counts()}")
    print(f"\nDistribución val:\n{val['dx'].value_counts()}")
    print(f"\nDistribución test:\n{test['dx'].value_counts()}")


if __name__ == "__main__":
    make_splits()