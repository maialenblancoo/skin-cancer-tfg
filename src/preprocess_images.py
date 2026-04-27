import os
import cv2
from tqdm import tqdm
import pandas as pd
from src.config import METADATA_CSV, IMAGE_DIRS, DATA_PROC
from src.preprocessing import preprocess_image


def find_image_path(image_id: str) -> str:
    for folder in IMAGE_DIRS:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No se encontró: {image_id}")


def preprocess_and_save(mode: str):
    """
    Preprocesa todas las imágenes del dataset con el modo indicado
    y las guarda en data/processed/{mode}/

    mode: "none" | "dullrazor" | "colorconstancy" | "both"
    """
    output_dir = os.path.join(DATA_PROC, mode)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(METADATA_CSV)
    image_ids = df["image_id"].tolist()

    print(f"\nPreprocesando {len(image_ids)} imágenes con modo: '{mode}'")
    print(f"Guardando en: {output_dir}\n")

    skipped = 0
    for image_id in tqdm(image_ids, desc=mode):
        output_path = os.path.join(output_dir, image_id + ".jpg")

        # Si ya existe, saltar (permite reanudar si se interrumpe)
        if os.path.exists(output_path):
            skipped += 1
            continue

        img = cv2.imread(find_image_path(image_id))
        img = preprocess_image(img, mode)
        cv2.imwrite(output_path, img)

    print(f"\nListo. {len(image_ids) - skipped} imágenes procesadas, {skipped} ya existían.")
    print(f"Guardadas en: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["none", "dullrazor", "colorconstancy", "both"],
        help="Modo de preprocesado"
    )
    args = parser.parse_args()
    preprocess_and_save(args.mode)