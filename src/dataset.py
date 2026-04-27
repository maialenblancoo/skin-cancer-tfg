import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from src.config import IMAGE_DIRS, CLASSES, DATA_PROC
from src.preprocessing import preprocess_image


def find_image_path(image_id: str, preprocess: str = "none") -> str:
    """
    Busca la imagen preprocesada en data/processed/{mode}/ si existe.
    Si no, busca en data/raw/ y aplica preprocesado on-the-fly.
    """
    # Primero busca en processed/
    processed_path = os.path.join(DATA_PROC, preprocess, image_id + ".jpg")
    if os.path.exists(processed_path):
        return processed_path, True   # True = ya preprocesada

    # Si no existe, busca en raw/
    for folder in IMAGE_DIRS:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path, False   # False = hay que preprocesar on-the-fly

    raise FileNotFoundError(f"No se encontró la imagen: {image_id}")


class SkinLesionDataset(Dataset):
    """
    Dataset de lesiones cutáneas HAM10000.

    Parámetros:
        df          : DataFrame con las columnas del split (image_id, dx, age, sex, localization)
        preprocess  : "none" | "dullrazor" | "colorconstancy" | "both"
        metadata    : lista de columnas a usar como metadatos, ej: ["age", "sex", "localization"]
                      lista vacía = solo imagen
        transform   : transformaciones de torchvision
    """

    def __init__(self, df: pd.DataFrame, preprocess: str = "none",
                 metadata: list = None, transform=None):
        self.df         = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.metadata   = metadata if metadata is not None else []
        self.transform  = transform

        # Mapeo clase -> índice
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Imagen ────────────────────────────────────────────────────────────
        path, already_processed = find_image_path(row["image_id"], self.preprocess)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Solo preprocesa on-the-fly si no está ya guardada en processed/
        if not already_processed:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_bgr = preprocess_image(image_bgr, self.preprocess)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Transformaciones (resize, augmentations, ToTensor)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        # ── Etiqueta ──────────────────────────────────────────────────────────
        label = self.class_to_idx[row["dx"]]

        # ── Metadatos ─────────────────────────────────────────────────────────
        if len(self.metadata) == 0:
            return image, label

        meta_vector = self._get_metadata(row)
        return image, meta_vector, label

    def _get_metadata(self, row) -> np.ndarray:
        """Construye el vector de metadatos normalizado."""
        vector = []

        if "age" in self.metadata:
            # Normalizar edad a [0, 1] (rango 0-90)
            age = row["age"] if not pd.isna(row["age"]) else 45.0
            vector.append(float(age) / 90.0)

        if "sex" in self.metadata:
            # One-hot: [male, female, unknown]
            sex = str(row["sex"]).lower()
            vector += [
                1.0 if sex == "male"   else 0.0,
                1.0 if sex == "female" else 0.0,
                1.0 if sex not in ("male", "female") else 0.0,
            ]

        if "localization" in self.metadata:
            # One-hot sobre las localizaciones del dataset
            locs = ["abdomen", "acral", "back", "chest", "ear", "face",
                    "foot", "genital", "hand", "lower extremity", "neck",
                    "scalp", "trunk", "unknown", "upper extremity"]
            loc = str(row["localization"]).lower()
            vector += [1.0 if loc == l else 0.0 for l in locs]

        return np.array(vector, dtype=np.float32)

    @property
    def metadata_dim(self) -> int:
        """Devuelve la dimensión del vector de metadatos."""
        dim = 0
        if "age"          in self.metadata: dim += 1
        if "sex"          in self.metadata: dim += 3
        if "localization" in self.metadata: dim += 15
        return dim