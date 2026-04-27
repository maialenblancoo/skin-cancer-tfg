import os

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC   = os.path.join(BASE_DIR, "data", "processed")
SPLITS_DIR  = os.path.join(BASE_DIR, "data", "splits")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(OUTPUTS_DIR, "models")
LOGS_DIR    = os.path.join(OUTPUTS_DIR, "logs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "results.csv")

# ── Dataset ──────────────────────────────────────────────────────────────────
METADATA_CSV = os.path.join(DATA_RAW, "HAM10000_metadata.csv")
IMAGE_DIRS   = [
    os.path.join(DATA_RAW, "HAM10000_images_part_1"),
    os.path.join(DATA_RAW, "HAM10000_images_part_2"),
]

# Clases del dataset (orden fijo siempre)
CLASSES     = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)

# ── Splits ───────────────────────────────────────────────────────────────────
SPLIT_SEED = 42
TEST_SIZE  = 0.15
VAL_SIZE   = 0.15

# ── Preprocesado ─────────────────────────────────────────────────────────────
IMAGE_SIZE = 224  # píxeles (224x224)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
BACKBONE      = "efficientnet_b0"
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
NUM_WORKERS   = 4
SEED          = 42

# ── Experimento (se sobreescribe desde run.py) ────────────────────────────────
# preprocess : "none" | "dullrazor" | "colorconstancy" | "both"
# metadata   : [] | ["age"] | ["sex"] | ["localization"] | ["age","sex","localization"] | ...
DEFAULT_EXPERIMENT = {
    "preprocess": "none",
    "metadata":   [],
}