from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
PLANT_PARTS = ("leaf", "stem_bark", "root", "flower", "fruit")

MODEL_NAMES = {
    "mobilenetv2": "MobileNetV2",
    "resnet50": "ResNet50",
    "efficientnetb0": "EfficientNetB0",
}


def data_dir(part):
    """Return the dataset root for a plant part."""
    if part not in PLANT_PARTS:
        raise ValueError(
            f"Unknown plant part '{part}'. Choose from: {', '.join(PLANT_PARTS)}"
        )
    return DATA_DIR / part


def model_dir(part):
    """Return the model directory for a plant part."""
    if part not in PLANT_PARTS:
        raise ValueError(
            f"Unknown plant part '{part}'. Choose from: {', '.join(PLANT_PARTS)}"
        )
    return MODEL_DIR if part == "leaf" else MODEL_DIR / part


def results_dir(part):
    """Return the results directory for a plant part."""
    if part not in PLANT_PARTS:
        raise ValueError(
            f"Unknown plant part '{part}'. Choose from: {', '.join(PLANT_PARTS)}"
        )
    return RESULTS_DIR if part == "leaf" else RESULTS_DIR / part
