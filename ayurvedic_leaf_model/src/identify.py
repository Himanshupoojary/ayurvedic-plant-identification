import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PLANT_PARTS, model_dir
from fusion import MODELS, fuse, load_image, load_models, validation_weights


def predict_part(image_path, part):
    part_model_dir = model_dir(part)
    class_file = part_model_dir / "class_names.json"
    if not class_file.exists():
        raise FileNotFoundError(
            f"No trained {part} model found: {class_file}. "
            f"Train this part first or omit its image."
        )

    with open(class_file, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    models = load_models(part)
    weights = validation_weights(models, class_names, part)
    image = load_image(image_path)
    probability_vectors = {
        model_name: models[model_name].predict(image, verbose=0)[0]
        for model_name in MODELS
    }
    return class_names, fuse(probability_vectors, weights)


def main():
    parser = argparse.ArgumentParser(
        description="Identify a plant from any available plant-part images."
    )
    for part in PLANT_PARTS:
        parser.add_argument(f"--{part}", type=Path, help=f"Optional {part} image")
    args = parser.parse_args()

    images = {
        part: getattr(args, part)
        for part in PLANT_PARTS
        if getattr(args, part) is not None
    }
    if not images:
        parser.error("Provide at least one image, such as --leaf leaf.jpg.")

    part_predictions = []
    for part, image_path in images.items():
        if not image_path.exists():
            raise FileNotFoundError(f"{part} image not found: {image_path}")
        class_names, probabilities = predict_part(image_path, part)
        part_predictions.append((part, class_names, probabilities))

    reference_classes = part_predictions[0][1]
    for part, class_names, _ in part_predictions:
        if class_names != reference_classes:
            raise ValueError(
                f"Class names for {part} do not match the other trained parts."
            )

    final_probabilities = np.mean(
        np.stack([probabilities for _, _, probabilities in part_predictions]),
        axis=0,
    )
    top_indices = np.argsort(final_probabilities)[::-1][:5]

    print(
        f"\nUsing {len(part_predictions)} image(s): "
        f"{', '.join(part for part, _, _ in part_predictions)}"
    )
    print("\nFINAL PLANT PREDICTION")
    print("=" * 50)
    for index in top_indices:
        print(
            f"{reference_classes[index]:20s}: "
            f"{final_probabilities[index] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
