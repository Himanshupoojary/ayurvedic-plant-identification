import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MODEL_NAMES,
    PLANT_PARTS,
    SEED,
    data_dir,
    model_dir,
)

MODELS = ["mobilenetv2", "resnet50", "efficientnetb0"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_models(part):
    models = {}
    part_model_dir = model_dir(part)

    for model_name in MODELS:
        path = part_model_dir / f"{model_name}_best.keras"

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        print(f"Loading {model_name}...")
        models[model_name] = tf.keras.models.load_model(path)

    return models


def load_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


def validation_weights(models, class_names, part):
    """Calculate model weights from validation accuracy, without using test data."""
    validation_dir = data_dir(part) / "val"
    if not validation_dir.exists():
        raise FileNotFoundError(
            f"Validation data not found: {validation_dir}. "
            "Prepare the dataset before using accuracy-weighted fusion."
        )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        seed=SEED,
    )
    if validation_ds.class_names != class_names:
        raise ValueError("Validation class folders do not match the model class names.")

    true_labels = np.concatenate(
        [labels.numpy() for _, labels in validation_ds],
        axis=0,
    )
    accuracies = {}
    for model_name in MODELS:
        probabilities = models[model_name].predict(validation_ds, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        accuracies[model_name] = float(accuracy_score(true_labels, predictions))

    total_accuracy = sum(accuracies.values())
    if total_accuracy <= 0:
        weights = {model_name: 1 / len(MODELS) for model_name in MODELS}
    else:
        weights = {
            model_name: accuracy / total_accuracy
            for model_name, accuracy in accuracies.items()
        }

    print("\nVALIDATION ACCURACY WEIGHTS")
    print("=" * 50)
    for model_name in MODELS:
        print(
            f"{MODEL_NAMES[model_name]:20s}: "
            f"accuracy={accuracies[model_name] * 100:.2f}% "
            f"weight={weights[model_name]:.4f}"
        )
    return weights


def fuse(probability_vectors, weights):
    return sum(
        weights[model_name] * probabilities
        for model_name, probabilities in probability_vectors.items()
    )


def predict_single(image_path, models, class_names, weights):

    image = load_image(image_path)

    probability_vectors = {}

    print("\nINDIVIDUAL MODEL PREDICTIONS")
    print("=" * 50)

    for model_name in MODELS:

        probabilities = models[model_name].predict(
            image,
            verbose=0
        )[0]

        probability_vectors[model_name] = probabilities

        index = int(np.argmax(probabilities))

        print(
            f"{model_name:20s}: "
            f"{class_names[index]} "
            f"({probabilities[index] * 100:.2f}%)"
        )

    fused = fuse(probability_vectors, weights)

    top_indices = np.argsort(fused)[::-1][:5]

    print("\nFUSED PREDICTION")
    print("=" * 50)

    for index in top_indices:
        print(
            f"{class_names[index]:20s}: "
            f"{fused[index] * 100:.2f}%"
        )


def get_test_images(class_names, part):
    samples = []

    test_dir = data_dir(part) / "test"
    for class_index, class_name in enumerate(class_names):

        class_dir = test_dir / class_name

        if not class_dir.exists():
            continue

        for image_path in class_dir.rglob("*"):

            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(
                    (image_path, class_index)
                )

    return samples


def evaluate_test_set(models, class_names, weights, part, report=False):

    samples = get_test_images(class_names, part)

    if not samples:
        raise ValueError("No test images found.")

    print(f"\nTotal test images: {len(samples)}")
    print("Running predictions...\n")

    true_labels = []

    predictions = {
        model_name: []
        for model_name in MODELS
    }

    fusion_predictions = []

    for image_path, true_label in samples:

        image = load_image(image_path)

        probability_vectors = {}

        for model_name in MODELS:

            probabilities = models[model_name].predict(
                image,
                verbose=0
            )[0]

            probability_vectors[model_name] = probabilities

            predicted_class = int(
                np.argmax(probabilities)
            )

            predictions[model_name].append(
                predicted_class
            )

        fused_probabilities = fuse(probability_vectors, weights)

        fused_class = int(
            np.argmax(fused_probabilities)
        )

        true_labels.append(true_label)
        fusion_predictions.append(fused_class)

    print("=" * 55)
    print(f"{part.upper()} MODEL COMPARISON")
    print("=" * 55)

    individual_accuracies = {}

    for model_name in MODELS:

        accuracy = accuracy_score(
            true_labels,
            predictions[model_name]
        )

        individual_accuracies[model_name] = accuracy

        print(
            f"{model_name:20s}: "
            f"{accuracy * 100:.2f}%"
        )

    fusion_accuracy = accuracy_score(
        true_labels,
        fusion_predictions
    )

    print("-" * 55)

    print(
        f"{'FUSION':20s}: "
        f"{fusion_accuracy * 100:.2f}%"
    )

    print("=" * 55)

    best_model = max(
        individual_accuracies,
        key=individual_accuracies.get
    )

    best_accuracy = individual_accuracies[best_model]

    difference = fusion_accuracy - best_accuracy

    print(
        f"\nBest individual model: "
        f"{best_model} "
        f"({best_accuracy * 100:.2f}%)"
    )

    print(
        f"Fusion improvement: "
        f"{difference * 100:+.2f} percentage points"
    )

    if report:

        print("\n\nFUSION CLASSIFICATION REPORT")
        print("=" * 55)

        print(
            classification_report(
                true_labels,
                fusion_predictions,
                target_names=class_names,
                zero_division=0
            )
        )

        print("FUSION CONFUSION MATRIX")
        print("=" * 55)

        print(
            confusion_matrix(
                true_labels,
                fusion_predictions
            )
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        help="Path to a single leaf image to identify"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed test-set report"
    )

    parser.add_argument("--part", choices=PLANT_PARTS, default="leaf")
    parser.add_argument(
        "--equal-weight",
        action="store_true",
        help="Use equal model weights instead of validation-accuracy weights",
    )
    args = parser.parse_args()

    part_model_dir = model_dir(args.part)
    class_file = part_model_dir / "class_names.json"

    if not class_file.exists():
        raise FileNotFoundError(
            f"Class names not found: {class_file}"
        )

    with open(
        class_file,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = json.load(f)

    models = load_models(args.part)
    if args.equal_weight:
        weights = {model_name: 1 / len(MODELS) for model_name in MODELS}
        print("\nUsing equal model weights.")
    else:
        weights = validation_weights(models, class_names, args.part)

    # Single-image prediction
    if args.image:

        image_path = Path(args.image)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        predict_single(
            image_path,
            models,
            class_names,
            weights,
        )

    # Full test-set evaluation
    else:

        evaluate_test_set(
            models,
            class_names,
            weights,
            args.part,
            args.report
        )


if __name__ == "__main__":
    main()