import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_NAMES, PLANT_PARTS, model_dir, results_dir
from data import load_datasets
from models import build_model


def plot_history(history, model_name, part):
    output_dir = results_dir(part)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{MODEL_NAMES[model_name]} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_accuracy.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{MODEL_NAMES[model_name]} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_loss.png", dpi=160)
    plt.close()


def save_confusion_matrix(model, test_ds, class_names, model_name, part):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{MODEL_NAMES[model_name]} Confusion Matrix")
    fig.tight_layout()
    output_dir = results_dir(part)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{model_name}_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def train_one(model_name, epochs, part):
    train_ds, val_ds, test_ds, class_names = load_datasets(part)

    print(f"\nTraining {MODEL_NAMES[model_name]} for {part}")
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(model_name, len(class_names))

    model_path = model_dir(part) / f"{model_name}_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Saved: {model_path}")

    plot_history(history, model_name, part)
    save_confusion_matrix(best_model, test_ds, class_names, model_name, part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mobilenetv2", "resnet50", "efficientnetb0", "all"],
        default="mobilenetv2",
    )
    parser.add_argument("--part", choices=PLANT_PARTS, default="leaf")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.model == "all":
        for name in ["mobilenetv2", "resnet50", "efficientnetb0"]:
            train_one(name, args.epochs, args.part)
    else:
        train_one(args.model, args.epochs, args.part)


if __name__ == "__main__":
    main()
