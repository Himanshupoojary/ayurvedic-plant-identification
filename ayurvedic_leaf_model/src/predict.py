import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import IMAGE_SIZE, PLANT_PARTS, model_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--part", choices=PLANT_PARTS, default="leaf")
    args = parser.parse_args()

    part_model_dir = model_dir(args.part)
    model_path = part_model_dir / f"{args.model}_best.keras"
    class_path = part_model_dir / "class_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not class_path.exists():
        raise FileNotFoundError(f"Class names not found: {class_path}")

    with open(class_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(model_path)

    image = tf.keras.utils.load_img(args.image, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    indices = np.argsort(probabilities)[::-1][:args.top]

    print("\nPrediction")
    print("----------")
    for i in indices:
        print(f"{class_names[i]}: {probabilities[i] * 100:.2f}%")


if __name__ == "__main__":
    main()
