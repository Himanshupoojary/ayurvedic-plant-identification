import json
from pathlib import Path

import tensorflow as tf

from config import (
    IMAGE_SIZE, BATCH_SIZE, SEED, model_dir, data_dir
)

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(part="leaf"):
    dataset_dir = data_dir(part)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"
    part_model_dir = model_dir(part)

    common = dict(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        seed=SEED,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, shuffle=False, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, **common
    )

    class_names = train_ds.class_names

    if val_ds.class_names != class_names or test_ds.class_names != class_names:
        raise ValueError("Train/val/test class folders do not match.")

    part_model_dir.mkdir(parents=True, exist_ok=True)
    with open(part_model_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
        class_names,
    )
