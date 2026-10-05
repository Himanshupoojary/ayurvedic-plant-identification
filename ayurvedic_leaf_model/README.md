# Ayurvedic Herbal Plant Identification — Plant-Part Models

## Current scope
This repository supports the same three-model ensemble for:

- leaf
- stem/bark
- root
- flower
- fruit

The current architecture is:

```text
                 PLANT-PART IMAGE
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
     MobileNetV2     ResNet50     EfficientNetB0
          |             |             |
          +-------------+-------------+
                        |
                 Probability Fusion
                        |
                        v
             FINAL PLANT-PART PREDICTION
```

The three models see the same complete image for the selected plant part.

## What is implemented

- Dataset folder structure
- Train/validation/test loading
- Image augmentation
- Transfer learning with:
  - MobileNetV2
  - ResNet50
  - EfficientNetB0
- Model checkpointing
- Accuracy/loss plots
- Confusion matrix
- Single-model prediction
- Three-model probability fusion
- Class-name persistence
- Part-specific model, class-name, and result directories
- Validation-accuracy-weighted fusion (with an equal-weight baseline option)

## Dataset format

Put all raw images in one common directory. Use one plant folder, then one
folder for each available plant part:

```text
data/raw/
    plant_a/
        leaf/
            image1.jpg
        stem_bark/
            image1.jpg
        root/
            image1.jpg
        flower/
            image1.jpg
        fruit/
            image1.jpg
    plant_b/
        leaf/
            image1.jpg
        flower/
            image1.jpg
```

The folder names `leaf`, `stem_bark`, `root`, `flower`, and `fruit` identify
the image type. Each part should have at least three images per plant, and
the same plant classes should ideally be represented for every part.

It creates:

```text
data/<part>/train/
data/<part>/val/
data/<part>/test/
```

with an 80/10/10 split by default.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 1. Prepare the dataset

```bash
python scripts/prepare_dataset.py --part leaf
python scripts/prepare_dataset.py --part stem_bark
python scripts/prepare_dataset.py --part root
python scripts/prepare_dataset.py --part flower
python scripts/prepare_dataset.py --part fruit
```

Optional:

```bash
python scripts/prepare_dataset.py --train 0.8 --val 0.1 --test 0.1
```

## 2. Train one model

```bash
python src/train.py --part leaf --model mobilenetv2 --epochs 10
```

Other choices:

```bash
python src/train.py --part leaf --model resnet50 --epochs 10
python src/train.py --part leaf --model efficientnetb0 --epochs 10
```

Train all three:

```bash
python src/train.py --part leaf --model all --epochs 10
```

Train the remaining plant-part branches with the same models:

```bash
python src/train.py --part stem_bark --model all --epochs 10
python src/train.py --part root --model all --epochs 10
python src/train.py --part flower --model all --epochs 10
python src/train.py --part fruit --model all --epochs 10
```

For a first run, keep the number of epochs small and verify that the pipeline works.

## 3. Predict with one model

```bash
python src/predict.py --model mobilenetv2 --image path/to/leaf.jpg
```

For another plant part:

```bash
python src/predict.py --part flower --model mobilenetv2 --image path/to/flower.jpg
```

## 4. Fuse all three models

After all three models have been trained:

```bash
python src/fusion.py --image path/to/leaf.jpg
```

By default, fusion now calculates each model's accuracy on the **validation
set** and normalizes those accuracies into weights:

```text
weight_i = validation_accuracy_i / sum(validation_accuracies)
P_final = sum(weight_i * P_i)
```

Validation accuracy is used for prediction-time weighting; test accuracy is
only used for final evaluation and is never used to choose the weights. To
run the original equal-weight baseline:

```bash
python src/fusion.py --image path/to/leaf.jpg --equal-weight
```

For example, flower identification uses:

```bash
python src/fusion.py --part flower --image path/to/flower.jpg
```

## 5. Identify with any available images

Identification does not require all five plant parts. Provide one or more
images using the common command:

```bash
python src/identify.py --leaf path/to/leaf.jpg
python src/identify.py --leaf path/to/leaf.jpg --flower path/to/flower.jpg
```

The command continues with just one image. When multiple images are
provided, each available part is predicted independently and the resulting
part probability vectors are averaged. Missing parts are skipped; they do
not stop identification. Every supplied part must already have its three
models trained.

## Outputs

Trained models:

```text
models/
    mobilenetv2_best.keras
    resnet50_best.keras
    efficientnetb0_best.keras
```

Additional parts are stored below `models/<part>/`.

Results:

```text
results/
    mobilenetv2_history.png
    mobilenetv2_confusion_matrix.png
    ...
```

Class names:

```text
models/class_names.json
```

## Important project note

This is the **first 20% baseline**, not the final architecture.

Later we can add:

1. Learned fusion across plant parts
2. Explainable AI / Grad-CAM
3. Real-world camera testing
4. Medicinal recommendation layer
