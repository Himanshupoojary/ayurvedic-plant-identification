import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PLANT_PARTS = ("leaf", "stem_bark", "root", "flower", "fruit")


def collect_images(class_dir, part):
    part_dir = class_dir / part
    if not part_dir.exists():
        return []

    return [
        p for p in part_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--part", choices=PLANT_PARTS, required=True)
    parser.add_argument(
        "--output",
        help="Dataset output root. Defaults to data/<part>.",
    )
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1.0")

    raw_dir = Path(args.raw)
    output_dir = Path(args.output) if args.output else Path("data") / args.part

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Put class folders inside data/raw first."
        )

    random.seed(args.seed)

    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    if not class_dirs:
        raise ValueError("No class folders found in data/raw.")

    for class_dir in class_dirs:
        images = collect_images(class_dir, args.part)
        if len(images) < 3:
            print(
                f"Skipping {class_dir.name}: fewer than 3 "
                f"{args.part} images."
            )
            continue

        random.shuffle(images)

        n = len(images)
        train_end = int(n * args.train)
        val_end = train_end + int(n * args.val)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, files in splits.items():
            destination = output_dir / split / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)

            for index, source in enumerate(files):
                # Prefix avoids collisions when source datasets contain repeated names.
                target = destination / f"{index:05d}_{source.name}"
                shutil.copy2(source, target)

        print(
            f"{class_dir.name}: "
            f"{len(splits['train'])} train / "
            f"{len(splits['val'])} val / "
            f"{len(splits['test'])} test"
        )

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
