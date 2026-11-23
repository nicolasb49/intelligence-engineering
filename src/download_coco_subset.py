"""Download and export a reproducible 50-image COCO-2017 training subset."""

import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


CLASSES = ("car", "truck", "bicycle", "person")
NUM_SAMPLES = 50
SEED = 42
DATASET_NAME = "coco-2017-train-yolo-50"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPO_ROOT / "data" / "processed" / "coco-2017-yolo-50"


def main() -> None:
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=list(CLASSES),
        only_matching=True,
        max_samples=NUM_SAMPLES,
        shuffle=True,
        seed=SEED,
        dataset_name=DATASET_NAME,
        drop_existing_dataset=True,
    )

    if len(dataset) != NUM_SAMPLES:
        raise RuntimeError(
            f"Expected {NUM_SAMPLES} matching images, got {len(dataset)}"
        )

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    dataset.export(
        export_dir=str(EXPORT_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=list(CLASSES),
        split="train",
    )
    print(f"Exported {len(dataset)} images to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
