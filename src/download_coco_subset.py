"""Download and export a reproducible 50-image COCO-2017 training subset."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import fiftyone as fo
import fiftyone.zoo as foz


CLASSES = ("car", "truck", "bicycle", "person")
NUM_SAMPLES = 50
DATASET_NAME = "coco-2017-train-yolo-50"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IMAGE_IDS = DATA_DIR / "coco-2017-train-50.txt"
EXPORT_DIR = DATA_DIR / "coco-2017-yolo-50"


def main() -> None:
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=list(CLASSES),
        only_matching=True,
        image_ids=str(IMAGE_IDS),
        dataset_name=DATASET_NAME,
        drop_existing_dataset=True,
    )

    if len(dataset) != NUM_SAMPLES:
        raise RuntimeError(
            f"Expected {NUM_SAMPLES} matching images, got {len(dataset)}"
        )

    EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=EXPORT_DIR.parent) as tmp:
        export = Path(tmp) / EXPORT_DIR.name
        dataset.export(
            export_dir=str(export),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=list(CLASSES),
            split="train",
            include_path=False,
        )
        if EXPORT_DIR.exists():
            shutil.rmtree(EXPORT_DIR)
        export.rename(EXPORT_DIR)
    print(f"Exported {len(dataset)} images to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
