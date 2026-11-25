"""Run pretrained YOLOv8n on a road image or the project's COCO subset."""

import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGES = ROOT / "data/coco-2017-yolo-50/images/train"
EXAMPLE = TRAIN_IMAGES / "000000061181.jpg"
WEIGHTS = ROOT / "models/yolov8n.pt"
CLASSES = ("car", "truck", "bicycle", "person")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, default=EXAMPLE, help="road image to annotate")
    source.add_argument("--all", action="store_true", help="summarize the entire COCO subset")
    args = parser.parse_args()
    inputs = sorted(TRAIN_IMAGES.glob("*.jpg")) if args.all else [args.image]
    if not inputs or not all(path.is_file() for path in inputs):
        parser.error("Missing input image(s). Restore the COCO subset or supply --image PATH.")

    WEIGHTS.parent.mkdir(exist_ok=True)
    model = YOLO(WEIGHTS)
    class_ids = [i for i, name in model.names.items() if name in CLASSES]
    counts = Counter(dict.fromkeys(CLASSES, 0))
    detections = {}
    output = ROOT / "runs/task2"
    output.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        result = model.predict(path, device="cpu", conf=0.25, classes=class_ids, verbose=False)[0]
        objects = result.summary()
        detections[path.name] = objects
        counts.update(obj["name"] for obj in objects)
        if not args.all:
            result.save(filename=str(output / f"{path.stem}-annotated.jpg"))

    report = output / ("subset.json" if args.all else f"{inputs[0].stem}.json")
    report.write_text(json.dumps({"counts": counts, "images": detections}, indent=2) + "\n")
    print(f"{len(inputs)} image(s): {dict(counts)}\n{report}")


if __name__ == "__main__":
    main()
