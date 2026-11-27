"""Run the KITTI smoke-test detector on a road image or the fixed subset."""
import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from .kitti import CLASSES, ROOT, SPLIT, image_path

EXAMPLE = image_path(SPLIT["val"][0])
WEIGHTS = ROOT / "models/kitti-yolov8n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, default=EXAMPLE, help="road image to annotate")
    source.add_argument("--all", action="store_true", help="summarize all 50 KITTI subset images")
    args = parser.parse_args()
    inputs = ([image_path(i) for split in SPLIT.values() for i in split]
              if args.all else [args.image])
    if not all(path.is_file() for path in inputs):
        parser.error("Missing input image(s). Run dvc pull or supply --image PATH.")
    if not WEIGHTS.is_file():
        parser.error("Restore the KITTI checkpoint with dvc pull models/kitti-yolov8n.pt.dvc")

    model = YOLO(WEIGHTS)
    counts = Counter(dict.fromkeys(CLASSES, 0))
    detections = {}
    output = ROOT / "runs/kitti/task2"
    output.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        result = model.predict(path, device="cpu", imgsz=320, conf=0.25, verbose=False)[0]
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
