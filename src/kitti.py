"""KITTI paths, 2D annotations and the YOLO smoke-test split."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data/raw/kitti/training"
SPLIT = json.loads((ROOT / "data/kitti-smoke-split.json").read_text())
CLASSES = ("Car", "Truck", "Cyclist", "Pedestrian")


def image_path(image_id):
    return TRAINING / "image_2" / f"{image_id}.png"


def annotations(image_id, width, height):
    """Read the four experiment classes as clipped KITTI pixel coordinates."""
    objects = []
    label_file = TRAINING / "label_2" / f"{image_id}.txt"
    for line in label_file.read_text().splitlines():
        fields = line.split()
        name = fields[0]
        if name not in CLASSES:
            continue
        left, top, right, bottom = map(float, fields[4:8])
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, width), min(bottom, height)
        if right > left and bottom > top:
            objects.append((name, (left, top, right, bottom)))
    return objects
