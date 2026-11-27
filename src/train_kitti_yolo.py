"""Train YOLOv8n from scratch on the fixed 40-image KITTI smoke split."""
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import yaml
from PIL import Image
from ultralytics import YOLO

from .kitti import CLASSES, ROOT, SPLIT, annotations, image_path


def export_subset(directory):
    for split, image_ids in SPLIT.items():
        images = directory / "images" / split
        labels = directory / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for image_id in image_ids:
            source = image_path(image_id)
            copyfile(source, images / source.name)
            with Image.open(source) as photo:
                width, height = photo.size
            rows = []
            for name, (left, top, right, bottom) in annotations(image_id, width, height):
                cx, cy = (left + right) / (2 * width), (top + bottom) / (2 * height)
                w, h = (right - left) / width, (bottom - top) / height
                rows.append(f"{CLASSES.index(name)} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n")
            (labels / f"{image_id}.txt").write_text("".join(rows))
    config = directory / "dataset.yaml"
    config.write_text(yaml.safe_dump(dict(path=str(directory), train="images/train",
                                          val="images/val", names=list(CLASSES))))
    return config


def train():
    with TemporaryDirectory(prefix="kitti-smoke-") as temp:
        config = export_subset(Path(temp))
        model = YOLO("yolov8n.yaml")
        model.train(data=str(config), epochs=3, imgsz=320, batch=8, device="cpu",
                    workers=0, seed=42, pretrained=False, amp=False, plots=False,
                    project=str(ROOT / "runs/kitti"), name="yolo", verbose=False)
    return model.trainer.last


if __name__ == "__main__":
    print(train())
