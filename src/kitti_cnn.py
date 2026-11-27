"""Dataset and evaluation for the drive-disjoint KITTI CNN experiment."""
from functools import cached_property

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import pil_to_tensor

from .kitti import CLASSES, TRAINING, annotations, image_path

SIZE = (640, 192)  # width, height


class Scenes(Dataset):
    def __init__(self, ids, augment=False):
        self.ids = list(ids)
        self.augment = augment
        self.jitter = ColorJitter(brightness=0.18, contrast=0.15, saturation=0.18)

    @cached_property
    def samples(self):
        # Cache resized RGB bytes so each epoch can draw fresh augmentations.
        samples = []
        for key in self.ids:
            with Image.open(image_path(key)) as photo:
                width, height = photo.size
                image = photo.convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)
            objects = annotations(key, width, height)
            boxes = [box for _, box in objects]
            labels = [CLASSES.index(name) + 1 for name, _ in objects]
            crowd = [False] * len(boxes)
            if not self.augment:
                for line in (TRAINING / "label_2" / f"{key}.txt").read_text().splitlines():
                    fields = line.split()
                    if fields[0] == "DontCare":
                        x1, y1, x2, y2 = map(float, fields[4:8])
                        box = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                        if box[2] > box[0] and box[3] > box[1]:
                            boxes.extend([box] * len(CLASSES))
                            labels.extend(range(1, len(CLASSES) + 1))
                            crowd.extend([True] * len(CLASSES))
            scale = torch.tensor([SIZE[0] / width, SIZE[1] / height] * 2)
            target = dict(boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4) * scale,
                          labels=torch.tensor(labels, dtype=torch.long),
                          iscrowd=torch.tensor(crowd, dtype=torch.bool))
            samples.append((image, target))
        return samples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, target = self.samples[index]
        if self.augment:
            image = self.jitter(image)
            if torch.rand(1).item() < 0.5:
                image = ImageOps.mirror(image)
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]
                target = {**target, "boxes": boxes}
        return pil_to_tensor(image).float().div_(255), target


def batches(scenes, shuffle=False, seed=42):
    return DataLoader(scenes, batch_size=8, shuffle=shuffle, collate_fn=list,
                      generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def evaluate(model, scenes):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True)
    support = torch.zeros(len(CLASSES) + 1, dtype=torch.long)
    for batch in batches(scenes):
        images, targets = zip(*batch)
        metric.update(model(list(images)), list(targets))
        for target in targets:
            labels = target["labels"][~target["iscrowd"]]
            support += labels.bincount(minlength=len(support))
    scores = metric.compute()
    if not support[1:].all():
        raise ValueError("Evaluation split has no reference box for at least one class")
    per_class = {CLASSES[i - 1]: value for i, value in
                 zip(scores["classes"].tolist(), scores["map_per_class"].tolist())}
    return {"mAP": scores["map"].item(), "AP50": scores["map_50"].item(),
            "AP75": scores["map_75"].item(), **per_class}, support[1:].tolist()
