"""Dataset module for loading symbol crops from COCO-annotated drawings.

Uses the SimCLR repo's augmentation pattern (ContrastiveLearningViewGenerator).
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

# Add SimCLR repo to path so we can import its components
sys.path.insert(0, str(Path(__file__).parent / "SimCLR"))
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator


class SymbolDataset(Dataset):
    """Loads cropped symbol patches from COCO-annotated drawing images.

    Each sample is a cropped symbol image with its category label.
    When using a ContrastiveLearningViewGenerator transform, returns
    [view1, view2, ...] list plus the label — matching the SimCLR repo's pattern.
    """

    def __init__(self, data_dir, split="train", transform=None,
                 min_crop_size=10, pad_ratio=0.1):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.min_crop_size = min_crop_size
        self.pad_ratio = pad_ratio

        ann_path = self.data_dir / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        # Build category mapping (skip parent category id=0 "firefighting-devices")
        self.categories = {}
        self.cat_id_to_label = {}
        label_idx = 0
        for cat in sorted(coco["categories"], key=lambda c: c["id"]):
            if cat["name"] == "firefighting-devices":
                continue
            self.categories[cat["id"]] = cat["name"]
            self.cat_id_to_label[cat["id"]] = label_idx
            label_idx += 1

        self.num_classes = len(self.categories)
        self.label_to_name = {v: self.categories[k] for k, v in self.cat_id_to_label.items()}

        # Build image lookup
        images = {img["id"]: img for img in coco["images"]}

        # Build samples: (image_path, bbox, label)
        self.samples = []
        for ann in coco["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in self.categories:
                continue
            img_info = images[ann["image_id"]]
            img_path = self.data_dir / img_info["file_name"]
            bbox = ann["bbox"]  # [x, y, w, h] in COCO format
            if bbox[2] < self.min_crop_size or bbox[3] < self.min_crop_size:
                continue
            label = self.cat_id_to_label[cat_id]
            self.samples.append((str(img_path), bbox, label))

        # Cache opened images to avoid re-reading
        self._img_cache = {}

    def _get_image(self, img_path):
        if img_path not in self._img_cache:
            self._img_cache[img_path] = Image.open(img_path).convert("RGB")
        return self._img_cache[img_path]

    def _crop_symbol(self, img_path, bbox):
        img = self._get_image(img_path)
        x, y, w, h = bbox
        iw, ih = img.size

        # Add padding around the bounding box
        pad_x = w * self.pad_ratio
        pad_y = h * self.pad_ratio
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(iw, int(x + w + pad_x))
        y2 = min(ih, int(y + h + pad_y))

        return img.crop((x1, y1, x2, y2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        crop = self._crop_symbol(img_path, bbox)

        if self.transform:
            crop = self.transform(crop)

        # If transform is a ContrastiveLearningViewGenerator, crop is a list of views.
        # The SimCLR repo's training loop expects (images_list, label).
        return crop, label


def get_simclr_pipeline_transform(size, s=1):
    """SimCLR augmentation pipeline — matches the repo's implementation
    with additional augmentations suited for technical drawing symbols."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor(),
    ])
    return data_transforms


def get_simclr_transform(img_size=64, n_views=2, s=1):
    """Returns a ContrastiveLearningViewGenerator that produces n_views
    augmented views of each crop — used for SimCLR training."""
    base_transform = get_simclr_pipeline_transform(img_size, s=s)
    return ContrastiveLearningViewGenerator(base_transform, n_views)


def get_eval_transform(img_size=64):
    """Deterministic transform for evaluation (single view)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def get_class_distribution(dataset):
    """Return a dict of label -> count."""
    dist = {}
    for _, _, label in dataset.samples:
        dist[label] = dist.get(label, 0) + 1
    return dist


class UniqueClassBatchSampler(Sampler):
    """Batch sampler where every sample in a batch comes from a different class.

    Each batch has exactly batch_size samples, each from a unique class.
    Requires batch_size <= num_classes.
    """

    def __init__(self, dataset, batch_size, drop_last=True):
        self.batch_size = batch_size

        # Group sample indices by class label
        self.class_to_indices = defaultdict(list)
        for idx, (_, _, label) in enumerate(dataset.samples):
            self.class_to_indices[label].append(idx)

        self.num_classes = len(self.class_to_indices)
        assert batch_size <= self.num_classes, (
            f"batch_size ({batch_size}) must be <= num_classes ({self.num_classes}) "
            f"to guarantee unique classes per batch"
        )

    def __iter__(self):
        # Shuffle indices within each class
        class_pools = {}
        for label, indices in self.class_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            class_pools[label] = shuffled

        class_labels = list(class_pools.keys())
        # Track position within each class pool
        class_pos = {label: 0 for label in class_labels}

        batches = []
        keep_going = True

        while keep_going:
            # Collect one sample from each class that still has data
            available = []
            for label in class_labels:
                pos = class_pos[label]
                if pos < len(class_pools[label]):
                    available.append((label, class_pools[label][pos]))
                    class_pos[label] = pos + 1

            if len(available) < self.batch_size:
                keep_going = False
                break

            # Shuffle available and take batch_size items
            random.shuffle(available)
            for i in range(0, len(available) - self.batch_size + 1, self.batch_size):
                batch = [idx for _, idx in available[i:i + self.batch_size]]
                batches.append(batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        # Approximate: total samples across all classes / batch_size
        total = sum(len(v) for v in self.class_to_indices.values())
        return total // self.batch_size
