"""ImageNet dataset validation and loader construction."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Sampler, Subset

from omniweave.data.transforms import build_eval_transform, build_train_transform


@dataclass(frozen=True)
class DatasetManifest:
    """Validated metadata about an ImageNet-style dataset."""

    train_samples: int
    val_samples: int
    class_to_idx: dict[str, int]
    relative_paths_sha256: str


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation samples across ranks without padding duplicates."""

    def __init__(
        self,
        data_source: Sized,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        if num_replicas < 1:
            raise ValueError("num_replicas must be at least 1")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset_size = len(data_source)
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.num_replicas))

    def __len__(self) -> int:
        remaining = max(self.dataset_size - self.rank, 0)
        return (remaining + self.num_replicas - 1) // self.num_replicas


def _sorted_class_dirs(split_root: Path) -> list[str]:
    """Return sorted class directory names under *split_root*."""
    return sorted(
        d.name for d in split_root.iterdir() if d.is_dir()
    )


def _collect_relative_paths(split_root: Path) -> list[str]:
    """Collect sorted relative image paths under *split_root*."""
    paths: list[str] = []
    for cls_dir in sorted(split_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.is_file():
                paths.append(str(img_path.relative_to(split_root)))
    return paths


def _validate_decodable_images(split_root: Path, relative_paths: list[str]) -> None:
    """Ensure every image can be decoded into a non-empty RGB tensor source."""
    for relative_path in relative_paths:
        image_path = split_root / relative_path
        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                if rgb.width < 1 or rgb.height < 1:
                    raise ValueError(f"image has invalid dimensions: {image_path}")
                rgb.load()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"image cannot be decoded: {image_path}") from exc


def validate_imagenet(
    root: str | Path,
    expected_classes: int = 1000,
) -> DatasetManifest:
    """Validate an ImageNet-style directory and return its manifest.

    Raises ``ValueError`` on structural problems.
    """
    root = Path(root)
    train_root = root / "train"
    val_root = root / "val"

    if not train_root.is_dir():
        raise ValueError(f"train directory not found: {train_root}")
    if not val_root.is_dir():
        raise ValueError(f"val directory not found: {val_root}")

    train_classes = _sorted_class_dirs(train_root)
    val_classes = _sorted_class_dirs(val_root)

    if train_classes != val_classes:
        raise ValueError(
            "class mappings differ between train and val splits"
        )

    if expected_classes is not None and len(train_classes) != expected_classes:
        raise ValueError(
            f"expected {expected_classes} classes, found {len(train_classes)}"
        )

    class_to_idx = {name: i for i, name in enumerate(train_classes)}

    train_paths = _collect_relative_paths(train_root)
    val_paths = _collect_relative_paths(val_root)

    if len(train_paths) != len(set(train_paths)):
        raise ValueError("duplicate relative paths in train split")
    if len(val_paths) != len(set(val_paths)):
        raise ValueError("duplicate relative paths in val split")

    _validate_decodable_images(train_root, train_paths)
    _validate_decodable_images(val_root, val_paths)

    all_paths = sorted(
        [f"train/{path}" for path in train_paths]
        + [f"val/{path}" for path in val_paths]
    )
    path_hash = hashlib.sha256("\n".join(all_paths).encode()).hexdigest()

    return DatasetManifest(
        train_samples=len(train_paths),
        val_samples=len(val_paths),
        class_to_idx=class_to_idx,
        relative_paths_sha256=path_hash,
    )


def _worker_init_fn(worker_id: int) -> None:
    """Seed each data worker deterministically."""
    import numpy as np
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


def build_imagenet_loaders(
    config: dict[str, Any],
    distributed: bool = False,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DatasetManifest]:
    """Build train and val DataLoaders from config.

    Config keys under ``data``:
    - ``root``: ImageNet root directory
    - ``image_size``: input resolution (default 224)
    - ``subset_samples``: optional subset for overfitting tests
    """
    from torchvision.datasets import ImageFolder

    data_cfg = config.get("data", config)
    root = Path(data_cfg["root"])
    image_size = data_cfg.get("image_size", 224)
    subset_samples = data_cfg.get("subset_samples", None)
    if batch_size is None:
        batch_size = config.get("train", {}).get("global_batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)

    manifest = validate_imagenet(root, expected_classes=data_cfg.get("expected_classes", 1000))

    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)

    train_dataset = ImageFolder(root / "train", transform=train_transform)
    val_dataset = ImageFolder(root / "val", transform=eval_transform)

    if subset_samples is not None:
        indices = list(range(min(subset_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True,
        )
        val_sampler = DistributedEvalSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader, manifest
