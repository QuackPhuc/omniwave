"""Tests for ImageNet data validation and transforms."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from omniweave.data.imagenet import DistributedEvalSampler, validate_imagenet
from omniweave.data.transforms import build_eval_transform, build_train_transform


def _create_fake_imagenet(
    tmp_path: Path,
    n_classes: int = 2,
    n_train_per_class: int = 3,
    n_val_per_class: int = 2,
) -> Path:
    """Create a minimal ImageNet-style directory for testing."""
    root = tmp_path / "imagenet"
    for split in ("train", "val"):
        n_per = n_train_per_class if split == "train" else n_val_per_class
        for cls_idx in range(n_classes):
            cls_name = f"n{cls_idx:08d}"
            cls_dir = root / split / cls_name
            cls_dir.mkdir(parents=True)
            for img_idx in range(n_per):
                Image.new("RGB", (2, 2), color=(cls_idx, img_idx, 0)).save(
                    cls_dir / f"img_{img_idx}.JPEG"
                )
    return root


def test_validate_matching_splits(tmp_path: Path) -> None:
    root = _create_fake_imagenet(tmp_path, n_classes=2)
    manifest = validate_imagenet(root, expected_classes=2)
    assert manifest.train_samples == 6
    assert manifest.val_samples == 4
    assert len(manifest.class_to_idx) == 2
    assert len(manifest.relative_paths_sha256) == 64


def test_validate_class_mismatch(tmp_path: Path) -> None:
    root = _create_fake_imagenet(tmp_path, n_classes=2)
    # Add an extra class only in train
    extra_cls = root / "train" / "n99999999"
    extra_cls.mkdir()
    Image.new("RGB", (2, 2)).save(extra_cls / "img_0.JPEG")

    with pytest.raises(ValueError, match="class mappings differ"):
        validate_imagenet(root, expected_classes=None)


def test_validate_wrong_class_count(tmp_path: Path) -> None:
    root = _create_fake_imagenet(tmp_path, n_classes=2)
    with pytest.raises(ValueError, match="expected 1000 classes"):
        validate_imagenet(root, expected_classes=1000)


def test_validate_missing_train(tmp_path: Path) -> None:
    root = tmp_path / "imagenet_broken"
    (root / "val" / "cls0").mkdir(parents=True)
    with pytest.raises(ValueError, match="train directory not found"):
        validate_imagenet(root)


def test_validate_rejects_undecodable_image(tmp_path: Path) -> None:
    root = _create_fake_imagenet(tmp_path, n_classes=1)
    (root / "train" / "n00000000" / "broken.JPEG").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="cannot be decoded"):
        validate_imagenet(root, expected_classes=1)


def test_train_transform_exists() -> None:
    t = build_train_transform(224)
    assert t is not None


def test_eval_transform_exists() -> None:
    t = build_eval_transform(224)
    assert t is not None


def test_distributed_eval_sampler_does_not_pad_duplicates() -> None:
    dataset = list(range(5))
    rank_zero = list(DistributedEvalSampler(
        dataset,
        num_replicas=2,
        rank=0,
    ))
    rank_one = list(DistributedEvalSampler(
        dataset,
        num_replicas=2,
        rank=1,
    ))
    combined = rank_zero + rank_one
    assert sorted(combined) == list(range(5))
    assert len(combined) == len(set(combined))
