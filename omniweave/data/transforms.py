"""Data transforms for ImageNet training and evaluation."""

from __future__ import annotations

from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Training transform: RandAugment, random erasing, ImageNet normalization."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Evaluation transform: resize, center crop, ImageNet normalization."""
    resize_size = int(image_size / 0.875)
    return transforms.Compose([
        transforms.Resize(
            resize_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
