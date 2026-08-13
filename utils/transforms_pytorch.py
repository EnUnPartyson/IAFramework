from __future__ import annotations

from torchvision import transforms

IMG_SIZE = 128
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        # scale/posicion variables: en camara real la mascota no siempre aparece centrada y grande
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
