from __future__ import annotations

from torchvision import transforms

IMG_SIZE = 128
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

AUG_BASE = "base"
AUG_STRONG = "strong"


def get_train_transforms(img_size: int = IMG_SIZE, aug: str = AUG_BASE) -> transforms.Compose:
    if aug not in (AUG_BASE, AUG_STRONG):
        raise ValueError(f"aug debe ser '{AUG_BASE}' o '{AUG_STRONG}', no {aug!r}")

    steps: list = [
        # scale/posicion variables: en camara real la mascota no siempre aparece centrada y grande
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if aug == AUG_STRONG:
        # razas: pocos datos por clase -> augmentation agresiva. RandAugment aplica 2 ops
        # aleatorias (contraste, cizalla, posterizado, etc) por imagen; el espejo TF usa
        # la capa RandAugment de Keras.
        steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
    else:
        steps += [
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
    if aug == AUG_STRONG:
        # borra un parche aleatorio: fuerza a mirar mas de un rasgo del animal
        steps.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(steps)


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
