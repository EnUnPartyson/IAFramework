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
            # jitter de luz mas ancho que el original (0.2): en produccion la camara varia
            # mucho mas de lo que varia un dataset curado
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25),
        ]
    # Domain shift: los modelos se entrenan con fotos de dataset (nitidas, bien enfocadas) pero
    # en produccion ven una webcam, con desenfoque de movimiento y foco pobre. Sin esto la
    # accuracy medida en test no se traslada a la camara. Se aplica en los dos modos de aug
    # porque los 3 modelos terminan en el mismo pipeline de camara.
    steps += [
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))], p=0.30),
        transforms.RandomAdjustSharpness(sharpness_factor=0.4, p=0.15),
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
