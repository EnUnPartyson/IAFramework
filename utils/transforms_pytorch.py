from __future__ import annotations

import io
import random

from PIL import Image
from torchvision import transforms

IMG_SIZE = 128
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]
# estadisticas de ImageNet: obligatorias cuando el modelo parte de pesos preentrenados
# (modo pro), porque el backbone aprendio con entradas normalizadas asi
NORM_IMAGENET_MEAN = [0.485, 0.456, 0.406]
NORM_IMAGENET_STD = [0.229, 0.224, 0.225]

AUG_BASE = "base"
AUG_STRONG = "strong"
# "produccion" (solo modo pro, PyTorch): strong + degradaciones tipicas de una foto real de
# celular/webcam que las fotos de dataset no tienen: recompresion JPEG, perspectiva no
# frontal y perdida de color por mala luz. El espejo TF no lo implementa: el pro es solo PyTorch.
AUG_PRODUCCION = "produccion"

NORM_SIMPLE = "simple"      # [-1, 1], la de los modelos desde cero
NORM_IMAGENET = "imagenet"  # la de los backbones preentrenados (modo pro)


def _norm_stats(norm: str) -> tuple[list[float], list[float]]:
    if norm == NORM_SIMPLE:
        return NORM_MEAN, NORM_STD
    if norm == NORM_IMAGENET:
        return NORM_IMAGENET_MEAN, NORM_IMAGENET_STD
    raise ValueError(f"norm debe ser '{NORM_SIMPLE}' o '{NORM_IMAGENET}', no {norm!r}")


class JPEGAleatorio:
    """Re-comprime la imagen como JPEG de calidad aleatoria (sobre PIL, antes de ToTensor).

    Las fotos que llegan por la app o la webcam vienen recomprimidas (a veces varias veces);
    los datasets de entrenamiento no. torchvision 0.28 no trae un transform JPEG, por eso
    existe este.
    """

    def __init__(self, calidad: tuple[int, int] = (30, 90), p: float = 0.30) -> None:
        self.calidad = calidad
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=random.randint(*self.calidad))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def get_train_transforms(
    img_size: int = IMG_SIZE, aug: str = AUG_BASE, norm: str = NORM_SIMPLE
) -> transforms.Compose:
    if aug not in (AUG_BASE, AUG_STRONG, AUG_PRODUCCION):
        raise ValueError(f"aug debe ser '{AUG_BASE}', '{AUG_STRONG}' o '{AUG_PRODUCCION}', no {aug!r}")
    mean, std = _norm_stats(norm)
    fuerte = aug in (AUG_STRONG, AUG_PRODUCCION)

    steps: list = [
        # scale/posicion variables: en camara real la mascota no siempre aparece centrada y grande
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if fuerte:
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
    if aug == AUG_PRODUCCION:
        steps += [
            JPEGAleatorio(calidad=(30, 90), p=0.30),
            # la mascota rara vez esta perfectamente de frente a la camara
            transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
            # camaras baratas con poca luz lavan el color
            transforms.RandomGrayscale(p=0.05),
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if fuerte:
        # borra un parche aleatorio: fuerza a mirar mas de un rasgo del animal
        steps.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(steps)


def get_eval_transforms(img_size: int = IMG_SIZE, norm: str = NORM_SIMPLE) -> transforms.Compose:
    mean, std = _norm_stats(norm)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
