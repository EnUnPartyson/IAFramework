"""Arquitecturas PyTorch. Debe mantenerse equivalente capa a capa con model_defs_tensorflow.py
(misma cantidad de convoluciones, filtros, kernels y cabezal denso) para que la comparacion
TF vs PyTorch sea valida. Nunca importar tensorflow aca.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DETECTOR_CLASS_NAMES = ("gato", "ninguno", "perro")
NONE_CLASS_IDX = DETECTOR_CLASS_NAMES.index("ninguno")

# alias retrocompatible con codigo previo del detector
CLASS_NAMES = DETECTOR_CLASS_NAMES


HEAD_FLATTEN = "flatten"
HEAD_GAP = "gap"


class SimpleCNN(nn.Module):
    """CNN desde cero usada por los 3 modelos; cambian num_classes y el cabezal.

    head="flatten": Flatten -> Linear(256*8*8, 256). ~4.2M parametros solo en esa capa,
        el 91% del modelo. Va bien cuando hay muchos datos (detector: ~25k imagenes).
    head="gap": GlobalAveragePooling -> Linear(256, 256). ~66k parametros, 60x menos.
        Necesario en las razas (~1.9k imagenes), donde el cabezal grande memoriza.
    """

    def __init__(self, num_classes: int, dropout: float = 0.4, head: str = HEAD_FLATTEN) -> None:
        super().__init__()
        if head not in (HEAD_FLATTEN, HEAD_GAP):
            raise ValueError(f"head debe ser '{HEAD_FLATTEN}' o '{HEAD_GAP}', no {head!r}")
        self.head = head
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        if head == HEAD_FLATTEN:
            # 4 maxpools sobre input 128x128 -> mapa 8x8; si IMG_SIZE cambia, ajustar tambien aca
            first_layer: nn.Module = nn.Flatten()
            in_features = 256 * 8 * 8
        else:
            first_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            in_features = 256  # independiente del tamano de imagen

        self.classifier = nn.Sequential(
            first_layer,
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# alias retrocompatible: el detector original se llamaba DetectorCNN
DetectorCNN = SimpleCNN


def predict_forced(logits: torch.Tensor) -> torch.Tensor:
    # ignora la clase "ninguno" para el modo forzado (siempre devuelve perro o gato)
    masked = logits.clone()
    masked[:, NONE_CLASS_IDX] = float("-inf")
    return masked.argmax(dim=1)


UNKNOWN_IDX = -1


def predict_with_unknown(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Modo "raza no identificada": devuelve UNKNOWN_IDX si la confianza no llega al umbral.

    El umbral sugerido para cada modelo queda en metrics/<tarea>_<fw>_metrics.json
    (seccion "raza_no_identificada"), calculado sobre validacion.
    """
    probs = torch.softmax(logits, dim=1)
    maxprob, idx = probs.max(dim=1)
    idx = idx.clone()
    idx[maxprob < threshold] = UNKNOWN_IDX
    return idx
