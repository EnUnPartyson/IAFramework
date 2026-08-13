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


class SimpleCNN(nn.Module):
    """CNN desde cero usada por los 3 modelos (detector y ambas razas); solo cambia num_classes."""

    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()
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
        # 4 maxpools sobre input 128x128 -> mapa 8x8; si IMG_SIZE cambia, ajustar tambien aca
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
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
