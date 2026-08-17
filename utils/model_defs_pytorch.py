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

# filtros por bloque conv; blocks=N usa los primeros N
BLOCK_FILTERS = (32, 64, 128, 256, 512)


class SimpleCNN(nn.Module):
    """CNN desde cero usada por los 3 modelos; cambian num_classes, cabezal y profundidad.

    head="flatten": Flatten -> Linear grande. Mucha capacidad: va bien cuando hay muchos
        datos (detector, ~25k imagenes). Depende de img_size.
    head="gap": GlobalAveragePooling -> Linear chico. 60x menos parametros en el cabezal:
        necesario en las razas, donde el cabezal grande memoriza. Independiente de img_size.
    blocks=5 agrega un bloque de 512 filtros: mas profundidad de features para tareas
        fine-grained (razas); el detector queda en 4.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.4,
        head: str = HEAD_FLATTEN,
        blocks: int = 4,
        img_size: int = 128,
    ) -> None:
        super().__init__()
        if head not in (HEAD_FLATTEN, HEAD_GAP):
            raise ValueError(f"head debe ser '{HEAD_FLATTEN}' o '{HEAD_GAP}', no {head!r}")
        if not 3 <= blocks <= len(BLOCK_FILTERS):
            raise ValueError(f"blocks debe estar entre 3 y {len(BLOCK_FILTERS)}, no {blocks}")
        if img_size % (2 ** blocks) != 0:
            raise ValueError(f"img_size={img_size} no es divisible por 2^{blocks}")
        self.head = head

        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in BLOCK_FILTERS[:blocks]:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        if head == HEAD_FLATTEN:
            feature_map = img_size // (2 ** blocks)
            first_layer: nn.Module = nn.Flatten()
            in_features = in_ch * feature_map * feature_map
        else:
            first_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            in_features = in_ch  # independiente del tamano de imagen

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
