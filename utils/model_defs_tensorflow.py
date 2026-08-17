"""Arquitecturas TensorFlow/Keras. Espejo capa a capa de model_defs_pytorch.SimpleCNN:
mismos filtros (32/64/128/256), mismos kernels 3x3, mismo cabezal denso (256 -> dropout -> logits).
La capa Rescaling inicial no tiene parametros: reemplaza la normalizacion que en PyTorch
hacen los transforms (mean=0.5, std=0.5 sobre [0,1] equivale a x/127.5 - 1 sobre [0,255]).
Nunca importar torch aca.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

DETECTOR_CLASS_NAMES = ("gato", "ninguno", "perro")
NONE_CLASS_IDX = DETECTOR_CLASS_NAMES.index("ninguno")

IMG_SIZE = 128

HEAD_FLATTEN = "flatten"
HEAD_GAP = "gap"

# filtros por bloque conv; blocks=N usa los primeros N (espejo de model_defs_pytorch)
BLOCK_FILTERS = (32, 64, 128, 256, 512)


def build_simple_cnn(
    num_classes: int,
    dropout: float = 0.4,
    img_size: int = IMG_SIZE,
    head: str = HEAD_FLATTEN,
    blocks: int = 4,
) -> tf.keras.Model:
    """Espejo de model_defs_pytorch.SimpleCNN, incluidos cabezal y profundidad (ver docstring alla)."""
    if head not in (HEAD_FLATTEN, HEAD_GAP):
        raise ValueError(f"head debe ser '{HEAD_FLATTEN}' o '{HEAD_GAP}', no {head!r}")
    if not 3 <= blocks <= len(BLOCK_FILTERS):
        raise ValueError(f"blocks debe estar entre 3 y {len(BLOCK_FILTERS)}, no {blocks}")
    if img_size % (2 ** blocks) != 0:
        raise ValueError(f"img_size={img_size} no es divisible por 2^{blocks}")

    steps: list = [
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1.0 / 127.5, offset=-1.0),
    ]
    for out_ch in BLOCK_FILTERS[:blocks]:
        steps += [
            layers.Conv2D(out_ch, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),
        ]
    steps += [
        layers.Flatten() if head == HEAD_FLATTEN else layers.GlobalAveragePooling2D(),
        layers.Dense(256),
        layers.ReLU(),
        layers.Dropout(dropout),
        layers.Dense(num_classes),  # logits, igual que la version PyTorch
    ]
    return tf.keras.Sequential(steps)


def predict_forced(logits: np.ndarray) -> np.ndarray:
    # ignora la clase "ninguno" para el modo forzado (siempre devuelve perro o gato)
    masked = logits.copy()
    masked[:, NONE_CLASS_IDX] = -np.inf
    return masked.argmax(axis=1)


UNKNOWN_IDX = -1


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def predict_with_unknown(logits: np.ndarray, threshold: float) -> np.ndarray:
    """Modo "raza no identificada": devuelve UNKNOWN_IDX si la confianza no llega al umbral.

    El umbral sugerido para cada modelo queda en metrics/<tarea>_<fw>_metrics.json
    (seccion "raza_no_identificada"), calculado sobre validacion.
    """
    probs = softmax(logits)
    idx = probs.argmax(axis=1)
    idx[probs.max(axis=1) < threshold] = UNKNOWN_IDX
    return idx
