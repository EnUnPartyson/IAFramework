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


def build_simple_cnn(
    num_classes: int, dropout: float = 0.4, img_size: int = IMG_SIZE, head: str = HEAD_FLATTEN
) -> tf.keras.Model:
    """Espejo de model_defs_pytorch.SimpleCNN, incluido el cabezal (ver docstring alla)."""
    if head not in (HEAD_FLATTEN, HEAD_GAP):
        raise ValueError(f"head debe ser '{HEAD_FLATTEN}' o '{HEAD_GAP}', no {head!r}")

    head_layer = layers.Flatten() if head == HEAD_FLATTEN else layers.GlobalAveragePooling2D()

    model = tf.keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1.0 / 127.5, offset=-1.0),

        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        layers.Conv2D(256, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        head_layer,
        layers.Dense(256),
        layers.ReLU(),
        layers.Dropout(dropout),
        layers.Dense(num_classes),  # logits, igual que la version PyTorch
    ])
    return model


def predict_forced(logits: np.ndarray) -> np.ndarray:
    # ignora la clase "ninguno" para el modo forzado (siempre devuelve perro o gato)
    masked = logits.copy()
    masked[:, NONE_CLASS_IDX] = -np.inf
    return masked.argmax(axis=1)
