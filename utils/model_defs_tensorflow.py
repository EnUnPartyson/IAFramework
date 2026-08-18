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

# BatchNorm: Keras y PyTorch usan convenciones OPUESTAS para momentum.
#   PyTorch: running = (1 - m) * running + m * batch      -> default m=0.1
#   Keras:   moving  = m * moving      + (1 - m) * batch  -> default m=0.99
# El 0.1 de PyTorch equivale a 0.9 en Keras. Dejar el default (0.99) actualiza las
# estadisticas 10x mas lento: en datasets chicos (razas, ~78 pasos por epoca) nunca
# convergen, y como validacion/test las usan, la accuracy se hunde. Ver DECISIONS.md.
BN_MOMENTUM = 0.9      # equivalente al momentum=0.1 de PyTorch
BN_EPSILON = 1e-5      # el default de Keras es 1e-3, el de PyTorch 1e-5


def _pytorch_style_init() -> tf.keras.initializers.Initializer:
    """Replica la inicializacion por defecto de Conv2d/Linear de PyTorch.

    PyTorch usa kaiming_uniform con a=sqrt(5), que equivale a uniforme en
    +-sqrt(1/fan_in). Keras por defecto usa glorot_uniform, ~40% mas ancho.
    VarianceScaling(scale=1/3, fan_in, uniform) da limite=sqrt(3*(1/3)/fan_in)=sqrt(1/fan_in).
    """
    return tf.keras.initializers.VarianceScaling(scale=1.0 / 3.0, mode="fan_in", distribution="uniform")


def build_simple_cnn(
    num_classes: int,
    dropout: float = 0.4,
    img_size: int = IMG_SIZE,
    head: str = HEAD_FLATTEN,
    blocks: int = 4,
    augmentation: tf.keras.Sequential | None = None,
) -> tf.keras.Model:
    """Espejo de model_defs_pytorch.SimpleCNN, incluidos cabezal y profundidad (ver docstring alla).

    Si se pasa `augmentation`, sus capas van primero y por lo tanto se ejecutan en GPU junto
    con el resto del modelo. Son capas sin parametros y quedan inactivas en inferencia
    (training=False), asi que no alteran las predicciones ni el conteo de pesos.
    """
    if head not in (HEAD_FLATTEN, HEAD_GAP):
        raise ValueError(f"head debe ser '{HEAD_FLATTEN}' o '{HEAD_GAP}', no {head!r}")
    if not 3 <= blocks <= len(BLOCK_FILTERS):
        raise ValueError(f"blocks debe estar entre 3 y {len(BLOCK_FILTERS)}, no {blocks}")
    if img_size % (2 ** blocks) != 0:
        raise ValueError(f"img_size={img_size} no es divisible por 2^{blocks}")

    init = _pytorch_style_init()
    steps: list = [layers.Input(shape=(img_size, img_size, 3))]
    if augmentation is not None:
        steps.append(augmentation)
    steps.append(layers.Rescaling(1.0 / 127.5, offset=-1.0))
    for out_ch in BLOCK_FILTERS[:blocks]:
        steps += [
            layers.Conv2D(out_ch, 3, padding="same", kernel_initializer=init),
            layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON),
            layers.ReLU(),
            layers.MaxPooling2D(2),
        ]
    steps += [
        layers.Flatten() if head == HEAD_FLATTEN else layers.GlobalAveragePooling2D(),
        layers.Dense(256, kernel_initializer=init),
        layers.ReLU(),
        layers.Dropout(dropout),
        # dtype float32 explicito: con la policy mixed_float16 la salida debe quedar en
        # float32 por estabilidad numerica del softmax y la loss. Sin la policy es un no-op.
        layers.Dense(num_classes, kernel_initializer=init, dtype="float32"),  # logits
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
