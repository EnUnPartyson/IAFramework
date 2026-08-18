"""Carga de datos y augmentation para los pipelines TensorFlow/Keras.

Lee los mismos directorios data/processed/<tarea>/{train,val,test}/<clase>/ que la version
PyTorch, asi ambos frameworks entrenan y evaluan sobre splits identicos.
La augmentation aproxima la de transforms_pytorch (flip horizontal, rotacion ~15 grados,
zoom que emula RandomResizedCrop, jitter de brillo/contraste). La normalizacion a [-1, 1]
vive dentro del modelo (capa Rescaling en model_defs_tensorflow).
Nunca importar torch aca.
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = 128

AUG_BASE = "base"
AUG_STRONG = "strong"


def build_augmentation(aug: str = AUG_BASE) -> tf.keras.Sequential:
    if aug not in (AUG_BASE, AUG_STRONG):
        raise ValueError(f"aug debe ser '{AUG_BASE}' o '{AUG_STRONG}', no {aug!r}")

    steps = [
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=(-0.3, 0.0)),  # solo zoom-in, emula RandomResizedCrop(0.5-1.0)
    ]
    if aug == AUG_STRONG:
        # espejo de torchvision RandAugment(num_ops=2, magnitude=9) + RandomErasing(p=0.25).
        # Los defaults de Keras NO coinciden: factor=(0, 0.5) en RandAugment (torchvision usa
        # 9/31 ~= 0.29) y factor=(0, 1.0) en RandomErasing (probabilidad media ~50% vs 25%).
        steps += [
            layers.RandAugment(value_range=(0, 255), num_ops=2, factor=0.29),
            layers.RandomErasing(value_range=(0, 255), factor=0.25),
        ]
    else:
        steps += [
            layers.RandomRotation(15 / 360),
            layers.RandomBrightness(0.2, value_range=(0.0, 255.0)),
            layers.RandomContrast(0.2),
        ]
    return tf.keras.Sequential(steps)


def count_train_images_per_class(data_dir: Path) -> tuple[list[str], list[int]]:
    train_dir = Path(data_dir) / "train"
    class_names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    counts = [len(list((train_dir / name).glob("*.jpg"))) for name in class_names]
    return class_names, counts


def make_datasets(
    data_dir: Path,
    batch_size: int,
    img_size: int = IMG_SIZE,
    aug: str = AUG_BASE,
    mixup: float = 0.0,
    gpu_augment: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Arma los tres splits.

    gpu_augment=True (default) NO aplica la augmentation aca: la aplica el modelo, con lo
    que corre en GPU. Aplicarla en el pipeline de tf.data la deja en CPU y en instancias
    con pocos vCPU la GPU se queda esperando datos (se midio GPU-Util entre 0 y 30%).
    """
    data_dir = Path(data_dir)
    common = dict(label_mode="int", image_size=(img_size, img_size), batch_size=batch_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", shuffle=True, seed=42, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir / "val", shuffle=False, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(data_dir / "test", shuffle=False, **common)

    class_names = list(train_ds.class_names)
    n_classes = len(class_names)

    if not gpu_augment:
        augmentation = build_augmentation(aug)
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE
        )

    if mixup > 0:
        # MixUp: cada batch se mezcla con si mismo invertido; las etiquetas pasan a ser
        # blandas (one-hot mezclado), por eso todos los splits se convierten a one-hot y
        # el motor entrena con CategoricalCrossentropy. Mismo algoritmo que la version PyTorch.
        def to_onehot(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return x, tf.one_hot(y, n_classes)

        def apply_mixup(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            # Beta(alpha, alpha) muestreada via dos Gamma(alpha)
            g1 = tf.random.gamma([], mixup)
            g2 = tf.random.gamma([], mixup)
            lam = g1 / (g1 + g2)
            x2 = tf.reverse(x, axis=[0])
            y2 = tf.reverse(y, axis=[0])
            return lam * x + (1.0 - lam) * x2, lam * y + (1.0 - lam) * y2

        train_ds = train_ds.map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE).map(
            apply_mixup, num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)

    # el presupuesto de RAM por defecto del autotuner es menor que un lote nuestro
    # (128 imgs de 160px ~= 64MB), asi que no llega a precargar y avisa por consola.
    # Con 2GB puede mantener varios lotes adelantados sin comprometer la memoria.
    opciones = tf.data.Options()
    opciones.autotune.ram_budget = 2 * 1024 ** 3

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune).with_options(opciones),
        val_ds.prefetch(autotune).with_options(opciones),
        test_ds.prefetch(autotune).with_options(opciones),
        class_names,
    )
