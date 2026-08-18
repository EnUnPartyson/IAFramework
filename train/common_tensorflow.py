"""Motor de entrenamiento TensorFlow/Keras compartido por los 3 modelos.

Espejo de common_pytorch.py: mismos hiperparametros por defecto, misma seleccion de modelo
(val_accuracy), mismo early stopping y scheduler (ReduceLROnPlateau), mismo esquema de metricas.
Nunca importar torch aca.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.model_defs_tensorflow import HEAD_FLATTEN, HEAD_GAP, build_simple_cnn, softmax  # noqa: E402
from utils.report_common import (  # noqa: E402
    TASK_DEFAULTS,
    class_weight_values,
    file_size_mb,
    forced_mode_metrics,
    metrics_from_predictions,
    open_set_analysis,
    resolve_hparams,
    save_confusion_matrix_plot,
    save_metrics_json,
    save_open_set_plot,
    save_training_curves_plot,
)
from utils.transforms_tensorflow import (  # noqa: E402
    AUG_BASE,
    AUG_STRONG,
    build_augmentation,
    count_train_images_per_class,
    make_datasets,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
# paciencia 4 y no 2: con val_accuracy ruidosa (normal en datasets chicos con MixUp), una
# paciencia corta interpreta el ruido como estancamiento y desploma el LR en pocas epocas
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.5


def build_arg_parser(task: str) -> argparse.ArgumentParser:
    # la receta por tarea (cabezal, augmentation, mixup, resolucion, profundidad, epocas)
    # vive en TASK_DEFAULTS, compartida con el motor PyTorch y con Optuna para que nunca diverjan
    d = TASK_DEFAULTS[task]
    parser = argparse.ArgumentParser(description=f"Entrena el modelo '{task}' en TensorFlow/Keras")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data" / "processed" / task)
    parser.add_argument("--epochs", type=int, default=d["epochs"])
    parser.add_argument(
        "--aug",
        choices=(AUG_BASE, AUG_STRONG),
        default=d["aug"],
        help="augmentation: 'base' o 'strong' (RandAugment + RandomErasing, para datasets chicos)",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=d["mixup"],
        help="alpha de MixUp (0 = apagado); mezcla pares de imagenes y etiquetas en el batch",
    )
    parser.add_argument(
        "--head",
        choices=(HEAD_FLATTEN, HEAD_GAP),
        default=d["head"],
        help="cabezal: 'flatten' (mas capacidad, necesita muchos datos) o 'gap' (60x menos parametros)",
    )
    parser.add_argument("--img-size", type=int, default=d["img_size"], help="resolucion de entrenamiento/eval")
    parser.add_argument("--blocks", type=int, default=d["blocks"], help="cantidad de bloques convolucionales (3-5)")
    # default None a proposito: distingue "no lo paso" de "lo paso igual al default",
    # necesario para que --hparams-from no pise un valor explicito (ver resolve_hparams)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--hparams-from",
        type=Path,
        default=None,
        help="JSON con los mejores hiperparametros (los mismos que la version PyTorch, para comparar parejo)",
    )
    parser.add_argument(
        "--patience", type=int, default=d["patience"], help="early stopping: epocas sin mejorar val_accuracy"
    )
    parser.add_argument("--model-out", type=Path, default=ROOT_DIR / "models" / f"{task}_tensorflow.keras")
    parser.add_argument("--metrics-out", type=Path, default=ROOT_DIR / "metrics" / f"{task}_tensorflow_metrics.json")
    return parser


def _labels_from_dataset(dataset: tf.data.Dataset) -> list[int]:
    labels = np.concatenate([lbl.numpy() for _, lbl in dataset])
    if labels.ndim == 2:  # one-hot (cuando mixup esta activo los splits van en one-hot)
        labels = labels.argmax(axis=1)
    return labels.tolist()


def train_model(task: str, args: argparse.Namespace, expected_classes: tuple[str, ...] | None = None) -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[{task}/tensorflow] GPUs visibles: {len(gpus)}")

    hp = resolve_hparams(args, args.hparams_from)
    print(f"[{task}/tensorflow] hiperparametros: {hp}")

    train_ds, val_ds, test_ds, class_names = make_datasets(
        args.data_dir, hp["batch_size"], img_size=args.img_size, aug=args.aug, mixup=args.mixup
    )
    if expected_classes is not None and tuple(class_names) != tuple(expected_classes):
        raise ValueError(f"Clases en {args.data_dir} ({class_names}) != esperadas ({expected_classes})")

    _, class_counts = count_train_images_per_class(args.data_dir)
    print(f"[{task}/tensorflow] clases ({len(class_names)}): {dict(zip(class_names, class_counts))}")

    if args.mixup > 0:
        # con etiquetas blandas (mixup) Keras no soporta class_weight; las razas estan
        # aproximadamente balanceadas asi que no se pierde nada relevante
        class_weight = None
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        class_weight = {i: w for i, w in enumerate(class_weight_values(class_counts))}
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # la augmentation va DENTRO del modelo para que corra en GPU: en el pipeline de tf.data
    # queda en CPU y con pocos vCPU la GPU se queda esperando datos (ver DECISIONS.md)
    model = build_simple_cnn(
        num_classes=len(class_names),
        dropout=hp["dropout"],
        img_size=args.img_size,
        head=args.head,
        blocks=args.blocks,
        augmentation=build_augmentation(args.aug),
    )
    print(
        f"[{task}/tensorflow] cabezal={args.head}, {args.blocks} bloques, {args.img_size}px, "
        f"{model.count_params():,} parametros (augmentation en GPU)"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["lr"], weight_decay=hp["weight_decay"]),
        loss=loss,
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
        ),
    ]

    start_time = time.time()
    fit_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    training_time = time.time() - start_time

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)

    # el .keras (y save_weights sobre un modelo compilado) incluye el estado del optimizador Adam
    # (~3x el tamano de los pesos); para que "tamano_pesos_mb" sea comparable con el state_dict de
    # PyTorch se copian los pesos a un modelo fresco sin compilar y se mide ese archivo
    export_model = build_simple_cnn(
        num_classes=len(class_names),
        dropout=hp["dropout"],
        img_size=args.img_size,
        head=args.head,
        blocks=args.blocks,
    )
    export_model.set_weights(model.get_weights())
    weights_tmp = args.model_out.parent / (args.model_out.stem + ".weights.h5")
    export_model.save_weights(weights_tmp)
    weights_only_mb = file_size_mb(weights_tmp)
    weights_tmp.unlink()

    logits = model.predict(test_ds, verbose=0)
    preds = logits.argmax(axis=1).tolist()
    labels = _labels_from_dataset(test_ds)
    test_metrics = metrics_from_predictions(labels, preds, class_names)

    # modo forzado del detector (requisito del profesor): perro vs gato ignorando "ninguno"
    forced_metrics = None
    if "ninguno" in class_names:
        forced_metrics = forced_mode_metrics(logits, labels, class_names, "ninguno")
        print(f"[{task}/tensorflow] modo forzado (perro vs gato): accuracy={forced_metrics['accuracy']:.4f}")

    # modo "raza no identificada": umbral de confianza calibrado en validacion, evaluado
    # contra razas nunca vistas si prepare_data.py dejo la carpeta unknown/
    val_logits = model.predict(val_ds, verbose=0)
    val_probs = softmax(val_logits)
    val_labels = _labels_from_dataset(val_ds)
    val_correct = [p == t for p, t in zip(val_probs.argmax(axis=1).tolist(), val_labels)]
    unknown_dir = args.data_dir / "unknown"
    unknown_maxprob: list[float] | None = None
    if unknown_dir.exists():
        unknown_ds = tf.keras.utils.image_dataset_from_directory(
            unknown_dir,
            label_mode="int",
            image_size=(args.img_size, args.img_size),
            batch_size=hp["batch_size"],
            shuffle=False,
        )
        unknown_maxprob = softmax(model.predict(unknown_ds, verbose=0)).max(axis=1).tolist()
    open_set = open_set_analysis(val_probs.max(axis=1).tolist(), val_correct, unknown_maxprob)
    print(f"[{task}/tensorflow] raza no identificada: {open_set['en_umbral_sugerido']}")

    epochs_run = len(fit_history.history["loss"])
    history = [
        {
            "epoch": i + 1,
            "train_loss": float(fit_history.history["loss"][i]),
            "train_acc": float(fit_history.history["accuracy"][i]),
            "val_loss": float(fit_history.history["val_loss"][i]),
            "val_acc": float(fit_history.history["val_accuracy"][i]),
            "lr": float(fit_history.history["learning_rate"][i])
            if "learning_rate" in fit_history.history
            else None,
        }
        for i in range(epochs_run)
    ]

    report = {
        "framework": "tensorflow",
        "tarea": task,
        "hiperparametros": {
            "epochs_max": args.epochs,
            "epochs_corridas": epochs_run,
            "batch_size": hp["batch_size"],
            "lr_inicial": hp["lr"],
            "dropout": hp["dropout"],
            "weight_decay": hp["weight_decay"],
            "hparams_origen": str(args.hparams_from) if args.hparams_from else "defaults/CLI",
            "early_stopping_paciencia": args.patience,
            "scheduler": f"ReduceLROnPlateau(factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})",
            "head": args.head,
            "blocks": args.blocks,
            "img_size": args.img_size,
            "n_parametros": int(model.count_params()),
            "aug": args.aug,
            "mixup_alpha": args.mixup,
        },
        "distribucion_train": dict(zip(class_names, class_counts)),
        "tiempo_entrenamiento_seg": round(training_time, 1),
        "mejor_val_accuracy": float(max(fit_history.history["val_accuracy"])),
        "tamano_pesos_mb": weights_only_mb,
        "tamano_archivo_modelo_mb": file_size_mb(args.model_out),
        "historia": history,
        "test": test_metrics,
        "raza_no_identificada": open_set,
    }
    if forced_metrics is not None:
        report["test_modo_forzado"] = forced_metrics
    save_metrics_json(report, args.metrics_out)

    plots_dir = args.metrics_out.parent
    save_confusion_matrix_plot(
        test_metrics["matriz_confusion"], class_names, plots_dir / f"{task}_tensorflow_confusion_matrix.png"
    )
    save_confusion_matrix_plot(
        test_metrics["matriz_confusion"],
        class_names,
        plots_dir / f"{task}_tensorflow_confusion_matrix_norm.png",
        normalize=True,
    )
    save_training_curves_plot(history, plots_dir / f"{task}_tensorflow_training_curves.png", f"{task} - TensorFlow")
    save_open_set_plot(open_set, plots_dir / f"{task}_tensorflow_umbral_desconocidas.png", f"{task} - TensorFlow")

    print(f"[{task}/tensorflow] modelo: {args.model_out}")
    print(f"[{task}/tensorflow] metricas: {args.metrics_out}")
    print(
        f"[{task}/tensorflow] test accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}"
    )
    return report
