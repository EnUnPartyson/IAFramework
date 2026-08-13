"""Evaluacion y reportes compartidos entre los pipelines PyTorch y TensorFlow.

Este modulo NO debe importar torch ni tensorflow: se usa desde ambos venvs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # EC2 no tiene display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402


def metrics_from_predictions(labels: list[int], preds: list[int], class_names: list[str]) -> dict:
    labels_range = list(range(len(class_names)))
    f1_per_class = f1_score(labels, preds, average=None, labels=labels_range, zero_division=0)
    accuracy = float(sum(p == t for p, t in zip(preds, labels)) / len(labels))
    cm = confusion_matrix(labels, preds, labels=labels_range)
    return {
        "accuracy": accuracy,
        "f1_macro": float(f1_per_class.mean()),
        "f1_por_clase": {name: float(score) for name, score in zip(class_names, f1_per_class)},
        "matriz_confusion": cm.tolist(),
        "clases": list(class_names),
    }


def class_weight_values(counts: list[int]) -> list[float]:
    # frecuencia inversa normalizada: mitigacion del desbalance (ver DECISIONS.md)
    total = sum(counts)
    n = len(counts)
    return [total / (n * c) for c in counts]


def save_confusion_matrix_plot(cm: list[list[int]], class_names: list[str], out_path: Path) -> None:
    cm_arr = np.array(cm)
    size = max(5, len(class_names) * 0.6)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    if len(class_names) <= 20:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, cm_arr[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_metrics_json(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


# hiperparametros que Optuna puede buscar y que ambos frameworks aceptan por igual
TUNABLE_HPARAMS = ("lr", "batch_size", "dropout", "weight_decay")
HPARAM_DEFAULTS = {"lr": 1e-3, "batch_size": 64, "dropout": 0.4, "weight_decay": 1e-4}


def resolve_hparams(args, hparams_from: Path | None) -> dict:
    """Precedencia: valor explicito en la linea de comandos > JSON de Optuna > default.

    Los argumentos tuneables llegan como None si el usuario no los paso, para poder
    distinguir "no lo especifico" de "lo puso igual al default".
    """
    from_json: dict = {}
    if hparams_from is not None:
        if not hparams_from.exists():
            raise FileNotFoundError(
                f"No existe {hparams_from}. Correr primero la busqueda con tune_detector_pytorch.py"
            )
        with open(hparams_from, "r", encoding="utf-8") as f:
            from_json = json.load(f).get("best_params", {})
        print(f"Hiperparametros tomados de {hparams_from}: {from_json}")

    resolved = {}
    for name in TUNABLE_HPARAMS:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            resolved[name] = cli_value
        elif name in from_json:
            resolved[name] = from_json[name]
        else:
            resolved[name] = HPARAM_DEFAULTS[name]
    return resolved
