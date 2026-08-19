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
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def metrics_from_predictions(labels: list[int], preds: list[int], class_names: list[str]) -> dict:
    labels_range = list(range(len(class_names)))
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, labels=labels_range, zero_division=0
    )
    accuracy = float(sum(p == t for p, t in zip(preds, labels)) / len(labels))
    cm = confusion_matrix(labels, preds, labels=labels_range)
    return {
        "accuracy": accuracy,
        "f1_macro": float(f1_per_class.mean()),
        "f1_por_clase": {name: float(score) for name, score in zip(class_names, f1_per_class)},
        "reporte_por_clase": {
            name: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "n_imagenes_test": int(s),
            }
            for name, p, r, f, s in zip(class_names, precision, recall, f1_per_class, support)
        },
        "confusiones_mas_frecuentes": top_confusions(cm, class_names),
        "matriz_confusion": cm.tolist(),
        "clases": list(class_names),
    }


def top_confusions(cm: np.ndarray, class_names: list[str], k: int = 10) -> list[dict]:
    """Pares (real, predicho) mas confundidos fuera de la diagonal, como % de la clase real."""
    cm = np.asarray(cm)
    rows = []
    totals = cm.sum(axis=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0 and totals[i] > 0:
                rows.append({
                    "real": class_names[i],
                    "predicho": class_names[j],
                    "cantidad": int(cm[i, j]),
                    "porcentaje_de_la_clase": round(float(cm[i, j] / totals[i]), 4),
                })
    rows.sort(key=lambda r: r["cantidad"], reverse=True)
    return rows[:k]


def forced_mode_metrics(logits: np.ndarray, labels: list[int], class_names: list[str], none_class: str) -> dict:
    """Modo forzado del detector: ignora la clase "ninguno" y clasifica solo perro vs gato.

    Se evalua sobre las imagenes de test cuya etiqueta real NO es "ninguno" (requisito del
    profesor: el modo forzado siempre responde una especie).
    """
    none_idx = class_names.index(none_class)
    masked = np.asarray(logits, dtype=float).copy()
    masked[:, none_idx] = -np.inf
    forced_preds = masked.argmax(axis=1)

    keep = [i for i, lbl in enumerate(labels) if lbl != none_idx]
    kept_names = [n for n in class_names if n != none_class]
    remap = {old: new for new, old in enumerate(i for i in range(len(class_names)) if i != none_idx)}
    sub_preds = [remap[int(forced_preds[i])] for i in keep]
    sub_labels = [remap[labels[i]] for i in keep]
    return metrics_from_predictions(sub_labels, sub_preds, kept_names)


def class_weight_values(counts: list[int]) -> list[float]:
    # frecuencia inversa normalizada: mitigacion del desbalance (ver DECISIONS.md)
    total = sum(counts)
    n = len(counts)
    return [total / (n * c) for c in counts]


def save_confusion_matrix_plot(
    cm: list[list[int]], class_names: list[str], out_path: Path, normalize: bool = False
) -> None:
    cm_arr = np.array(cm, dtype=float)
    if normalize:
        totals = cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.divide(cm_arr, totals, out=np.zeros_like(cm_arr), where=totals > 0)
    size = max(5, len(class_names) * 0.6)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(cm_arr, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion" + (" (normalizada por clase real)" if normalize else " (conteos)"))
    if len(class_names) <= 25:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = f"{cm_arr[i, j]:.2f}" if normalize else f"{int(cm_arr[i, j])}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6 if len(class_names) > 12 else 8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_training_curves_plot(history: list[dict], out_path: Path, title: str) -> None:
    """Loss y accuracy por epoca (train vs val) + learning rate: diagnostico de un vistazo."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set_xlabel("Epoca")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")
    lrs = [h.get("lr") for h in history]
    if all(lr is not None for lr in lrs):
        ax_lr = ax_loss.twinx()
        ax_lr.plot(epochs, lrs, color="gray", linestyle=":", alpha=0.7)
        ax_lr.set_ylabel("LR", color="gray")
        ax_lr.set_yscale("log")

    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax_acc.set_xlabel("Epoca")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(loc="lower right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_open_set_plot(open_set: dict, out_path: Path, title: str) -> None:
    """Curva del modo "raza no identificada": que se gana y se pierde al mover el umbral."""
    curva = open_set.get("curva", [])
    if not curva:
        return
    thresholds = [r["umbral"] for r in curva]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, [r["cobertura_conocidas"] for r in curva], marker="o", label="cobertura razas conocidas")
    ax.plot(thresholds, [r["accuracy_aceptadas"] for r in curva], marker="s", label="accuracy de lo aceptado")
    if "rechazo_desconocidas" in curva[0]:
        ax.plot(thresholds, [r["rechazo_desconocidas"] for r in curva], marker="^", label="rechazo razas desconocidas")
    ax.axvline(open_set["umbral_sugerido"], color="gray", linestyle="--", alpha=0.8, label="umbral sugerido")
    ax.set_xlabel("Umbral de confianza")
    ax.set_ylabel("Proporcion")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_metrics_json(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convierte logits en probabilidades. Se resta el maximo por estabilidad numerica:
    sin eso, exp() de un logit grande desborda."""
    logits = np.asarray(logits, dtype=float)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


# las curvas se submuestrean antes de guardarlas: con 20 clases, los puntos crudos harian
# un JSON de varios MB sin aportar resolucion visible al graficarlas
PUNTOS_CURVA = 100


def _submuestrear(x: np.ndarray, y: np.ndarray, puntos: int = PUNTOS_CURVA) -> list[list[float]]:
    idx = np.arange(len(x)) if len(x) <= puntos else np.linspace(0, len(x) - 1, puntos).astype(int)
    return [[round(float(x[i]), 5), round(float(y[i]), 5)] for i in idx]


def roc_pr_analysis(labels: list[int], logits: np.ndarray, class_names: list[str]) -> dict:
    """Curvas ROC y precision-recall sobre test, una-contra-el-resto por clase.

    A diferencia de accuracy o F1, que miden en UN punto de operacion, estas curvas barren
    todos los umbrales posibles. Por eso hacen falta las probabilidades y no alcanza con las
    predicciones finales.
    """
    probs = softmax(logits)
    n = len(class_names)
    binarias = np.eye(n, dtype=int)[np.asarray(labels)]

    por_clase: dict[str, dict] = {}
    for i, nombre in enumerate(class_names):
        presentes = int(binarias[:, i].sum())
        # sin ejemplos de ambos lados la curva no esta definida
        if presentes == 0 or presentes == len(labels):
            continue
        fpr, tpr, _ = roc_curve(binarias[:, i], probs[:, i])
        prec, rec, _ = precision_recall_curve(binarias[:, i], probs[:, i])
        por_clase[nombre] = {
            "roc_auc": round(float(auc(fpr, tpr)), 4),
            "pr_auc": round(float(average_precision_score(binarias[:, i], probs[:, i])), 4),
            "soporte": presentes,
            "curva_roc": _submuestrear(fpr, tpr),
            "curva_pr": _submuestrear(rec, prec),
        }

    if not por_clase:
        return {}

    resultado = {
        "roc_auc_macro": round(float(np.mean([v["roc_auc"] for v in por_clase.values()])), 4),
        "pr_auc_macro": round(float(np.mean([v["pr_auc"] for v in por_clase.values()])), 4),
        "por_clase": por_clase,
    }
    try:
        resultado["roc_auc_ponderado"] = round(
            float(roc_auc_score(binarias, probs, average="weighted")), 4
        )
    except ValueError:
        pass  # alguna clase sin ejemplos en test
    return resultado


def save_roc_pr_plot(analisis: dict, out_path: Path, title: str) -> None:
    """Dibuja ROC y PR lado a lado: una linea fina por clase mas el promedio destacado."""
    if not analisis or not analisis.get("por_clase"):
        return
    clases = analisis["por_clase"]
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))

    for datos in clases.values():
        xs, ys = zip(*datos["curva_roc"])
        ax_roc.plot(xs, ys, linewidth=0.9, alpha=0.45)
        xs, ys = zip(*datos["curva_pr"])
        ax_pr.plot(xs, ys, linewidth=0.9, alpha=0.45)

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="azar")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Recall (tasa de verdaderos positivos)")
    ax_roc.set_title(f"ROC · AUC macro = {analisis['roc_auc_macro']:.3f}")
    ax_roc.legend(loc="lower right", fontsize=8)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall · AUC macro = {analisis['pr_auc_macro']:.3f}")

    for ax in (ax_roc, ax_pr):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{title} · una linea por clase ({len(clases)} clases)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def open_set_analysis(
    val_maxprob: list[float],
    val_correct: list[bool],
    unknown_maxprob: list[float] | None = None,
) -> dict:
    """Modo "raza no identificada": elige el umbral de confianza sobre validacion.

    Si la probabilidad maxima del softmax queda debajo del umbral, la prediccion se
    reporta como "no identificada". El umbral sugerido maximiza (accuracy de lo aceptado
    x cobertura) + tasa de rechazo de razas nunca vistas, cuando hay muestra de estas.
    """
    rows = []
    for t in [x / 100 for x in range(30, 96, 5)]:
        accepted = [p >= t for p in val_maxprob]
        n_accepted = sum(accepted)
        coverage = n_accepted / len(val_maxprob)
        acc_accepted = (
            sum(c for c, a in zip(val_correct, accepted) if a) / n_accepted if n_accepted else 0.0
        )
        row = {"umbral": t, "cobertura_conocidas": round(coverage, 4), "accuracy_aceptadas": round(acc_accepted, 4)}
        if unknown_maxprob:
            row["rechazo_desconocidas"] = round(sum(p < t for p in unknown_maxprob) / len(unknown_maxprob), 4)
        rows.append(row)

    def score(row: dict) -> float:
        return row["accuracy_aceptadas"] * row["cobertura_conocidas"] + row.get("rechazo_desconocidas", 0.0)

    best = max(rows, key=score)
    return {"umbral_sugerido": best["umbral"], "en_umbral_sugerido": best, "curva": rows}


# Receta por tarea, compartida por ambos frameworks Y por la busqueda de Optuna, para que
# nunca diverjan: el detector tiene datos de sobra (cabezal grande, augmentation suave);
# las razas son fine-grained con pocos datos (mas resolucion, red mas profunda, cabezal GAP,
# augmentation fuerte y MixUp).
TASK_DEFAULTS = {
    "detector": {
        "head": "flatten", "aug": "base", "mixup": 0.0,
        "img_size": 128, "blocks": 4, "epochs": 30, "patience": 6,
    },
    "dog_breed": {
        "head": "gap", "aug": "strong", "mixup": 0.2,
        "img_size": 160, "blocks": 5, "epochs": 80, "patience": 12,
    },
    "cat_breed": {
        "head": "gap", "aug": "strong", "mixup": 0.2,
        "img_size": 160, "blocks": 5, "epochs": 80, "patience": 12,
    },
}

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
