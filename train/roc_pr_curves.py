"""Calcula curvas ROC y precision-recall sobre el conjunto de test.

El entrenamiento guarda solo las metricas en el punto de operacion (accuracy, F1, matriz
de confusion). ROC y PR necesitan las PROBABILIDADES de cada clase para poder barrer el
umbral, asi que hay que volver a pasar el test set por el modelo.

Con varias clases se usa el enfoque uno-contra-el-resto: para cada clase se toma su
probabilidad como score y se la compara contra "es esa clase o no".

    venv-torch/bin/python train/roc_pr_curves.py                     # todos, PyTorch
    venv-torch/bin/python train/roc_pr_curves.py --tareas detector   # una sola
    venv-torch/bin/python train/roc_pr_curves.py --framework tensorflow
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

sys.path.append(str(Path(__file__).resolve().parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "metrics"
# la curva se guarda submuestreada: 200 puntos bastan para dibujarla y evitan un JSON enorme
PUNTOS_CURVA = 200


def _submuestrear(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    if len(x) <= PUNTOS_CURVA:
        idx = np.arange(len(x))
    else:
        idx = np.linspace(0, len(x) - 1, PUNTOS_CURVA).astype(int)
    return [[round(float(x[i]), 5), round(float(y[i]), 5)] for i in idx]


def probabilidades_pytorch(task: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    from utils.model_defs_pytorch import SimpleCNN
    from utils.transforms_pytorch import get_eval_transforms

    ckpt = torch.load(ROOT_DIR / "models" / f"{task}_pytorch.pt", map_location="cpu", weights_only=False)
    clases, img_size = list(ckpt["class_names"]), int(ckpt.get("img_size", 128))
    modelo = SimpleCNN(
        num_classes=len(clases), dropout=float(ckpt.get("dropout", 0.4)),
        head=ckpt.get("head", "flatten"), blocks=int(ckpt.get("blocks", 4)), img_size=img_size,
    )
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()

    ds = ImageFolder(ROOT_DIR / "data" / "processed" / task / "test", transform=get_eval_transforms(img_size=img_size))
    if list(ds.classes) != clases:
        raise ValueError(f"Las clases del test {ds.classes} no coinciden con las del modelo {clases}")

    probs, etiquetas = [], []
    with torch.no_grad():
        for imgs, lbls in DataLoader(ds, batch_size=64, num_workers=0):
            probs.append(torch.softmax(modelo(imgs), dim=1).numpy())
            etiquetas.extend(lbls.tolist())
    return np.concatenate(probs), np.array(etiquetas), clases


def probabilidades_tensorflow(task: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import tensorflow as tf

    with open(METRICS_DIR / f"{task}_tensorflow_metrics.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    clases = list(meta["test"]["clases"])
    img_size = int(meta["hiperparametros"].get("img_size", 128))

    modelo = tf.keras.models.load_model(ROOT_DIR / "models" / f"{task}_tensorflow.keras", compile=False)
    ds = tf.keras.utils.image_dataset_from_directory(
        ROOT_DIR / "data" / "processed" / task / "test",
        label_mode="int", image_size=(img_size, img_size), batch_size=64, shuffle=False,
    )
    if list(ds.class_names) != clases:
        raise ValueError(f"Las clases del test {ds.class_names} no coinciden con las del modelo {clases}")

    etiquetas = np.concatenate([l.numpy() for _, l in ds])
    logits = modelo.predict(ds, verbose=0)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True), etiquetas, clases


def calcular(task: str, framework: str) -> dict:
    obtener = probabilidades_pytorch if framework == "pytorch" else probabilidades_tensorflow
    probs, etiquetas, clases = obtener(task)
    n = len(clases)
    # uno-contra-el-resto: una fila por clase indicando si el ejemplo pertenece a ella
    binarias = np.eye(n, dtype=int)[etiquetas]

    por_clase = {}
    for i, nombre in enumerate(clases):
        presentes = int(binarias[:, i].sum())
        if presentes == 0 or presentes == len(etiquetas):
            continue  # sin ambas clases la curva no esta definida
        fpr, tpr, _ = roc_curve(binarias[:, i], probs[:, i])
        prec, rec, _ = precision_recall_curve(binarias[:, i], probs[:, i])
        por_clase[nombre] = {
            "roc_auc": round(float(auc(fpr, tpr)), 4),
            "pr_auc": round(float(average_precision_score(binarias[:, i], probs[:, i])), 4),
            "soporte": presentes,
            "curva_roc": _submuestrear(fpr, tpr),
            "curva_pr": _submuestrear(rec, prec),
        }

    aucs = [v["roc_auc"] for v in por_clase.values()]
    praucs = [v["pr_auc"] for v in por_clase.values()]
    return {
        "tarea": task,
        "framework": framework,
        "n_test": int(len(etiquetas)),
        "roc_auc_macro": round(float(np.mean(aucs)), 4),
        "pr_auc_macro": round(float(np.mean(praucs)), 4),
        "roc_auc_ponderado": round(float(roc_auc_score(binarias, probs, average="weighted")), 4),
        "por_clase": por_clase,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Curvas ROC y PR sobre el test set")
    parser.add_argument("--tareas", nargs="+", default=["detector", "dog_breed", "cat_breed"])
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), default="pytorch")
    args = parser.parse_args()

    for task in args.tareas:
        try:
            r = calcular(task, args.framework)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{task}/{args.framework}] omitido: {exc}")
            continue
        salida = METRICS_DIR / f"{task}_{args.framework}_roc_pr.json"
        with open(salida, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(
            f"[{task}/{args.framework}] {r['n_test']} imgs · "
            f"ROC-AUC macro={r['roc_auc_macro']:.4f} · PR-AUC macro={r['pr_auc_macro']:.4f} -> {salida.name}"
        )


if __name__ == "__main__":
    main()
