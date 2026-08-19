"""Compara la version base (v1-presentacion, desde cero) contra el modo pro (transfer
learning) por tarea. No importa torch ni tensorflow.

Ojo con la lectura: no es una comparacion 1 a 1. El pro clasifica MAS razas (30 perros vs 20,
25 gatos vs 12), asi que igualar el accuracy base ya seria ganancia; superarlo con mas clases
es la victoria doble. La tabla muestra n_clases justamente para eso.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "metrics"
OUT_PATH = METRICS_DIR / "comparacion_base_vs_pro.json"

TASKS = ("detector", "dog_breed", "cat_breed")


def fila(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)
    hp = r["hiperparametros"]
    out = {
        "test_accuracy": r["test"]["accuracy"],
        "test_f1_macro": r["test"]["f1_macro"],
        "n_clases": len(r["distribucion_train"]),
        "n_parametros": hp.get("n_parametros"),
        "arch": hp.get("arch", "SimpleCNN"),
        "img_size": hp.get("img_size"),
        "tiempo_entrenamiento_seg": r["tiempo_entrenamiento_seg"],
    }
    if r.get("roc_pr"):
        out["roc_auc_macro"] = r["roc_pr"]["roc_auc_macro"]
    if "test_modo_forzado" in r:
        out["modo_forzado_accuracy"] = r["test_modo_forzado"]["accuracy"]
    return out


def main() -> None:
    comparacion: dict[str, dict] = {}
    for task in TASKS:
        comparacion[task] = {
            "base": fila(METRICS_DIR / f"{task}_pytorch_metrics.json"),
            "pro": fila(METRICS_DIR / f"{task}_pro_metrics.json"),
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparacion, f, indent=2, ensure_ascii=False)
    print(f"Comparacion guardada en {OUT_PATH}\n")

    header = f"{'modelo':<12} {'version':<8} {'arch':<18} {'clases':>6} {'acc':>8} {'f1':>8} {'roc_auc':>8} {'px':>5}"
    print(header)
    print("-" * len(header))
    for task in TASKS:
        for version in ("base", "pro"):
            r = comparacion[task][version]
            if r is None:
                print(f"{task:<12} {version:<8} {'(sin metricas)'}")
                continue
            roc = f"{r['roc_auc_macro']:.4f}" if "roc_auc_macro" in r else "-"
            print(
                f"{task:<12} {version:<8} {r['arch']:<18} {r['n_clases']:>6} "
                f"{r['test_accuracy']:>8.4f} {r['test_f1_macro']:>8.4f} {roc:>8} {r['img_size']:>5}"
            )


if __name__ == "__main__":
    main()
