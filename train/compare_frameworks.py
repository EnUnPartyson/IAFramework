"""Consolida las metricas de los 6 entrenamientos (3 modelos x 2 frameworks) en una tabla
comparativa. Solo usa stdlib: puede correrse desde cualquiera de los dos venvs.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "metrics"
OUT_PATH = METRICS_DIR / "comparacion_tf_vs_pytorch.json"

TASKS = ("detector", "dog_breed", "cat_breed")
FRAMEWORKS = ("pytorch", "tensorflow")


def summarize(report: dict) -> dict:
    return {
        "test_accuracy": report["test"]["accuracy"],
        "test_f1_macro": report["test"]["f1_macro"],
        "tiempo_entrenamiento_seg": report["tiempo_entrenamiento_seg"],
        "epochs_corridas": report["hiperparametros"]["epochs_corridas"],
        "tamano_pesos_mb": report["tamano_pesos_mb"],
    }


def main() -> None:
    comparison: dict[str, dict] = {}
    missing: list[str] = []

    for task in TASKS:
        comparison[task] = {}
        for fw in FRAMEWORKS:
            metrics_path = METRICS_DIR / f"{task}_{fw}_metrics.json"
            if not metrics_path.exists():
                missing.append(f"{task}/{fw}")
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                comparison[task][fw] = summarize(json.load(f))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"Comparacion guardada en {OUT_PATH}\n")
    header = f"{'modelo':<12} {'framework':<11} {'test_acc':>9} {'f1_macro':>9} {'tiempo_s':>9} {'epocas':>7} {'pesos_MB':>9}"
    print(header)
    print("-" * len(header))
    for task in TASKS:
        for fw in FRAMEWORKS:
            row = comparison[task].get(fw)
            if row is None:
                print(f"{task:<12} {fw:<11} {'(sin metricas)':>9}")
                continue
            print(
                f"{task:<12} {fw:<11} {row['test_accuracy']:>9.4f} {row['test_f1_macro']:>9.4f} "
                f"{row['tiempo_entrenamiento_seg']:>9.1f} {row['epochs_corridas']:>7} {row['tamano_pesos_mb']:>9.2f}"
            )

    if missing:
        print(f"\nFaltan metricas de: {', '.join(missing)}")


if __name__ == "__main__":
    main()
