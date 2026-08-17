"""Consolida las metricas de los 6 entrenamientos (3 modelos x 2 frameworks) en una tabla
comparativa (JSON + PNG). No importa torch ni tensorflow: corre desde cualquiera de los dos venvs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "metrics"
OUT_PATH = METRICS_DIR / "comparacion_tf_vs_pytorch.json"
PLOT_PATH = METRICS_DIR / "comparacion_tf_vs_pytorch.png"

TASKS = ("detector", "dog_breed", "cat_breed")
FRAMEWORKS = ("pytorch", "tensorflow")


def summarize(report: dict) -> dict:
    row = {
        "test_accuracy": report["test"]["accuracy"],
        "test_f1_macro": report["test"]["f1_macro"],
        "tiempo_entrenamiento_seg": report["tiempo_entrenamiento_seg"],
        "epochs_corridas": report["hiperparametros"]["epochs_corridas"],
        "tamano_pesos_mb": report["tamano_pesos_mb"],
        "n_parametros": report["hiperparametros"].get("n_parametros"),
    }
    if "test_modo_forzado" in report:
        row["modo_forzado_accuracy"] = report["test_modo_forzado"]["accuracy"]
    if "raza_no_identificada" in report:
        row["umbral_no_identificada"] = report["raza_no_identificada"]["umbral_sugerido"]
    return row


def save_comparison_plot(comparison: dict[str, dict]) -> None:
    panels = [
        ("test_accuracy", "Accuracy en test", None),
        ("test_f1_macro", "F1 macro en test", None),
        ("tiempo_entrenamiento_seg", "Tiempo de entrenamiento (s)", "log"),
    ]
    x = np.arange(len(TASKS))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (key, title, yscale) in zip(axes, panels):
        for i, fw in enumerate(FRAMEWORKS):
            values = [comparison[t].get(fw, {}).get(key, 0) or 0 for t in TASKS]
            bars = ax.bar(x + (i - 0.5) * width, values, width, label=fw)
            if yscale != "log":
                ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS)
        ax.set_title(title)
        if yscale:
            ax.set_yscale(yscale)
        if key.startswith("test_"):
            ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("Comparacion TF vs PyTorch (mismos datos, arquitectura e hiperparametros)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)


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
    save_comparison_plot(comparison)

    print(f"Comparacion guardada en {OUT_PATH}")
    print(f"Grafico comparativo en {PLOT_PATH}\n")
    header = (
        f"{'modelo':<12} {'framework':<11} {'test_acc':>9} {'f1_macro':>9} {'forzado':>8} "
        f"{'tiempo_s':>9} {'epocas':>7} {'pesos_MB':>9}"
    )
    print(header)
    print("-" * len(header))
    for task in TASKS:
        for fw in FRAMEWORKS:
            row = comparison[task].get(fw)
            if row is None:
                print(f"{task:<12} {fw:<11} {'(sin metricas)':>9}")
                continue
            forced = f"{row['modo_forzado_accuracy']:.4f}" if "modo_forzado_accuracy" in row else "-"
            print(
                f"{task:<12} {fw:<11} {row['test_accuracy']:>9.4f} {row['test_f1_macro']:>9.4f} {forced:>8} "
                f"{row['tiempo_entrenamiento_seg']:>9.1f} {row['epochs_corridas']:>7} {row['tamano_pesos_mb']:>9.2f}"
            )

    if missing:
        print(f"\nFaltan metricas de: {', '.join(missing)}")


if __name__ == "__main__":
    main()
