"""Busqueda de hiperparametros con Optuna para cualquiera de los 3 modelos.

Corre sobre PyTorch por velocidad, pero el JSON resultante lo consumen los dos
frameworks (--hparams-from), para que la comparacion TF vs PyTorch no quede sesgada.
Cada trial entrena con LA MISMA receta que el entrenamiento final (cabezal, resolucion,
profundidad, augmentation y mixup de TASK_DEFAULTS): tunear con una receta distinta
encuentra hiperparametros que no transfieren.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import optuna
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.common_pytorch import (  # noqa: E402
    build_dataloaders,
    class_weights_from_counts,
    evaluate,
    run_epoch,
)
from utils.model_defs_pytorch import SimpleCNN  # noqa: E402
from utils.report_common import TASK_DEFAULTS  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent

# epocas cortas por trial: alcanza para comparar hiperparametros entre si, la corrida final
# completa se hace aparte. Las razas usan mas epocas porque con mixup + augmentation fuerte
# las primeras epocas son ruidosas (y sus datasets chicos hacen la epoca barata).
TUNE_EPOCHS = {"detector": 8, "dog_breed": 14, "cat_breed": 14}


def objective(trial: optuna.Trial, args: argparse.Namespace, device: torch.device) -> float:
    recipe = TASK_DEFAULTS[args.task]
    tune_epochs = TUNE_EPOCHS[args.task]

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    print(
        f"\n--- Trial {trial.number + 1}/{args.trials}: lr={lr:.2e} batch={batch_size} "
        f"dropout={dropout:.2f} wd={weight_decay:.2e} ---",
        flush=True,
    )

    train_loader, val_loader, _, class_names, class_counts = build_dataloaders(
        args.data_dir, batch_size, aug=recipe["aug"], img_size=recipe["img_size"]
    )
    model = SimpleCNN(
        num_classes=len(class_names),
        dropout=dropout,
        head=recipe["head"],
        blocks=recipe["blocks"],
        img_size=recipe["img_size"],
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_counts(class_counts, device))
    # AdamW para igualar el weight_decay desacoplado de Keras (ver common_pytorch.py)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    for epoch in range(tune_epochs):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, mixup=recipe["mixup"]
        )
        val_metrics = evaluate(model, val_loader, device, class_names)
        best_val_f1 = max(best_val_f1, val_metrics["f1_macro"])
        # train_acc muy por encima de val_acc = el modelo esta memorizando
        print(
            f"  epoca {epoch + 1}/{tune_epochs}: train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f} "
            f"({time.time() - epoch_start:.0f}s)",
            flush=True,
        )

        trial.report(val_metrics["f1_macro"], epoch)
        if trial.should_prune():
            print("  -> descartado por el pruner (va peor que la mediana)", flush=True)
            raise optuna.TrialPruned()

    return best_val_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda de hiperparametros con Optuna (PyTorch)")
    parser.add_argument("task", choices=tuple(TASK_DEFAULTS))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = ROOT_DIR / "data" / "processed" / args.task
    if args.out is None:
        args.out = ROOT_DIR / "metrics" / f"{args.task}_best_hparams.json"

    recipe = TASK_DEFAULTS[args.task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Tarea: {args.task} | receta: {recipe} | device: {device}")
    print(
        f"Optuna: {args.trials} trials x {TUNE_EPOCHS[args.task]} epocas cada uno. "
        "Los trials que van peor que la mediana se cortan antes (MedianPruner)."
    )

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, args, device), n_trials=args.trials)

    print(f"\nMejor val_f1_macro ({TUNE_EPOCHS[args.task]} epocas por trial): {study.best_value:.4f}")
    print(f"Mejores hiperparametros: {study.best_params}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tarea": args.task,
                "receta": recipe,
                "tune_epochs": TUNE_EPOCHS[args.task],
                "trials": args.trials,
                "best_val_f1_macro": study.best_value,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )
    print(f"Guardado en {args.out}")


if __name__ == "__main__":
    main()
