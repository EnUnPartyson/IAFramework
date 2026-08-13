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

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed" / "detector"
DEFAULT_OUT_PATH = ROOT_DIR / "metrics" / "detector_best_hparams.json"

# epocas cortas por trial: alcanza para comparar hiperparametros, la corrida final completa
# se hace aparte con train_detector_pytorch.py usando los valores que encuentre este script
TUNE_EPOCHS = 8


def objective(trial: optuna.Trial, data_dir: Path, device: torch.device, n_trials: int) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    print(
        f"\n--- Trial {trial.number + 1}/{n_trials}: lr={lr:.2e} batch={batch_size} "
        f"dropout={dropout:.2f} wd={weight_decay:.2e} ---",
        flush=True,
    )

    train_loader, val_loader, _, class_names, class_counts = build_dataloaders(data_dir, batch_size)
    model = SimpleCNN(num_classes=len(class_names), dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_counts(class_counts, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    for epoch in range(TUNE_EPOCHS):
        epoch_start = time.time()
        run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = evaluate(model, val_loader, device, class_names)
        best_val_f1 = max(best_val_f1, val_metrics["f1_macro"])
        print(
            f"  epoca {epoch + 1}/{TUNE_EPOCHS}: val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} ({time.time() - epoch_start:.0f}s)",
            flush=True,
        )

        trial.report(val_metrics["f1_macro"], epoch)
        if trial.should_prune():
            print("  -> descartado por el pruner (va peor que la mediana)", flush=True)
            raise optuna.TrialPruned()

    return best_val_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda de hiperparametros del Modelo 1 (PyTorch) con Optuna")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    print(
        f"Optuna: {args.trials} trials x {TUNE_EPOCHS} epocas cada uno. "
        "Los trials que van peor que la mediana se cortan antes (MedianPruner)."
    )
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda trial: objective(trial, args.data_dir, device, args.trials), n_trials=args.trials
    )

    print(f"Mejor val_f1_macro ({TUNE_EPOCHS} epocas por trial): {study.best_value:.4f}")
    print(f"Mejores hiperparametros: {study.best_params}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"tune_epochs": TUNE_EPOCHS, "best_val_f1_macro": study.best_value, "best_params": study.best_params},
            f,
            indent=2,
        )
    print(f"Guardado en {args.out}")
    print(
        "Para la corrida final: python train/train_detector_pytorch.py "
        f"--lr {study.best_params['lr']} --batch-size {study.best_params['batch_size']} "
        f"--dropout {study.best_params['dropout']} --weight-decay {study.best_params['weight_decay']}"
    )


if __name__ == "__main__":
    main()
