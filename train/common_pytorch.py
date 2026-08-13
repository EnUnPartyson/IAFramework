"""Motor de entrenamiento PyTorch compartido por los 3 modelos (detector, raza perro, raza gato).

Cada script train_<modelo>_pytorch.py es un wrapper fino sobre train_model(). El criterio de
seleccion de modelo, early stopping y scheduler usan val_accuracy en ambos frameworks para que
la comparacion TF vs PyTorch sea simetrica (ver DECISIONS.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.model_defs_pytorch import SimpleCNN  # noqa: E402
from utils.report_common import (  # noqa: E402
    class_weight_values,
    file_size_mb,
    metrics_from_predictions,
    save_confusion_matrix_plot,
    save_metrics_json,
)
from utils.transforms_pytorch import get_eval_transforms, get_train_transforms  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5


def build_arg_parser(task: str, default_epochs: int = 30) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Entrena el modelo '{task}' en PyTorch")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data" / "processed" / task)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6, help="early stopping: epocas sin mejorar val_accuracy")
    parser.add_argument("--model-out", type=Path, default=ROOT_DIR / "models" / f"{task}_pytorch.pt")
    parser.add_argument("--metrics-out", type=Path, default=ROOT_DIR / "metrics" / f"{task}_pytorch_metrics.json")
    return parser


def build_dataloaders(
    data_dir: Path, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], list[int]]:
    train_ds = ImageFolder(data_dir / "train", transform=get_train_transforms())
    val_ds = ImageFolder(data_dir / "val", transform=get_eval_transforms())
    test_ds = ImageFolder(data_dir / "test", transform=get_eval_transforms())

    class_names = list(train_ds.classes)
    class_counts = [0] * len(class_names)
    for _, label in train_ds.samples:
        class_counts[label] += 1

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, class_names, class_counts


def class_weights_from_counts(counts: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(class_weight_values(counts), dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return metrics_from_predictions(all_labels, all_preds, class_names)


def train_model(task: str, args: argparse.Namespace, expected_classes: tuple[str, ...] | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{task}/pytorch] device: {device}")

    train_loader, val_loader, test_loader, class_names, class_counts = build_dataloaders(
        args.data_dir, args.batch_size
    )
    if expected_classes is not None and tuple(class_names) != tuple(expected_classes):
        raise ValueError(f"Clases en {args.data_dir} ({class_names}) != esperadas ({expected_classes})")
    print(f"[{task}/pytorch] clases ({len(class_names)}): {dict(zip(class_names, class_counts))}")

    model = SimpleCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_counts(class_counts, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    history: list[dict] = []
    best_val_acc = -1.0
    epochs_without_improvement = 0
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })
        print(
            f"[{task}/pytorch][{epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {"state_dict": model.state_dict(), "class_names": class_names, "dropout": args.dropout},
                args.model_out,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[{task}/pytorch] early stopping en epoca {epoch} (paciencia {args.patience})")
                break

    training_time = time.time() - start_time

    checkpoint = torch.load(args.model_out, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader, device, class_names)

    report = {
        "framework": "pytorch",
        "tarea": task,
        "hiperparametros": {
            "epochs_max": args.epochs,
            "epochs_corridas": epochs_run,
            "batch_size": args.batch_size,
            "lr_inicial": args.lr,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "early_stopping_paciencia": args.patience,
            "scheduler": f"ReduceLROnPlateau(factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})",
        },
        "distribucion_train": dict(zip(class_names, class_counts)),
        "tiempo_entrenamiento_seg": round(training_time, 1),
        "mejor_val_accuracy": best_val_acc,
        "tamano_pesos_mb": file_size_mb(args.model_out),
        "historia": history,
        "test": test_metrics,
    }
    save_metrics_json(report, args.metrics_out)
    save_confusion_matrix_plot(
        test_metrics["matriz_confusion"],
        class_names,
        args.metrics_out.parent / f"{task}_pytorch_confusion_matrix.png",
    )

    print(f"[{task}/pytorch] modelo: {args.model_out}")
    print(f"[{task}/pytorch] metricas: {args.metrics_out}")
    print(f"[{task}/pytorch] test accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
    return report
