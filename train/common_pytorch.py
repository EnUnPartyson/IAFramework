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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.model_defs_pytorch import HEAD_FLATTEN, HEAD_GAP, SimpleCNN  # noqa: E402
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
from utils.transforms_pytorch import AUG_BASE, AUG_STRONG, get_eval_transforms, get_train_transforms  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
# paciencia 4 y no 2: con val_accuracy ruidosa (normal en datasets chicos con MixUp), una
# paciencia corta interpreta el ruido como estancamiento y desploma el LR en pocas epocas
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.5


def build_arg_parser(task: str) -> argparse.ArgumentParser:
    # la receta por tarea (cabezal, augmentation, mixup, resolucion, profundidad, epocas)
    # vive en TASK_DEFAULTS, compartida con el motor TF y con Optuna para que nunca diverjan
    d = TASK_DEFAULTS[task]
    parser = argparse.ArgumentParser(description=f"Entrena el modelo '{task}' en PyTorch")
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
        help="JSON generado por tune_pytorch.py con los mejores hiperparametros",
    )
    parser.add_argument(
        "--patience", type=int, default=d["patience"], help="early stopping: epocas sin mejorar val_accuracy"
    )
    parser.add_argument("--model-out", type=Path, default=ROOT_DIR / "models" / f"{task}_pytorch.pt")
    parser.add_argument("--metrics-out", type=Path, default=ROOT_DIR / "metrics" / f"{task}_pytorch_metrics.json")
    return parser


def build_dataloaders(
    data_dir: Path, batch_size: int, aug: str = AUG_BASE, img_size: int = 128
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], list[int]]:
    train_ds = ImageFolder(data_dir / "train", transform=get_train_transforms(img_size=img_size, aug=aug))
    val_ds = ImageFolder(data_dir / "val", transform=get_eval_transforms(img_size=img_size))
    test_ds = ImageFolder(data_dir / "test", transform=get_eval_transforms(img_size=img_size))

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
    mixup: float = 0.0,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    beta = torch.distributions.Beta(mixup, mixup) if (is_train and mixup > 0) else None
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()

            if beta is not None:
                # MixUp: mezcla cada imagen con otra del batch; la loss se reparte entre
                # ambas etiquetas. Sin peso por clase a proposito (ver DECISIONS.md): con
                # peso, lam*CE(y1,w1)+(1-lam)*CE(y2,w2) no es igual a la loss de TF sobre la
                # etiqueta blanda combinada (la ponderacion rompe la linealidad de la cross
                # entropy); sin peso, ambas formulas SI son matematicamente identicas.
                lam = beta.sample().item()
                perm = torch.randperm(images.size(0), device=device)
                mixed = lam * images + (1.0 - lam) * images[perm]
                logits = model(mixed)
                loss = lam * F.cross_entropy(logits, labels) + (1.0 - lam) * F.cross_entropy(logits, labels[perm])
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            # con mixup el accuracy de train es aproximado (se compara contra la etiqueta dominante)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[float], list[int], list[int]]:
    """Devuelve (probabilidad maxima softmax, prediccion, etiqueta real) por imagen."""
    model.eval()
    maxprobs, preds, labels_out = [], [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        p, idx = probs.max(dim=1)
        maxprobs.extend(p.cpu().tolist())
        preds.extend(idx.cpu().tolist())
        labels_out.extend(labels.tolist())
    return maxprobs, preds, labels_out


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, list[int]]:
    """Logits crudos + etiquetas: base para metricas de test, modo forzado y umbrales."""
    model.eval()
    chunks, labels_out = [], []
    for images, labels in loader:
        chunks.append(model(images.to(device)).cpu().numpy())
        labels_out.extend(labels.tolist())
    return np.concatenate(chunks), labels_out


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

    hp = resolve_hparams(args, args.hparams_from)
    print(f"[{task}/pytorch] hiperparametros: {hp}")

    train_loader, val_loader, test_loader, class_names, class_counts = build_dataloaders(
        args.data_dir, hp["batch_size"], aug=args.aug, img_size=args.img_size
    )
    if expected_classes is not None and tuple(class_names) != tuple(expected_classes):
        raise ValueError(f"Clases en {args.data_dir} ({class_names}) != esperadas ({expected_classes})")
    print(f"[{task}/pytorch] clases ({len(class_names)}): {dict(zip(class_names, class_counts))}")

    model = SimpleCNN(
        num_classes=len(class_names),
        dropout=hp["dropout"],
        head=args.head,
        blocks=args.blocks,
        img_size=args.img_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{task}/pytorch] cabezal={args.head}, {args.blocks} bloques, {args.img_size}px, {n_params:,} parametros")
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_counts(class_counts, device))
    # AdamW y no Adam: Keras aplica weight_decay DESACOPLADO (estilo AdamW), mientras que
    # torch.optim.Adam lo aplica ACOPLADO (L2 sumado al gradiente). Con wd=1e-2, tras 50
    # pasos, Adam lleva un peso de 1.0 a 0.950 y AdamW/Keras a 0.9995: 100x de diferencia
    # en regularizacion efectiva. Como ambos frameworks comparten los hiperparametros que
    # busca Optuna, usar Adam aca invalidaba la comparacion. Ver DECISIONS.md.
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
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
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, mixup=args.mixup)
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
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "dropout": hp["dropout"],
                    "head": args.head,
                    "blocks": args.blocks,
                    "img_size": args.img_size,
                },
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
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_metrics = metrics_from_predictions(test_labels, test_logits.argmax(axis=1).tolist(), class_names)

    # modo forzado del detector (requisito del profesor): perro vs gato ignorando "ninguno"
    forced_metrics = None
    if "ninguno" in class_names:
        forced_metrics = forced_mode_metrics(test_logits, test_labels, class_names, "ninguno")
        print(f"[{task}/pytorch] modo forzado (perro vs gato): accuracy={forced_metrics['accuracy']:.4f}")

    # modo "raza no identificada": umbral de confianza calibrado en validacion, evaluado
    # contra razas nunca vistas si prepare_data.py dejo la carpeta unknown/
    val_maxprob, val_preds, val_labels = collect_probs(model, val_loader, device)
    val_correct = [p == t for p, t in zip(val_preds, val_labels)]
    unknown_dir = args.data_dir / "unknown"
    unknown_maxprob: list[float] | None = None
    if unknown_dir.exists():
        unknown_ds = ImageFolder(unknown_dir, transform=get_eval_transforms(img_size=args.img_size))
        unknown_loader = DataLoader(unknown_ds, batch_size=hp["batch_size"], shuffle=False, num_workers=2)
        unknown_maxprob, _, _ = collect_probs(model, unknown_loader, device)
    open_set = open_set_analysis(val_maxprob, val_correct, unknown_maxprob)
    print(f"[{task}/pytorch] raza no identificada: {open_set['en_umbral_sugerido']}")

    report = {
        "framework": "pytorch",
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
            "n_parametros": n_params,
            "aug": args.aug,
            "mixup_alpha": args.mixup,
        },
        "distribucion_train": dict(zip(class_names, class_counts)),
        "tiempo_entrenamiento_seg": round(training_time, 1),
        "mejor_val_accuracy": best_val_acc,
        "tamano_pesos_mb": file_size_mb(args.model_out),
        "historia": history,
        "test": test_metrics,
        "raza_no_identificada": open_set,
    }
    if forced_metrics is not None:
        report["test_modo_forzado"] = forced_metrics
    save_metrics_json(report, args.metrics_out)

    plots_dir = args.metrics_out.parent
    save_confusion_matrix_plot(
        test_metrics["matriz_confusion"], class_names, plots_dir / f"{task}_pytorch_confusion_matrix.png"
    )
    save_confusion_matrix_plot(
        test_metrics["matriz_confusion"],
        class_names,
        plots_dir / f"{task}_pytorch_confusion_matrix_norm.png",
        normalize=True,
    )
    save_training_curves_plot(history, plots_dir / f"{task}_pytorch_training_curves.png", f"{task} - PyTorch")
    save_open_set_plot(open_set, plots_dir / f"{task}_pytorch_umbral_desconocidas.png", f"{task} - PyTorch")

    print(f"[{task}/pytorch] modelo: {args.model_out}")
    print(f"[{task}/pytorch] metricas: {args.metrics_out}")
    print(f"[{task}/pytorch] test accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
    return report
