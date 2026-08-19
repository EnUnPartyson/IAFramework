"""Entrenamiento del modo PRO: transfer learning con backbones preentrenados (solo PyTorch).

Es la rama posterior al tag v1-presentacion. Los 6 modelos base (desde cero, TF vs PyTorch)
quedan congelados como entregable; aca se levanta la restriccion de no usar preentrenados y
se aplica todo lo que mejora resultados:

  - backbone preentrenado en ImageNet + fine-tuning en dos fases
    (fase 1: solo la cabeza, backbone congelado; fase 2: todo, con LR chico y diferenciado)
  - label smoothing, MixUp, cosine annealing, precision mixta por defecto
  - misma captura de metricas que el pipeline base (ROC/PR, umbral de desconocidas,
    matriz de confusion, curvas) para poder comparar base vs pro con compare_pro.py

Uso (en la EC2, dentro de venv-torch):
    python train/train_pro_pytorch.py --task detector
    python train/train_pro_pytorch.py --task dog_breed --arch efficientnet_v2_s
    python train/train_pro_pytorch.py --task cat_breed --epochs 40

Los modelos salen a models/<task>_pro_pytorch.pt y las metricas a metrics/<task>_pro_metrics.json:
nunca pisan los archivos de la version base.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.common_pytorch import (  # noqa: E402
    build_dataloaders,
    class_weights_from_counts,
    collect_logits,
    default_workers,
    run_epoch,
)
from utils.model_defs_pro_pytorch import (  # noqa: E402
    PRO_ARCH_DEFAULT,
    PRO_ARCHS,
    build_pretrained,
    freeze_backbone,
    head_parameters,
)
from utils.report_common import (  # noqa: E402
    file_size_mb,
    forced_mode_metrics,
    metrics_from_predictions,
    open_set_analysis,
    roc_pr_analysis,
    save_confusion_matrix_plot,
    save_metrics_json,
    save_open_set_plot,
    save_roc_pr_plot,
    save_training_curves_plot,
    softmax,
)
from utils.transforms_pytorch import (  # noqa: E402
    AUG_BASE,
    AUG_PRODUCCION,
    AUG_STRONG,
    NORM_IMAGENET,
    get_eval_transforms,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

# recetas pro por tarea. Con transfer learning hacen falta muchas menos epocas que desde
# cero: el backbone ya sabe ver; solo se adapta. img_size 224 = la resolucion nativa de los
# preentrenados de torchvision.
# aug "produccion" en las 3 tareas: strong + degradaciones de camara real (JPEG, perspectiva,
# gris). El modelo pro existe para produccion, no para ganar un benchmark de test limpio.
PRO_DEFAULTS = {
    "detector": {"epochs": 15, "warmup_epochs": 2, "aug": "produccion", "mixup": 0.0, "patience": 5},
    "dog_breed": {"epochs": 30, "warmup_epochs": 3, "aug": "produccion", "mixup": 0.2, "patience": 8},
    "cat_breed": {"epochs": 30, "warmup_epochs": 3, "aug": "produccion", "mixup": 0.2, "patience": 8},
}
LABEL_SMOOTHING = 0.1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrenamiento pro (transfer learning, PyTorch)")
    parser.add_argument("--task", required=True, choices=tuple(PRO_DEFAULTS))
    parser.add_argument("--arch", choices=PRO_ARCHS, default=PRO_ARCH_DEFAULT)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="default: data/processed/<task>_pro si existe, si no data/processed/<task>")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="epocas de fase 1 (solo cabeza, backbone congelado)")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-head", type=float, default=1e-3, help="LR de la cabeza (ambas fases)")
    parser.add_argument("--lr-backbone", type=float, default=1e-4,
                        help="LR del backbone en fase 2; 10x menor que la cabeza para no borrar lo preentrenado")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aug", choices=(AUG_BASE, AUG_STRONG, AUG_PRODUCCION), default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="apaga la precision mixta (en pro va encendida por defecto: los "
                             "backbones grandes si la aprovechan)")
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    d = PRO_DEFAULTS[args.task]
    for campo in ("epochs", "warmup_epochs", "patience", "aug", "mixup"):
        if getattr(args, campo) is None:
            setattr(args, campo, d[campo])
    if args.data_dir is None:
        pro_dir = ROOT_DIR / "data" / "processed" / f"{args.task}_pro"
        args.data_dir = pro_dir if pro_dir.exists() else ROOT_DIR / "data" / "processed" / args.task
    if args.model_out is None:
        args.model_out = ROOT_DIR / "models" / f"{args.task}_pro_pytorch.pt"
    if args.metrics_out is None:
        args.metrics_out = ROOT_DIR / "metrics" / f"{args.task}_pro_metrics.json"
    return args


def calibrar_temperatura(val_logits: "np.ndarray", val_labels: list[int]) -> float:
    """Temperature scaling (Guo et al. 2017): un escalar T que corrige la sobreconfianza.

    Las redes modernas dicen "95% seguro" cuando aciertan el 80%: en produccion eso infla
    las confianzas que ve el usuario y desajusta el umbral de "raza no identificada".
    Dividir los logits por T (ajustado en validacion) arregla la calibracion sin cambiar
    NINGUNA prediccion: softmax(z/T) preserva el argmax.
    """
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels)
    log_t = torch.nn.Parameter(torch.zeros(1))  # T = exp(log_t) > 0 siempre
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp().clamp(0.1, 10.0))
    antes = float(nn.functional.cross_entropy(logits, labels))
    despues = float(nn.functional.cross_entropy(logits / t, labels))
    print(f"[calibracion] T={t:.3f} | NLL validacion: {antes:.4f} -> {despues:.4f}")
    return t


def main() -> None:
    args = resolve_args(build_arg_parser().parse_args())
    task = args.task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{task}/pro] device: {device}, arch: {args.arch}, datos: {args.data_dir}")

    if device.type == "cuda":
        # TF32 (gratis en Ampere+: A10G, L4, L40S) y autotuning de cuDNN con tamano fijo
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, class_names, class_counts = build_dataloaders(
        args.data_dir, args.batch_size, aug=args.aug, img_size=args.img_size,
        workers=args.workers, norm=NORM_IMAGENET,
        # razas: PetFinder deja clases muy desparejas (3000 vs ~400); el sampler las nivela
        # por epoca. El detector NO: su desbalance 35/35/30 es de diseno y lo cubre la loss
        balanced=(task != "detector"),
    )
    print(f"[{task}/pro] clases ({len(class_names)}): {dict(zip(class_names, class_counts))}")

    model = build_pretrained(args.arch, num_classes=len(class_names), dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{task}/pro] {n_params:,} parametros, {args.img_size}px, label_smoothing={LABEL_SMOOTHING}")

    # weighted loss solo para el detector (desbalance intencional 35/35/30); las razas ya
    # llegan relativamente parejas y el smoothing basta
    weight = class_weights_from_counts(class_counts, device) if "ninguno" in class_names else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=LABEL_SMOOTHING)

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"[{task}/pro] precision mixta: {'si' if use_amp else 'no'}")

    history: list[dict] = []
    best_val_acc = -1.0
    epochs_sin_mejora = 0
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    epochs_run = 0

    def guardar() -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": args.img_size,
                "arch": args.arch,
                "norm": "imagenet",
                "dropout": args.dropout,
                "modo": "pro",
            },
            args.model_out,
        )

    # ---- fase 1: warmup de la cabeza con el backbone congelado -------------------------
    freeze_backbone(model, args.arch, frozen=True)
    opt_warmup = torch.optim.AdamW(head_parameters(model, args.arch), lr=args.lr_head,
                                   weight_decay=args.weight_decay)
    for epoch in range(1, args.warmup_epochs + 1):
        epochs_run = epoch
        tl, ta = run_epoch(model, train_loader, criterion, device, opt_warmup, mixup=0.0, scaler=scaler)
        vl, va = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "fase": "warmup", "train_loss": tl, "train_acc": ta,
                        "val_loss": vl, "val_acc": va, "lr": args.lr_head})
        print(f"[{task}/pro][warmup {epoch}/{args.warmup_epochs}] train_acc={ta:.4f} val_acc={va:.4f}")
        if va > best_val_acc:
            best_val_acc = va
            guardar()

    # ---- fase 2: fine-tuning completo con LR diferenciado y cosine annealing -----------
    freeze_backbone(model, args.arch, frozen=False)
    head_ids = {id(p) for p in head_parameters(model, args.arch)}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_parameters(model, args.arch), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    fine_epochs = max(1, args.epochs - args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_epochs)

    for epoch in range(args.warmup_epochs + 1, args.epochs + 1):
        epochs_run = epoch
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer, mixup=args.mixup, scaler=scaler)
        vl, va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr_actual = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "fase": "fine", "train_loss": tl, "train_acc": ta,
                        "val_loss": vl, "val_acc": va, "lr": lr_actual})
        print(f"[{task}/pro][{epoch}/{args.epochs}] train_loss={tl:.4f} train_acc={ta:.4f} "
              f"val_loss={vl:.4f} val_acc={va:.4f} lr_backbone={lr_actual:.2e}")
        if va > best_val_acc:
            best_val_acc = va
            epochs_sin_mejora = 0
            guardar()
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= args.patience:
                print(f"[{task}/pro] early stopping en epoca {epoch} (paciencia {args.patience})")
                break

    training_time = time.time() - start_time

    # ---- evaluacion final con el mejor checkpoint (mismas metricas que el pipeline base)
    ckpt = torch.load(args.model_out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_metrics = metrics_from_predictions(test_labels, test_logits.argmax(axis=1).tolist(), class_names)
    roc_pr = roc_pr_analysis(test_labels, test_logits, class_names)

    forced_metrics = None
    if "ninguno" in class_names:
        forced_metrics = forced_mode_metrics(test_logits, test_labels, class_names, "ninguno")
        print(f"[{task}/pro] modo forzado: accuracy={forced_metrics['accuracy']:.4f}")

    # calibracion de confianza: T se ajusta en validacion, viaja en el checkpoint y la
    # inferencia divide los logits por T. El umbral de "raza no identificada" se calcula
    # sobre las probabilidades YA calibradas, que son las que vera el pipeline en produccion.
    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperatura = calibrar_temperatura(val_logits, val_labels)
    ckpt["temperature"] = temperatura
    torch.save(ckpt, args.model_out)

    val_probs = softmax(val_logits / temperatura)
    val_maxprob = val_probs.max(axis=1).tolist()
    val_correct = (val_probs.argmax(axis=1) == np.asarray(val_labels)).tolist()
    unknown_dir = args.data_dir / "unknown"
    unknown_maxprob = None
    if unknown_dir.exists():
        unknown_ds = ImageFolder(unknown_dir, transform=get_eval_transforms(args.img_size, norm=NORM_IMAGENET))
        unknown_loader = DataLoader(unknown_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        unknown_logits, _ = collect_logits(model, unknown_loader, device)
        unknown_maxprob = softmax(unknown_logits / temperatura).max(axis=1).tolist()
    open_set = open_set_analysis(val_maxprob, val_correct, unknown_maxprob)

    report = {
        "framework": "pytorch",
        "modo": "pro",
        "tarea": task,
        "hiperparametros": {
            "arch": args.arch,
            "transfer_learning": True,
            "epochs_max": args.epochs,
            "epochs_corridas": epochs_run,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.batch_size,
            "lr_head": args.lr_head,
            "lr_backbone": args.lr_backbone,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "label_smoothing": LABEL_SMOOTHING,
            "scheduler": "CosineAnnealingLR",
            "img_size": args.img_size,
            "n_parametros": n_params,
            "precision_mixta": use_amp,
            "aug": args.aug,
            "mixup_alpha": args.mixup,
            "early_stopping_paciencia": args.patience,
            "norm": "imagenet",
            "sampling_balanceado": task != "detector",
            "temperatura_calibracion": temperatura,
        },
        "distribucion_train": dict(zip(class_names, class_counts)),
        "tiempo_entrenamiento_seg": round(training_time, 1),
        "mejor_val_accuracy": best_val_acc,
        "tamano_pesos_mb": file_size_mb(args.model_out),
        "historia": history,
        "test": test_metrics,
        "raza_no_identificada": open_set,
        "roc_pr": roc_pr,
    }
    if forced_metrics is not None:
        report["test_modo_forzado"] = forced_metrics
    save_metrics_json(report, args.metrics_out)

    plots_dir = args.metrics_out.parent
    save_confusion_matrix_plot(test_metrics["matriz_confusion"], class_names,
                               plots_dir / f"{task}_pro_confusion_matrix.png")
    save_confusion_matrix_plot(test_metrics["matriz_confusion"], class_names,
                               plots_dir / f"{task}_pro_confusion_matrix_norm.png", normalize=True)
    save_training_curves_plot(history, plots_dir / f"{task}_pro_training_curves.png", f"{task} - PRO ({args.arch})")
    save_open_set_plot(open_set, plots_dir / f"{task}_pro_umbral_desconocidas.png", f"{task} - PRO")
    save_roc_pr_plot(roc_pr, plots_dir / f"{task}_pro_roc_pr.png", f"{task} - PRO ({args.arch})")

    print(f"[{task}/pro] modelo: {args.model_out}")
    print(f"[{task}/pro] test accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
