"""Mide cuanto se degrada un modelo entrenado al pasar de fotos de dataset a condiciones de
camara (domain shift), y cuanto de esa degradacion recupera el test-time augmentation.

No entrena nada: solo carga pesos ya entrenados y evalua el split de test en 4 condiciones.

    venv-torch/bin/python train/evaluar_robustez.py --task detector
    venv-torch/bin/python train/evaluar_robustez.py --task dog_breed --task cat_breed

Sale una tabla y un JSON en metrics/<task>_robustez.json. Solo PyTorch: el objetivo es medir
el efecto de las tecnicas, no comparar frameworks (para eso esta compare_frameworks.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utils.model_defs_pytorch import SimpleCNN  # noqa: E402
from utils.report_common import metrics_from_predictions, top_confusions  # noqa: E402
from utils.transforms_pytorch import NORM_MEAN, NORM_STD  # noqa: E402

PROCESSED_ROOT = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"


def transform_limpia(img_size: int) -> transforms.Compose:
    """Lo mismo que ve el modelo en el test normal."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def transform_camara(img_size: int) -> transforms.Compose:
    """Simula una webcam: desenfoque de foco, luz distinta y perdida de nitidez.

    Los parametros son aleatorios, pero se fija la semilla antes de cada evaluacion para que
    las corridas con y sin TTA vean exactamente la misma degradacion y la comparacion sea justa.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.8, 1.6)),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def cargar_modelo(task: str, device: torch.device) -> tuple[SimpleCNN, list[str], int]:
    path = MODELS_DIR / f"{task}_pytorch.pt"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    class_names = list(ckpt["class_names"])
    img_size = int(ckpt.get("img_size", 128))
    model = SimpleCNN(
        num_classes=len(class_names),
        dropout=float(ckpt.get("dropout", 0.4)),
        head=ckpt.get("head", "flatten"),
        blocks=int(ckpt.get("blocks", 4)),
        img_size=img_size,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_names, img_size


@torch.no_grad()
def evaluar(model: SimpleCNN, loader: DataLoader, device: torch.device, tta: bool) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1)
        if tta:
            # promedia con el espejo horizontal, igual que hace el pipeline de inferencia
            probs = (probs + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2
        y_pred.extend(probs.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return y_true, y_pred


def confusion_perro_gato(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict | None:
    """Especifico del detector: cuantos errores son perro<->gato en vez de involucrar 'ninguno'."""
    if not {"perro", "gato"} <= set(class_names):
        return None
    ip, ig = class_names.index("perro"), class_names.index("gato")
    perro_a_gato = sum(1 for t, p in zip(y_true, y_pred) if t == ip and p == ig)
    gato_a_perro = sum(1 for t, p in zip(y_true, y_pred) if t == ig and p == ip)
    errores = sum(1 for t, p in zip(y_true, y_pred) if t != p)
    mutua = perro_a_gato + gato_a_perro
    return {
        "perro_predicho_gato": perro_a_gato,
        "gato_predicho_perro": gato_a_perro,
        "confusion_mutua": mutua,
        "errores_totales": errores,
        "pct_de_los_errores": round(100 * mutua / errores, 1) if errores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", required=True,
                        choices=["detector", "dog_breed", "cat_breed"],
                        help="se puede repetir para evaluar varios modelos")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    for task in args.task:
        model, class_names, img_size = cargar_modelo(task, device)
        test_dir = PROCESSED_ROOT / task / "test"
        if not test_dir.exists():
            print(f"[{task}] falta {test_dir}, se omite")
            continue

        resultados: dict[str, dict] = {}
        for cond, build in (("limpia", transform_limpia), ("camara", transform_camara)):
            for tta in (False, True):
                # misma semilla -> la degradacion aleatoria es identica con y sin TTA
                torch.manual_seed(args.seed)
                ds = ImageFolder(test_dir, transform=build(img_size))
                assert ds.classes == class_names, f"clases del disco {ds.classes} != del checkpoint {class_names}"
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
                y_true, y_pred = evaluar(model, loader, device, tta)
                m = metrics_from_predictions(y_true, y_pred, class_names)
                clave = f"{cond}_tta" if tta else cond
                resultados[clave] = {
                    "accuracy": m["accuracy"],
                    "f1_macro": m["f1_macro"],
                    "top_confusiones": top_confusions(m["matriz_confusion"], class_names, k=3),
                }
                pg = confusion_perro_gato(y_true, y_pred, class_names)
                if pg:
                    resultados[clave]["perro_vs_gato"] = pg

        print(f"=== {task} ({len(class_names)} clases, {img_size}px) ===")
        print(f"{'condicion':<14} {'accuracy':>9} {'f1_macro':>9}")
        for clave, r in resultados.items():
            print(f"{clave:<14} {r['accuracy']:>9.4f} {r['f1_macro']:>9.4f}")

        base = resultados["limpia"]["accuracy"]
        print(f"\n  caida por domain shift  : {resultados['camara']['accuracy'] - base:+.4f}")
        print(f"  ganancia de TTA (limpia): {resultados['limpia_tta']['accuracy'] - base:+.4f}")
        print(f"  ganancia de TTA (camara): "
              f"{resultados['camara_tta']['accuracy'] - resultados['camara']['accuracy']:+.4f}")
        if "perro_vs_gato" in resultados["limpia"]:
            pg = resultados["limpia"]["perro_vs_gato"]
            print(f"  confusion perro<->gato  : {pg['confusion_mutua']} imgs "
                  f"({pg['pct_de_los_errores']}% de los errores)")

        METRICS_DIR.mkdir(exist_ok=True)
        out = METRICS_DIR / f"{task}_robustez.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"task": task, "clases": class_names, "condiciones": resultados}, f,
                      indent=2, ensure_ascii=False)
        print(f"  guardado en {out}\n")


if __name__ == "__main__":
    main()
