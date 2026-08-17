from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.model_defs_pytorch import DETECTOR_CLASS_NAMES as CLASS_NAMES  # noqa: E402
from utils.model_defs_pytorch import SimpleCNN  # noqa: E402
from utils.transforms_pytorch import NORM_MEAN, NORM_STD, get_eval_transforms  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "detector_pytorch.pt"
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed" / "detector" / "val"
DEFAULT_OUT_DIR = ROOT_DIR / "metrics" / "gradcam"

def _last_relu_index(model: SimpleCNN) -> int:
    # ultimo ReLU de "features" (activacion previa al pooling final): capa estandar para
    # Grad-CAM. Se busca dinamicamente porque la cantidad de bloques es configurable.
    import torch.nn as nn

    return max(i for i, layer in enumerate(model.features) if isinstance(layer, nn.ReLU))


class GradCAM:
    def __init__(self, model: SimpleCNN) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer = model.features[_last_relu_index(model)]
        # el hook de gradiente va sobre el tensor de salida (no register_full_backward_hook
        # sobre el modulo): los ReLU inplace de la red rompen los backward hooks de modulo
        target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inp, out) -> None:
        self.activations = out.detach().clone()
        out.register_hook(self._save_gradients)

    def _save_gradients(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach().clone()

    def generate(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(image.unsqueeze(0))
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def save_overlay(image: torch.Tensor, cam: np.ndarray, pred_name: str, true_name: str, out_path: Path) -> None:
    img = denormalize(image)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img)
    axes[0].set_title(f"real={true_name}")
    axes[0].axis("off")
    axes[1].imshow(img)
    axes[1].imshow(cam, cmap="jet", alpha=0.5)
    axes[1].set_title(f"pred={pred_name}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera mapas Grad-CAM del detector sobre imagenes de validacion")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-per-class", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    class_names = checkpoint.get("class_names", list(CLASS_NAMES))
    model = SimpleCNN(
        num_classes=len(class_names),
        head=checkpoint.get("head", "flatten"),
        blocks=checkpoint.get("blocks", 4),
        img_size=checkpoint.get("img_size", 128),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    gradcam = GradCAM(model)
    dataset = ImageFolder(args.data_dir, transform=get_eval_transforms())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in CLASS_NAMES}
    for image, label in dataset:
        true_name = CLASS_NAMES[label]
        if counts[true_name] >= args.n_per_class:
            continue

        image = image.to(device)
        pred_idx = model(image.unsqueeze(0)).argmax(dim=1).item()
        cam = gradcam.generate(image, pred_idx)

        out_path = args.out_dir / f"{true_name}_{counts[true_name]:02d}.png"
        save_overlay(image, cam, CLASS_NAMES[pred_idx], true_name, out_path)
        counts[true_name] += 1

        if all(c >= args.n_per_class for c in counts.values()):
            break

    print(f"Grad-CAM guardado en {args.out_dir}")


if __name__ == "__main__":
    main()
