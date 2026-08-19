"""Pipeline de inferencia de 2 etapas (PyTorch), compartido por el script de camara
(predict_camera.py) y por el servidor web para celular (server.py).

Etapa 1: detector -> perro / gato / ninguno
Etapa 2: si es perro -> modelo de raza de perro; si es gato -> modelo de raza de gato

Soporta el modo "raza no identificada": si la confianza de la raza no llega al umbral
calibrado durante el entrenamiento, se reporta como no identificada en vez de inventar.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.model_defs_pytorch import SimpleCNN  # noqa: E402
from utils.transforms_pytorch import get_eval_transforms  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"

# si el entrenamiento no dejo un umbral calibrado, se usa este
DEFAULT_UNKNOWN_THRESHOLD = 0.45


@dataclass
class LoadedModel:
    model: SimpleCNN
    class_names: list[str]
    img_size: int
    unknown_threshold: float


@dataclass
class Prediction:
    """Resultado de correr el pipeline sobre una imagen."""

    especie: str                      # "perro" | "gato" | "ninguno"
    especie_confianza: float
    raza: str | None = None           # None si no aplica (especie "ninguno")
    raza_confianza: float | None = None
    raza_identificada: bool = True    # False -> confianza bajo el umbral
    top_razas: list[tuple[str, float]] = field(default_factory=list)

    @property
    def resumen(self) -> str:
        if self.especie == "ninguno":
            return f"Sin mascota detectada ({self.especie_confianza:.0%})"
        if self.raza is None:
            return f"{self.especie.capitalize()} ({self.especie_confianza:.0%})"
        if not self.raza_identificada:
            return f"{self.especie.capitalize()} ({self.especie_confianza:.0%}) - raza no identificada"
        return f"{self.especie.capitalize()} {self.raza} ({self.raza_confianza:.0%})"

    def to_dict(self) -> dict:
        return {
            "especie": self.especie,
            "especie_confianza": round(self.especie_confianza, 4),
            "raza": self.raza,
            "raza_confianza": round(self.raza_confianza, 4) if self.raza_confianza is not None else None,
            "raza_identificada": self.raza_identificada,
            "top_razas": [(n, round(p, 4)) for n, p in self.top_razas],
            "resumen": self.resumen,
        }


def _read_threshold(task: str) -> float:
    """Umbral de 'raza no identificada' calibrado durante el entrenamiento."""
    path = METRICS_DIR / f"{task}_pytorch_metrics.json"
    if not path.exists():
        return DEFAULT_UNKNOWN_THRESHOLD
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f)["raza_no_identificada"]["umbral_sugerido"])
    except (KeyError, ValueError, TypeError):
        return DEFAULT_UNKNOWN_THRESHOLD


def load_model(task: str, device: torch.device) -> LoadedModel:
    path = MODELS_DIR / f"{task}_pytorch.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}. Bajar los pesos de la EC2 con scp (ver README) antes de correr inferencia."
        )
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
    return LoadedModel(model, class_names, img_size, _read_threshold(task))


class PetPipeline:
    """Carga los 3 modelos una sola vez y clasifica imagenes."""

    def __init__(self, device: str | None = None, forced: bool = False, tta: bool = True) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # modo forzado: ignora la clase "ninguno" y responde siempre perro o gato
        self.forced = forced
        # TTA: promedia la prediccion de la imagen y su espejo horizontal. Perros y gatos son
        # simetricos, asi que el espejo es una vista valida de la misma mascota; promediar dos
        # vistas baja la varianza de la prediccion. Cuesta el doble de forward pass (irrelevante
        # en modelos de este tamano) y no requiere reentrenar nada.
        self.tta = tta
        self.detector = load_model("detector", self.device)
        # los modelos de raza son opcionales: sin ellos el pipeline igual detecta la especie
        self.breed_models: dict[str, LoadedModel] = {}
        for especie, task in (("perro", "dog_breed"), ("gato", "cat_breed")):
            try:
                self.breed_models[especie] = load_model(task, self.device)
            except FileNotFoundError:
                print(f"[aviso] falta el modelo de raza de {especie}; solo se detectara la especie")

    @torch.no_grad()
    def _probs(self, loaded: LoadedModel, image: Image.Image) -> torch.Tensor:
        tensor = get_eval_transforms(img_size=loaded.img_size)(image).unsqueeze(0).to(self.device)
        if self.tta:
            # [original, espejo] en un solo batch: una pasada, dos vistas
            tensor = torch.cat([tensor, torch.flip(tensor, dims=[3])], dim=0)
        return torch.softmax(loaded.model(tensor), dim=1).mean(dim=0)

    def predict(self, image: Image.Image, top_k: int = 3) -> Prediction:
        image = image.convert("RGB")

        probs = self._probs(self.detector, image)
        if self.forced and "ninguno" in self.detector.class_names:
            probs = probs.clone()
            probs[self.detector.class_names.index("ninguno")] = -1.0
        idx = int(probs.argmax())
        especie = self.detector.class_names[idx]
        result = Prediction(especie=especie, especie_confianza=float(probs[idx]))

        loaded = self.breed_models.get(especie)
        if loaded is None:
            return result

        breed_probs = self._probs(loaded, image)
        k = min(top_k, len(loaded.class_names))
        top_p, top_i = torch.topk(breed_probs, k)
        result.top_razas = [(loaded.class_names[int(i)], float(p)) for p, i in zip(top_p, top_i)]
        result.raza = loaded.class_names[int(top_i[0])]
        result.raza_confianza = float(top_p[0])
        result.raza_identificada = result.raza_confianza >= loaded.unknown_threshold
        return result

    def describe(self) -> dict:
        """Info de los modelos cargados, util para mostrar en la app."""
        info = {
            "device": str(self.device),
            "modo_forzado": self.forced,
            "tta": self.tta,
            "detector": {"clases": self.detector.class_names, "img_size": self.detector.img_size},
        }
        for especie, loaded in self.breed_models.items():
            info[f"razas_{especie}"] = {
                "clases": loaded.class_names,
                "img_size": loaded.img_size,
                "umbral_no_identificada": loaded.unknown_threshold,
            }
        return info
