"""Pipeline de inferencia de 2 etapas (TensorFlow/Keras).

Espejo de pipeline_pytorch.py: misma interfaz y mismo objeto Prediction, para que el
servidor pueda usar cualquiera de los dos —o ambos a la vez— sin ramificar codigo.

A diferencia del checkpoint de PyTorch, un .keras no guarda los nombres de clase, asi que
se leen de metrics/<tarea>_tensorflow_metrics.json, que los deja el entrenamiento.
Nunca importar torch aca.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from inference.pipeline_pytorch import DEFAULT_UNKNOWN_THRESHOLD, Prediction  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"


@dataclass
class LoadedModelTF:
    model: tf.keras.Model
    class_names: list[str]
    img_size: int
    unknown_threshold: float


def _read_metadata(task: str) -> tuple[list[str], int, float]:
    """Nombres de clase, resolucion y umbral, desde las metricas del entrenamiento."""
    path = METRICS_DIR / f"{task}_tensorflow_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}. El pipeline TF necesita las metricas para saber los nombres de clase."
        )
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)
    clases = list(r["test"]["clases"])
    img_size = int(r["hiperparametros"].get("img_size", 128))
    umbral = float(r.get("raza_no_identificada", {}).get("umbral_sugerido", DEFAULT_UNKNOWN_THRESHOLD))
    return clases, img_size, umbral


def load_model(task: str) -> LoadedModelTF:
    path = MODELS_DIR / f"{task}_tensorflow.keras"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Bajar los pesos antes de correr inferencia.")
    clases, img_size, umbral = _read_metadata(task)
    model = tf.keras.models.load_model(path, compile=False)
    return LoadedModelTF(model, clases, img_size, umbral)


class PetPipelineTF:
    """Carga los 3 modelos de Keras una sola vez y clasifica imagenes."""

    def __init__(self, forced: bool = False, tta: bool = True) -> None:
        self.forced = forced
        # TTA con espejo horizontal, espejo exacto de PetPipeline (PyTorch): ver el comentario ahi
        self.tta = tta
        self.detector = load_model("detector")
        self.breed_models: dict[str, LoadedModelTF] = {}
        for especie, task in (("perro", "dog_breed"), ("gato", "cat_breed")):
            try:
                self.breed_models[especie] = load_model(task)
            except FileNotFoundError:
                print(f"[aviso] falta el modelo TF de raza de {especie}")

    def _probs(self, loaded: LoadedModelTF, image: Image.Image) -> np.ndarray:
        # el modelo espera [0,255]: la normalizacion vive en su capa Rescaling
        arr = np.asarray(image.resize((loaded.img_size, loaded.img_size)), dtype="float32")
        batch = arr[None, ...]
        if self.tta:
            # [original, espejo] en un solo batch; axis=2 es el ancho en (N, alto, ancho, canales)
            batch = np.concatenate([batch, batch[:, :, ::-1, :]], axis=0)
        logits = loaded.model.predict(batch, verbose=0)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.mean(axis=0)

    def predict(self, image: Image.Image, top_k: int = 3) -> Prediction:
        image = image.convert("RGB")

        probs = self._probs(self.detector, image)
        if self.forced and "ninguno" in self.detector.class_names:
            probs = probs.copy()
            probs[self.detector.class_names.index("ninguno")] = -1.0
        idx = int(probs.argmax())
        especie = self.detector.class_names[idx]
        result = Prediction(especie=especie, especie_confianza=float(probs[idx]))

        loaded = self.breed_models.get(especie)
        if loaded is None:
            return result

        breed_probs = self._probs(loaded, image)
        k = min(top_k, len(loaded.class_names))
        top_i = np.argsort(breed_probs)[::-1][:k]
        result.top_razas = [(loaded.class_names[int(i)], float(breed_probs[i])) for i in top_i]
        result.raza = loaded.class_names[int(top_i[0])]
        result.raza_confianza = float(breed_probs[top_i[0]])
        result.raza_identificada = result.raza_confianza >= loaded.unknown_threshold
        return result

    def describe(self) -> dict:
        info = {
            "framework": "tensorflow",
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
