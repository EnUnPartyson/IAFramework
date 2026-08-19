"""Descarga los datasets EXTRA del modo pro. No entrena nada (regla del profesor).

Los datasets base (Cats&Dogs, COCO, Food101, STL10, Places365, Stanford, Tsinghua, Oxford)
se descargan con download_dataset.py y no cambian. Este script suma dos fuentes nuevas que
viven en Kaggle:

  1. ma7555/cat-breeds-dataset  (~67 razas de gato, ~126k imagenes de PetFinder)
     -> el gato era el punto debil de la v1: Oxford solo aporta ~200 imgs/raza
  2. andrewmvd/animal-faces     (AFHQ: ~16k imagenes 512px de perro / gato / salvaje)
     -> "salvaje" (zorros, leones, tigres...) va a la clase "ninguno" del detector como
        negativo dificil: animales con cara de mascota que NO son perro ni gato

CREDENCIALES (unico paso manual, ver README):
  Kaggle exige una API key gratuita. En https://www.kaggle.com -> cuenta -> Settings ->
  API -> "Create New Token" se descarga kaggle.json; ponerlo en ~/.kaggle/kaggle.json
  (o exportar KAGGLE_USERNAME y KAGGLE_KEY). Sin eso este script falla con instrucciones.

Idempotente: si la carpeta destino ya tiene contenido, no re-descarga.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"

DATASETS = {
    # id de kaggle -> carpeta destino bajo data/raw/
    "ma7555/cat-breeds-dataset": "cat_breeds_extra",
    "andrewmvd/animal-faces": "afhq",
}


def _credenciales_ok() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _tiene_contenido(destino: Path) -> bool:
    return destino.exists() and any(destino.rglob("*.jpg"))


def descargar(dataset_id: str, carpeta: str) -> None:
    destino = RAW_DIR / carpeta
    if _tiene_contenido(destino):
        print(f"[pro] {carpeta} ya descargado, se omite")
        return

    destino.mkdir(parents=True, exist_ok=True)
    print(f"[pro] Descargando {dataset_id} -> {destino} ...")
    # el CLI de kaggle viene con el paquete pip "kaggle" (esta en requirements-torch del pro)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", dataset_id, "-p", str(destino)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kaggle fallo para {dataset_id}:\n{result.stderr}")

    for zip_path in destino.glob("*.zip"):
        print(f"[pro] Extrayendo {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(destino)
        zip_path.unlink()
    print(f"[pro] {carpeta}: {sum(1 for _ in destino.rglob('*.jpg'))} jpg extraidos")


def main() -> None:
    if not _credenciales_ok():
        print(
            "ERROR: faltan credenciales de Kaggle.\n"
            "  1. Cuenta gratuita en https://www.kaggle.com\n"
            "  2. Settings -> API -> 'Create New Token' (descarga kaggle.json)\n"
            "  3. mkdir -p ~/.kaggle && mv ~/Descargas/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
            "     (o exportar KAGGLE_USERNAME y KAGGLE_KEY)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if shutil.which("kaggle") is None:
        # el modulo puede estar instalado aunque el ejecutable no este en PATH; se prueba
        probe = subprocess.run([sys.executable, "-m", "kaggle", "--version"], capture_output=True)
        if probe.returncode != 0:
            print("ERROR: falta el paquete 'kaggle' (pip install kaggle)", file=sys.stderr)
            sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id, carpeta in DATASETS.items():
        descargar(dataset_id, carpeta)
    print("[pro] Descargas del modo pro completas.")


if __name__ == "__main__":
    main()
