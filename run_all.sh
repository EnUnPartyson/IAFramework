#!/usr/bin/env bash
# Orquestador de punta a punta para EC2 (Ubuntu). Desde un clon limpio del repo:
#
#   bash run_all.sh
#
# deja los 6 modelos entrenados en models/ y todas las metricas en metrics/.
# No requiere credenciales: todos los datasets se descargan de URLs publicas.
#
# Variables opcionales:
#   SKIP_SETUP=1      no crear venvs ni instalar dependencias (ya instaladas)
#   SKIP_DATA=1       no descargar ni preparar datos (ya preparados)
#   RUN_TUNE=1        correr la busqueda de hiperparametros con Optuna antes del detector
#   TORCH_INDEX_URL   indice extra de pip para wheels de torch (por defecto no se usa:
#                     en Linux los wheels de PyPI ya traen CUDA)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
TORCH_PY="venv-torch/bin/python"
TF_PY="venv-tf/bin/python"

log() { echo ""; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

if [ "${SKIP_SETUP:-0}" != "1" ]; then
    log "Creando venvs e instalando dependencias"
    [ -d venv-torch ] || "$PYTHON" -m venv venv-torch
    venv-torch/bin/pip install --upgrade pip -q
    if [ -n "${TORCH_INDEX_URL:-}" ]; then
        venv-torch/bin/pip install -r requirements-torch.txt --extra-index-url "$TORCH_INDEX_URL" -q
    else
        venv-torch/bin/pip install -r requirements-torch.txt -q
    fi

    [ -d venv-tf ] || "$PYTHON" -m venv venv-tf
    venv-tf/bin/pip install --upgrade pip -q
    venv-tf/bin/pip install -r requirements-tf.txt -q
fi

if [ "${SKIP_DATA:-0}" != "1" ]; then
    log "Descargando datasets (idempotente: omite lo ya descargado)"
    "$TORCH_PY" data/download_dataset.py

    log "Preparando splits train/val/test de los 3 modelos"
    "$TORCH_PY" data/prepare_data.py
fi

if [ "${RUN_TUNE:-0}" = "1" ]; then
    log "Busqueda de hiperparametros del detector (Optuna)"
    "$TORCH_PY" train/tune_detector_pytorch.py
fi

log "Modelo 1 (detector) - PyTorch"
"$TORCH_PY" train/train_detector_pytorch.py

log "Modelo 1 (detector) - TensorFlow"
"$TF_PY" train/train_detector_tensorflow.py

log "Modelo 2 (raza perro) - PyTorch"
"$TORCH_PY" train/train_dog_breed_pytorch.py

log "Modelo 2 (raza perro) - TensorFlow"
"$TF_PY" train/train_dog_breed_tensorflow.py

log "Modelo 3 (raza gato) - PyTorch"
"$TORCH_PY" train/train_cat_breed_pytorch.py

log "Modelo 3 (raza gato) - TensorFlow"
"$TF_PY" train/train_cat_breed_tensorflow.py

log "Diagnostico Grad-CAM del detector (no critico)"
"$TORCH_PY" train/gradcam_detector_pytorch.py || echo "Grad-CAM fallo, se continua igual"

log "Comparacion TF vs PyTorch"
"$TORCH_PY" train/compare_frameworks.py

log "Pipeline completo. Modelos en models/, metricas en metrics/"
ls -lh models/ || true
