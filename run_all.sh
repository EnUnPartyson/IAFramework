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
#   RUN_TUNE=1        correr la busqueda de hiperparametros con Optuna y aplicar el
#                     resultado a las dos versiones del detector
#   TUNE_TRIALS=N     cuantos trials prueba Optuna (default 20; bajar a 5-8 en CPU)
#   SKIP_TORCH=1      no reentrenar los modelos PyTorch (util para rehacer solo los de TF)
#   SKIP_TF=1         no reentrenar los modelos TensorFlow
#   ALLOW_CPU=1       permitir entrenar sin GPU (por defecto aborta: entrenar en CPU sin
#                     querer da resultados no comparables y tarda ~12x mas)
#   TORCH_INDEX_URL   indice extra de pip para wheels de torch (por defecto no se usa:
#                     en Linux los wheels de PyPI ya traen CUDA)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
TORCH_PY="venv-torch/bin/python"
TF_PY="venv-tf/bin/python"

log() { echo ""; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

# TensorFlow no resuelve solo las rutas de las librerias CUDA que pip deja en
# site-packages/nvidia/*/lib: sin esto no las encuentra y cae a CPU en silencio.
# PyTorch no lo necesita porque resuelve esas rutas al importarse.
export_tf_cuda_path() {
    local libs
    libs=$("$TF_PY" - <<'PY' 2>/dev/null || true
import os
try:
    import nvidia
except ImportError:
    raise SystemExit
base = os.path.dirname(nvidia.__file__)
paths = [os.path.join(base, d, "lib") for d in sorted(os.listdir(base))]
print(":".join(p for p in paths if os.path.isdir(p)))
PY
)
    if [ -n "$libs" ]; then
        export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}

# Falla temprano si falta la GPU: descubrirlo despues de horas de entrenamiento
# invalida la comparacion TF vs PyTorch (seria GPU contra CPU).
check_gpu() {
    local torch_gpu tf_gpu
    torch_gpu=$("$TORCH_PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
    tf_gpu=$("$TF_PY" -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null || echo 0)
    echo "PyTorch ve GPU: $torch_gpu | TensorFlow ve GPUs: $tf_gpu"

    if [ "${ALLOW_CPU:-0}" = "1" ]; then
        echo "ALLOW_CPU=1: se continua aunque falte GPU."
        return 0
    fi
    if [ "$torch_gpu" != "1" ] || [ "$tf_gpu" = "0" ]; then
        echo "" >&2
        echo "ERROR: algun framework no ve la GPU y los resultados no serian comparables." >&2
        echo "  - Revisar 'nvidia-smi' y que el venv tenga los paquetes nvidia-*." >&2
        echo "  - Para entrenar en CPU a proposito: ALLOW_CPU=1 bash run_all.sh" >&2
        exit 1
    fi
}

if [ "${SKIP_SETUP:-0}" != "1" ]; then
    # sin -q a proposito: torch+CUDA son ~2.5GB y TF ~600MB, sin barra de progreso
    # parece colgado durante 10-25 minutos
    log "Creando venv de PyTorch e instalando dependencias (~2.5GB, puede tardar)"
    [ -d venv-torch ] || "$PYTHON" -m venv venv-torch
    venv-torch/bin/pip install --upgrade pip
    if [ -n "${TORCH_INDEX_URL:-}" ]; then
        venv-torch/bin/pip install -r requirements-torch.txt --extra-index-url "$TORCH_INDEX_URL"
    else
        venv-torch/bin/pip install -r requirements-torch.txt
    fi

    log "Creando venv de TensorFlow e instalando dependencias (~600MB, puede tardar)"
    [ -d venv-tf ] || "$PYTHON" -m venv venv-tf
    venv-tf/bin/pip install --upgrade pip
    venv-tf/bin/pip install -r requirements-tf.txt

fi

log "Verificando que ambos frameworks vean la GPU"
export_tf_cuda_path
check_gpu

if [ "${SKIP_DATA:-0}" != "1" ]; then
    log "Descargando datasets (idempotente: omite lo ya descargado)"
    "$TORCH_PY" data/download_dataset.py

    log "Preparando splits train/val/test de los 3 modelos"
    "$TORCH_PY" data/prepare_data.py
fi

# Si se corre la busqueda, sus resultados alimentan las DOS versiones del detector:
# tunear solo PyTorch sesgaria la comparacion TF vs PyTorch.
HPARAMS_ARG=""
if [ -f metrics/detector_best_hparams.json ]; then
    # ya hay una busqueda previa: reusarla para no invalidar la comparacion si se
    # reentrena solo una mitad (ej. SKIP_TORCH=1)
    HPARAMS_ARG="--hparams-from metrics/detector_best_hparams.json"
fi
if [ "${RUN_TUNE:-0}" = "1" ]; then
    log "Busqueda de hiperparametros del detector (Optuna, ${TUNE_TRIALS:-20} trials)"
    "$TORCH_PY" train/tune_detector_pytorch.py --trials "${TUNE_TRIALS:-20}"
    HPARAMS_ARG="--hparams-from metrics/detector_best_hparams.json"
fi

run_torch() { [ "${SKIP_TORCH:-0}" != "1" ]; }
run_tf() { [ "${SKIP_TF:-0}" != "1" ]; }

if run_torch; then
    log "Modelo 1 (detector) - PyTorch"
    "$TORCH_PY" train/train_detector_pytorch.py $HPARAMS_ARG
fi

if run_tf; then
    log "Modelo 1 (detector) - TensorFlow"
    "$TF_PY" train/train_detector_tensorflow.py $HPARAMS_ARG
fi

if run_torch; then
    log "Modelo 2 (raza perro) - PyTorch"
    "$TORCH_PY" train/train_dog_breed_pytorch.py
fi

if run_tf; then
    log "Modelo 2 (raza perro) - TensorFlow"
    "$TF_PY" train/train_dog_breed_tensorflow.py
fi

if run_torch; then
    log "Modelo 3 (raza gato) - PyTorch"
    "$TORCH_PY" train/train_cat_breed_pytorch.py
fi

if run_tf; then
    log "Modelo 3 (raza gato) - TensorFlow"
    "$TF_PY" train/train_cat_breed_tensorflow.py
fi

log "Diagnostico Grad-CAM del detector (no critico)"
"$TORCH_PY" train/gradcam_detector_pytorch.py || echo "Grad-CAM fallo, se continua igual"

log "Comparacion TF vs PyTorch"
"$TORCH_PY" train/compare_frameworks.py

log "Pipeline completo. Modelos en models/, metricas en metrics/"
ls -lh models/ || true
