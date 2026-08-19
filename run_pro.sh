#!/usr/bin/env bash
# Orquestador del modo PRO para EC2 (Ubuntu). Solo PyTorch, transfer learning permitido.
#
#   bash run_pro.sh
#
# Deja detector_pro_pytorch.pt, dog_breed_pro_pytorch.pt y cat_breed_pro_pytorch.pt en
# models/, sus metricas *_pro_metrics.json en metrics/ y la comparacion base vs pro.
#
# CREDENCIALES: los datasets extra viven en Kaggle y requieren una API key gratuita en
# ~/.kaggle/kaggle.json (ver data/download_dataset_pro.py o el README). Los datasets base
# siguen siendo de URLs publicas.
#
# Variables opcionales:
#   SKIP_SETUP=1     no crear venv ni instalar dependencias
#   SKIP_DATA=1      no descargar ni preparar datos (ya preparados)
#   ARCH=<nombre>    backbone para los 3 modelos (default convnext_tiny;
#                    tambien: resnet18, resnet50, efficientnet_v2_s)
#   ALLOW_CPU=1      permitir entrenar sin GPU (por defecto aborta)
#   TORCH_INDEX_URL  indice extra de pip para wheels de torch
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
TORCH_PY="venv-torch/bin/python"
ARCH=${ARCH:-convnext_tiny}

log() { echo ""; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

if [ -z "${SKIP_SETUP:-}" ]; then
    log "Creando venv de PyTorch e instalando dependencias"
    [ -d venv-torch ] || "$PYTHON" -m venv venv-torch
    venv-torch/bin/pip install --upgrade pip
    if [ -n "${TORCH_INDEX_URL:-}" ]; then
        venv-torch/bin/pip install -r requirements-torch.txt --extra-index-url "$TORCH_INDEX_URL"
    else
        venv-torch/bin/pip install -r requirements-torch.txt
    fi
fi

if [ -z "${ALLOW_CPU:-}" ]; then
    gpu=$("$TORCH_PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
    if [ "$gpu" != "1" ]; then
        echo "ERROR: PyTorch no ve GPU. Entrenar el modo pro en CPU tarda un dia entero." >&2
        echo "  - Revisar 'nvidia-smi' y los drivers de la instancia." >&2
        echo "  - Para forzar CPU igualmente: ALLOW_CPU=1 bash run_pro.sh" >&2
        exit 1
    fi
    log "GPU verificada"
fi

if [ -z "${SKIP_DATA:-}" ]; then
    log "Descargando datasets base (publicos, idempotente)"
    "$TORCH_PY" data/download_dataset.py

    log "Descargando datasets extra del pro (Kaggle, requiere ~/.kaggle/kaggle.json)"
    "$TORCH_PY" data/download_dataset_pro.py

    log "Preparando splits del modo pro (no toca los de la v1)"
    "$TORCH_PY" data/prepare_data_pro.py
fi

for task in detector dog_breed cat_breed; do
    log "Entrenando ${task}_pro (arch=$ARCH, transfer learning)"
    "$TORCH_PY" train/train_pro_pytorch.py --task "$task" --arch "$ARCH"
done

log "Comparacion base vs pro"
"$TORCH_PY" train/compare_pro.py

log "Robustez e impacto del TTA sobre los modelos pro"
# con --pro evalua <task>_pro_pytorch.pt sobre processed/<task>_pro; si algo falta no aborta
"$TORCH_PY" train/evaluar_robustez.py --pro --task detector --task dog_breed --task cat_breed || true

log "Modo pro completo. Modelos en models/*_pro_pytorch.pt, metricas en metrics/*_pro_*"
