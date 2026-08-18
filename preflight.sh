#!/usr/bin/env bash
# Chequeo previo (~15 min) antes de lanzar run_all.sh completo.
#
#   bash preflight.sh
#
# Verifica que ambos frameworks vean la GPU, entrena unas pocas epocas de un modelo real
# en cada uno midiendo el uso efectivo de GPU, y estima cuanto tardaria la corrida entera.
# La idea es detectar en 15 minutos los problemas que si no se descubren a las 10 horas.
set -uo pipefail
cd "$(dirname "$0")"

TORCH_PY="venv-torch/bin/python"
TF_PY="venv-tf/bin/python"
TMP="$(mktemp -d)"
EPOCHS="${PREFLIGHT_EPOCHS:-2}"
TASK="${PREFLIGHT_TASK:-cat_breed}"   # el dataset mas chico: itera rapido
trap 'rm -rf "$TMP"' EXIT

titulo() { echo ""; echo "=============================================="; echo "  $*"; echo "=============================================="; }
ok()   { echo "  [OK]    $*"; }
warn() { echo "  [AVISO] $*"; }
fail() { echo "  [FALLA] $*"; }

# TF no resuelve solo las rutas de CUDA que pip instala (ver run_all.sh)
if [ -x "$TF_PY" ]; then
    libs=$("$TF_PY" - <<'PY' 2>/dev/null || true
import os
try:
    import nvidia
except ImportError:
    raise SystemExit
base = os.path.dirname(nvidia.__file__)
print(":".join(p for p in (os.path.join(base, d, "lib") for d in sorted(os.listdir(base))) if os.path.isdir(p)))
PY
)
    [ -n "$libs" ] && export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---------------------------------------------------------------- 1. entorno
titulo "1/4 · Entorno"

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    ok "GPU: $gpu_name ($gpu_mem)"
else
    fail "no hay nvidia-smi: no se detecta GPU"; exit 1
fi
echo "  vCPU: $(nproc)  ·  RAM: $(free -g | awk '/^Mem:/{print $2}') GB  ·  Disco libre: $(df -h . | awk 'NR==2{print $4}')"

for par in "PyTorch:$TORCH_PY" "TensorFlow:$TF_PY"; do
    nombre="${par%%:*}"; py="${par##*:}"
    [ -x "$py" ] || { fail "$nombre: falta el venv ($py)"; exit 1; }
done

torch_gpu=$("$TORCH_PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
tf_gpu=$("$TF_PY" -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null || echo 0)
[ "$torch_gpu" = "1" ] && ok "PyTorch ve la GPU" || { fail "PyTorch NO ve la GPU"; exit 1; }
[ "$tf_gpu" != "0" ] && ok "TensorFlow ve la GPU" || { fail "TensorFlow NO ve la GPU"; exit 1; }

# ---------------------------------------------------------------- 2. datos
titulo "2/4 · Datos preparados"

falta_datos=0
for t in detector dog_breed cat_breed; do
    d="data/processed/$t/train"
    if [ -d "$d" ]; then
        n_clases=$(find "$d" -mindepth 1 -maxdepth 1 -type d | wc -l)
        n_imgs=$(find "$d" -name '*.jpg' | wc -l)
        echo "  $t: $n_clases clases, $n_imgs imagenes de entrenamiento"
        [ "$n_imgs" -lt 100 ] && { warn "$t tiene muy pocas imagenes"; falta_datos=1; }
    else
        fail "$t: falta $d"; falta_datos=1
    fi
done
if [ -d "data/raw/tsinghua_dogs/low-resolution" ]; then
    ok "Tsinghua Dogs descargado"
else
    warn "Tsinghua Dogs ausente: dog_breed tendra muchos menos datos"
fi
[ "$falta_datos" = "1" ] && { fail "faltan datos; correr 'venv-torch/bin/python data/prepare_data.py'"; exit 1; }

# ---------------------------------------------------------------- 3. entrenamiento medido
titulo "3/4 · Entrenamiento de prueba ($TASK, $EPOCHS epocas por framework)"
echo "  Se mide el uso efectivo de GPU: si queda bajo, la GPU esta esperando datos."

medir() {
    local nombre="$1" py="$2" script="$3"
    local log="$TMP/${nombre}.log" util="$TMP/${nombre}.util"

    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "$util" 2>/dev/null &
    local monitor=$!
    local t0=$(date +%s)
    "$py" "$script" --epochs "$EPOCHS" \
        --model-out "$TMP/${nombre}.model" --metrics-out "$TMP/${nombre}.json" > "$log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    kill "$monitor" 2>/dev/null; wait "$monitor" 2>/dev/null

    if [ $rc -ne 0 ]; then
        fail "$nombre fallo. Ultimas lineas:"; tail -12 "$log" | sed 's/^/      /'; return 1
    fi

    local seg=$((t1 - t0))
    # se descartan los primeros samples: incluyen el arranque, antes de que entrene
    local prom=$(awk 'NR>8 {s+=$1; n++} END {if(n) printf "%.0f", s/n; else print 0}' "$util")
    local pico=$(awk 'NR>8 && $1>m {m=$1} END {print m+0}' "$util")
    echo "  $nombre: ${seg}s total  ·  GPU-Util promedio ${prom}%  ·  pico ${pico}%"
    echo "$seg" > "$TMP/${nombre}.seg"

    if [ "$prom" -lt 40 ]; then
        warn "$nombre usa poca GPU (${prom}%): la carga de datos es el cuello de botella"
    else
        ok "$nombre aprovecha bien la GPU"
    fi
    return 0
}

medir "pytorch" "$TORCH_PY" "train/train_${TASK}_pytorch.py" || exit 1
medir "tensorflow" "$TF_PY" "train/train_${TASK}_tensorflow.py" || exit 1

# ---------------------------------------------------------------- 4. estimacion
titulo "4/4 · Estimacion de la corrida completa"

seg_pt=$(cat "$TMP/pytorch.seg" 2>/dev/null || echo 0)
seg_tf=$(cat "$TMP/tensorflow.seg" 2>/dev/null || echo 0)
python3 - "$seg_pt" "$seg_tf" "$EPOCHS" <<'PY'
import sys
seg_pt, seg_tf, epochs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
# por epoca, descontando ~20s de arranque de cada corrida
ep_pt = max(1, (seg_pt - 20) / epochs)
ep_tf = max(1, (seg_tf - 20) / epochs)
# cat_breed es el dataset mas chico; los otros escalan por cantidad de imagenes. El detector
# lleva un 0.6 extra porque corre a 128px (vs 160) y sin mixup ni augmentation fuerte.
FACTOR = {"cat_breed": 1.0, "dog_breed": 12.0, "detector": 20.0 * 0.6}
EPOCAS = {"cat_breed": 80, "dog_breed": 80, "detector": 30}
total = 0.0
print(f"  {'modelo':<12}{'PyTorch':>12}{'TensorFlow':>14}")
print("  " + "-" * 38)
for tarea, factor in FACTOR.items():
    h_pt = ep_pt * factor * EPOCAS[tarea] / 3600
    h_tf = ep_tf * factor * EPOCAS[tarea] / 3600
    total += h_pt + h_tf
    print(f"  {tarea:<12}{h_pt:>11.1f}h{h_tf:>13.1f}h")
print("  " + "-" * 38)
print(f"  {'TOTAL':<12}{total:>25.1f}h  (cota alta: el early stopping suele cortar antes)")
print()
print("  Nota: es una extrapolacion desde el modelo mas chico; sirve para el orden de")
print("  magnitud, no como promesa. RUN_TUNE=1 suma aparte la busqueda de Optuna.")
PY

titulo "Listo"
echo "  Si todo dio OK y el tiempo estimado es aceptable:"
echo ""
echo "    tmux new -s full"
echo "    RUN_TUNE=1 TUNE_TRIALS=12 SKIP_SETUP=1 SKIP_DATA=1 bash run_all.sh"
echo ""
