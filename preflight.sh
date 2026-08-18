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
# 6 epocas y no 2: con el dataset chico, 2 epocas duran ~15s y la medicion de GPU queda
# dominada por el arranque (init de CUDA, escaneo del dataset) en vez del entrenamiento
EPOCHS="${PREFLIGHT_EPOCHS:-6}"
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
libre_gb=$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')
echo "  vCPU: $(nproc)  ·  RAM: $(free -g | awk '/^Mem:/{print $2}') GB  ·  Disco libre: ${libre_gb} GB"
if [ "${libre_gb:-0}" -lt 10 ]; then
    fail "menos de 10 GB libres: el entrenamiento puede quedarse sin espacio"; exit 1
elif [ "${libre_gb:-0}" -lt 25 ]; then
    warn "quedan ${libre_gb} GB. Alcanza para entrenar, pero si hay que re-preparar datos puede faltar"
fi

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
        echo "$n_imgs" > "$TMP/count_$t"   # lo usa la estimacion final
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
if [ "$TASK" = "cat_breed" ]; then
    echo "  OJO: cat_breed tiene ~1.7k imagenes, 20x menos que los otros. El GPU-Util sale"
    echo "  PESIMISTA porque el arranque pesa mucho. Para un numero representativo del"
    echo "  trabajo real:  PREFLIGHT_TASK=dog_breed PREFLIGHT_EPOCHS=2 bash preflight.sh"
fi

medir() {
    local nombre="$1" py="$2" script="$3"; shift 3
    local log="$TMP/${nombre}.log" util="$TMP/${nombre}.util"
    # cada framework exige su extension: Keras rechaza cualquier cosa que no sea .keras/.h5
    local ext=".pt"; case "$script" in *tensorflow*) ext=".keras";; esac

    echo "  -> $nombre en curso... (log: $log)"
    echo "     seguirlo en otra terminal con:  tail -f $log"

    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > "$util" 2>/dev/null &
    local monitor=$!
    local t0=$(date +%s)
    "$py" "$script" --epochs "$EPOCHS" "$@" \
        --model-out "$TMP/${nombre}${ext}" --metrics-out "$TMP/${nombre}.json" > "$log" 2>&1
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

# ---------------------------------------------------------- 3b. precision mixta
titulo "3b/4 · Precision mixta (float16)"
echo "  Repite lo mismo con --amp. Solo acelera si la GPU esta saturada:"
echo "  si arriba el GPU-Util quedo bajo, el cuello son los datos y esto no cambiara nada."

medir "pytorch-amp" "$TORCH_PY" "train/train_${TASK}_pytorch.py" --amp || warn "AMP fallo en PyTorch"
medir "tensorflow-amp" "$TF_PY" "train/train_${TASK}_tensorflow.py" --amp || warn "AMP fallo en TensorFlow"

for fw in pytorch tensorflow; do
    base=$(cat "$TMP/${fw}.seg" 2>/dev/null || echo 0)
    amp=$(cat "$TMP/${fw}-amp.seg" 2>/dev/null || echo 0)
    if [ "$base" -gt 0 ] && [ "$amp" -gt 0 ]; then
        printf "  %-12s fp32 %4ss  ->  amp %4ss  (%.2fx)\n" "$fw" "$base" "$amp" \
            "$(awk "BEGIN{print $base/$amp}")"
    fi
done
echo "  Si la mejora es menor a ~1.2x no vale la pena: se gana poco y se agrega riesgo numerico."
echo "  Nota: RandomErasing de Keras no soporta float16, asi que --amp no funciona en TF con"
echo "  augmentation fuerte. Usarlo solo en PyTorch romperia la comparacion entre frameworks."

# ---------------------------------------------------------------- 4. estimacion
titulo "4/4 · Estimacion de la corrida completa"

seg_pt=$(cat "$TMP/pytorch.seg" 2>/dev/null || echo 0)
seg_tf=$(cat "$TMP/tensorflow.seg" 2>/dev/null || echo 0)
python3 - "$seg_pt" "$seg_tf" "$EPOCHS" "$TASK" \
    "$(cat "$TMP/count_detector" 2>/dev/null || echo 0)" \
    "$(cat "$TMP/count_dog_breed" 2>/dev/null || echo 0)" \
    "$(cat "$TMP/count_cat_breed" 2>/dev/null || echo 0)" <<'PY'
import sys
seg_pt, seg_tf, epochs, medido = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
IMGS = {"detector": int(sys.argv[5]), "dog_breed": int(sys.argv[6]), "cat_breed": int(sys.argv[7])}

# segundos por epoca del modelo QUE SE MIDIO, descontando el arranque (init de CUDA,
# escaneo del dataset), que no se repite por epoca
ARRANQUE = 25
ep_pt = max(0.5, (seg_pt - ARRANQUE) / epochs)
ep_tf = max(0.5, (seg_tf - ARRANQUE) / epochs)

# se escala por cantidad real de imagenes, no por factores fijos: asi la estimacion vale
# sea cual sea el modelo que se midio (PREFLIGHT_TASK)
EPOCAS = {"detector": 30, "dog_breed": 80, "cat_breed": 80}
# el detector corre a 128px y 4 bloques; las razas a 160px y 5. El costo por imagen escala
# aprox. con el area: (128/160)^2 = 0.64
COSTO_IMG = {"detector": 0.64, "dog_breed": 1.0, "cat_breed": 1.0}
TUNE_EPOCAS = {"detector": 8, "dog_breed": 14, "cat_breed": 14}

base = IMGS[medido] * COSTO_IMG[medido]
if base <= 0:
    print("  No se pudo estimar: faltan los conteos de imagenes.")
    raise SystemExit

print(f"  Medido sobre {medido}: {ep_pt:.0f}s/epoca en PyTorch, {ep_tf:.0f}s/epoca en TensorFlow.\n")
print(f"  {'modelo':<12}{'PyTorch':>12}{'TensorFlow':>14}")
print("  " + "-" * 38)
total = 0.0
for tarea in ("detector", "dog_breed", "cat_breed"):
    escala = (IMGS[tarea] * COSTO_IMG[tarea]) / base
    h_pt = ep_pt * escala * EPOCAS[tarea] / 3600
    h_tf = ep_tf * escala * EPOCAS[tarea] / 3600
    total += h_pt + h_tf
    print(f"  {tarea:<12}{h_pt:>11.1f}h{h_tf:>13.1f}h")
print("  " + "-" * 38)
print(f"  {'entrenamiento':<12}{total:>25.1f}h")

# Optuna corre solo en PyTorch, con menos epocas por trial y con pruning (~35% de ahorro)
tune = sum(
    ep_pt * ((IMGS[t] * COSTO_IMG[t]) / base) * TUNE_EPOCAS[t] * 12 * 0.65 / 3600
    for t in EPOCAS
)
print(f"  {'+ Optuna':<12}{tune:>25.1f}h  (12 trials por modelo, con pruning)")
print(f"  {'TOTAL':<12}{total + tune:>25.1f}h")
print()
print("  Es una cota ALTA: asume que los 3 modelos agotan sus epocas maximas, y el early")
print("  stopping suele cortar bastante antes. Sin RUN_TUNE=1 se ahorra la parte de Optuna.")
PY

titulo "Listo"
echo "  Si todo dio OK y el tiempo estimado es aceptable:"
echo ""
echo "    tmux new -s full"
echo "    RUN_TUNE=1 TUNE_TRIALS=12 SKIP_SETUP=1 SKIP_DATA=1 bash run_all.sh"
echo ""
echo "  Agregar USE_AMP=1 adelante solo si la precision mixta dio una mejora clara arriba."
echo ""
