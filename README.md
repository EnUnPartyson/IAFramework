# Clasificador de Mascotas (Detector + Raza)

Sistema en dos etapas: detecta si una imagen/frame de cámara contiene un perro, un gato o ninguno, y luego identifica la raza. Pensado como herramienta de pre-llenado de fichas clínicas para veterinarias y refugios.

**Requisito central:** cada uno de los 3 modelos se construye 2 veces (PyTorch y TensorFlow/Keras) — 6 pipelines de entrenamiento comparados entre sí. Ver `CLAUDE.md` y `DECISIONS.md`.

## Entrenamiento en EC2 (camino recomendado)

### 1. Lanzar la instancia

| Parámetro | Valor |
|---|---|
| Tipo | `g4dn.xlarge` (GPU T4, ~USD 0.53/h) |
| AMI | **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** — ya trae el driver NVIDIA |
| Disco | 80 GB gp3 (el default de 8 GB no alcanza: ~12 GB de datasets crudos + procesados + venvs) |
| Security group | SSH (puerto 22) desde tu IP |
| Key pair | crear/usar uno `.pem` y guardarlo local |

### 2. Subir el código

El proyecto pesa ~200 KB (solo código; datos y pesos se generan en la instancia).

```bash
# Desde la carpeta que CONTIENE IAFramework, en la maquina local:
scp -i key.pem -r IAFramework ubuntu@<ip-ec2>:~/proyecto
```

Si el repo ya está en GitHub, `git clone <url> proyecto` en la instancia es equivalente.

### 3. Correr el pipeline

```bash
ssh -i key.pem ubuntu@<ip-ec2>
sudo apt update && sudo apt install -y python3-venv   # por si la AMI no lo trae
cd ~/proyecto
tmux new -s train        # para que siga corriendo si se corta el SSH
bash run_all.sh
# Ctrl+B, luego D  -> te despegas; "tmux attach -t train" para volver
```

Eso crea los dos venvs, instala dependencias, descarga los datasets, prepara los splits, entrena los 6 modelos (con early stopping y scheduler de LR) y deja:

- `models/` — los 6 archivos de pesos (`*_pytorch.pt` / `*_tensorflow.keras`)
- `metrics/` — métricas por modelo/framework + `comparacion_tf_vs_pytorch.json` + matrices de confusión en PNG + Grad-CAM del detector

**No se necesitan credenciales**: todos los datasets se descargan de URLs públicas (Microsoft, COCO, torchvision, Stanford, Oxford). La descarga total ronda los ~12GB, prever disco acorde (~30GB libres contando los datos procesados).

Variables opcionales de `run_all.sh`:

| Variable | Efecto |
|---|---|
| `SKIP_SETUP=1` | no recrear venvs ni reinstalar dependencias |
| `SKIP_DATA=1` | no descargar/preparar datos (ya preparados) |
| `RUN_TUNE=1` | correr búsqueda de hiperparámetros (Optuna) y aplicar el resultado a las dos versiones del detector |
| `TUNE_TRIALS=N` | cuántos trials prueba Optuna (default 20) |
| `SKIP_TORCH=1` / `SKIP_TF=1` | reentrenar solo la mitad del otro framework |
| `ALLOW_CPU=1` | permitir entrenar sin GPU (por defecto **aborta**: entrenar en CPU sin querer da resultados no comparables y tarda ~12× más) |

Los scripts de descarga son idempotentes: si algo se corta, volver a correr `bash run_all.sh` retoma sin re-descargar lo que ya está.

### 4. Prueba rápida antes de la corrida completa (recomendado)

Antes de largar las ~5 horas, validar que GPU y datos funcionan con el modelo más chico:

```bash
nvidia-smi                                          # la GPU se ve?
venv-torch/bin/python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
venv-torch/bin/python train/train_cat_breed_pytorch.py --epochs 2
```

Si eso termina bien (~5 min), `bash run_all.sh` va a andar.

### 5. Al terminar: bajar resultados y APAGAR

```bash
# Desde la maquina local:
scp -i key.pem -r "ubuntu@<ip-ec2>:~/proyecto/models"  ./
scp -i key.pem -r "ubuntu@<ip-ec2>:~/proyecto/metrics" ./
```

Después **STOP** de la instancia desde la consola (no *terminate* si querés conservar el disco).
Una instancia detenida solo cobra el disco (~USD 5/mes por 80 GB); olvidarla prendida es el único
riesgo real para el presupuesto. Conviene además crear un AWS Budget con acción automática de
"stop EC2" en ~USD 30.

## Pasos manuales (equivalentes a run_all.sh)

```bash
# Entorno PyTorch (en Linux los wheels de PyPI ya traen CUDA)
python3 -m venv venv-torch && venv-torch/bin/pip install -r requirements-torch.txt
# En Windows para GPU: agregar --extra-index-url https://download.pytorch.org/whl/cu130

# Entorno TensorFlow
python3 -m venv venv-tf && venv-tf/bin/pip install -r requirements-tf.txt

# Datos (venv-torch)
venv-torch/bin/python data/download_dataset.py
venv-torch/bin/python data/prepare_data.py

# Entrenamiento: cada modelo 2 veces, cada script con su venv
venv-torch/bin/python train/train_detector_pytorch.py
venv-tf/bin/python    train/train_detector_tensorflow.py
venv-torch/bin/python train/train_dog_breed_pytorch.py
venv-tf/bin/python    train/train_dog_breed_tensorflow.py
venv-torch/bin/python train/train_cat_breed_pytorch.py
venv-tf/bin/python    train/train_cat_breed_tensorflow.py

# Comparacion final
venv-torch/bin/python train/compare_frameworks.py
```

## Usar los modelos en la máquina local

Requiere webcam, por eso no corre en EC2. Los modelos vienen con el repo
(`git clone` los trae), así que solo hace falta el entorno.

### Preparar el entorno (una sola vez)

Para inferencia alcanza con PyTorch en versión **CPU**: son ~200 MB en vez de 2,5 GB, y
clasificar un frame lleva milisegundos igual.

```bat
python -m venv venv-torch
venv-torch\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv-torch\Scripts\python.exe -m pip install fastapi uvicorn[standard] python-multipart pillow numpy opencv-python
```

### Script de cámara con OpenCV

```bat
venv-torch\Scripts\python.exe inference\predict_camera.py
```

Teclas: `q` salir · `espacio` congelar · `g` guardar el frame.

### API para la app del celular

```bat
venv-torch\Scripts\python.exe inference\server.py
```

Imprime al arrancar la IP a usar desde el celular. Documentación interactiva para probarla
desde el navegador en `http://localhost:8000/docs`.

> **Rutas según la consola.** Los ejemplos de arriba son para `cmd.exe` (barras invertidas).
> En **PowerShell** hay que anteponer `.\`: `.\venv-torch\Scripts\python.exe ...`.
> En **Git Bash** funcionan las barras normales: `venv-torch/Scripts/python.exe ...`.
> Los bloques con `venv-torch/bin/python` de más arriba son para la EC2 (Linux).

La app Ionic se corre aparte: ver [`app/README.md`](app/README.md).

## Documentación

- `CLAUDE.md` — contexto para Claude Code
- `ARCHITECTURE.md` — diseño del pipeline y estructura de carpetas
- `DECISIONS.md` — registro de decisiones tomadas y pendientes
- `MEJORAS_FUTURAS.md` — mejoras posibles para modelos y datos, no bloqueantes

## Equipo

2 integrantes — repartir por modelo o por etapa del pipeline (ver DECISIONS.md, pendiente de definir).
