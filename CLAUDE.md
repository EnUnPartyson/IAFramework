# CLAUDE.md

Este archivo le da contexto a Claude Code para trabajar en este repositorio. Léelo completo antes de proponer cambios.

## Qué es este proyecto

Sistema de clasificación de mascotas en dos etapas, pensado como producto para veterinarias/refugios: identifica si una imagen (o frame de cámara) contiene un perro, un gato o ninguno de los dos, y si es perro o gato, intenta determinar la raza.

**Uso comercial objetivo:** pre-llenado automático de fichas clínicas en veterinarias/refugios al ingresar una mascota nueva (identificación de raza para estimar predisposiciones genéticas).

## Equipo

Somos 2 personas (informáticos, no data scientists). El entregable son scripts de Python, no notebooks de Jupyter.

## Arquitectura del pipeline (3 modelos, cada uno x2 frameworks)

**Requisito no negociable del profesor: los 3 modelos se construyen 2 veces cada uno, una vez en PyTorch y una vez en TensorFlow/Keras.** No es "un framework por modelo" — son 6 implementaciones de entrenamiento en total (3 modelos × 2 frameworks), todas comparadas entre sí. Ver DECISIONS.md.

1. **Modelo 1 — Detector** (perro / gato / ninguno) — PyTorch **y** TensorFlow/Keras
   - Clases desbalanceadas a propósito (35% perro / 35% gato / 30% ninguno). Usar weighted loss, reportar F1 por clase y matriz de confusión, no solo accuracy.
   - Debe soportar modo "forzado" (ignora clase "ninguno", devuelve argmax entre perro/gato), en ambas implementaciones.

2. **Modelo 2 — Raza de perro** — TensorFlow/Keras **y** PyTorch
   - Se ejecuta solo si Modelo 1 predice "perro"

3. **Modelo 3 — Raza de gato** — PyTorch **y** TensorFlow/Keras
   - Se ejecuta solo si Modelo 1 predice "gato"

Todos entrenados **desde cero** (sin transfer learning), con arquitecturas CNN equivalentes entre frameworks para poder comparar TF vs PyTorch de forma justa dentro de cada modelo (mismas capas, mismo optimizador, mismo learning rate, mismo batch size).

### Convención de nombres (dual framework)

Todo script/módulo que dependa de un framework específico lleva el sufijo `_pytorch` o `_tensorflow` en el nombre (ej. `train_detector_pytorch.py`, `train_detector_tensorflow.py`, `model_defs_pytorch.py`, `model_defs_tensorflow.py`). Nunca mezclar imports de `torch` y `tensorflow` en el mismo archivo — los venvs están separados justamente por conflictos de CUDA (ver "Entornos" en README.md), así que un módulo con ambos imports rompería ese aislamiento.

### Qué framework corre en inferencia

`inference/predict_camera.py` sí necesita cargar modelos de ambos frameworks (los pesos ya entrenados, no el código de entrenamiento) para poder correr el pipeline completo o comparar predicciones lado a lado — evaluar si esto requiere tener ambos venvs disponibles al mismo tiempo o si conviene exportar a un formato común (ONNX) para inferencia. Pendiente de decidir (ver DECISIONS.md).

## Reglas de flujo obligatorias (del profesor)

- **Separar obtención de datos del entrenamiento.** `data/download_dataset.py` y `data/prepare_data.py` no entrenan nada; `train/*.py` no descarga nada.
- **Script de consumo independiente y obligatorio.** `inference/predict_camera.py` usa OpenCV para capturar cámara y correr el pipeline completo (Modelo 1 → Modelo 2 o 3 según corresponda).
- **OpenCV es mandatorio** para el manejo de cámara, no opcional.
- No usar Jupyter notebooks como entregable. Si se explora algo en notebook, el código final migra a script `.py`.

## Estructura de carpetas

Ver ARCHITECTURE.md para el detalle completo.

## Dónde se entrena vs dónde se corre inferencia

- **Entrenamiento:** en una instancia EC2 con GPU (AWS). El repo se clona ahí y se corre `bash run_all.sh` (orquestador de punta a punta: venvs + datos + los 6 entrenamientos + comparación). También se puede correr cada paso a mano (ver README.md).
- **Inferencia con cámara:** en la laptop local (la EC2 no tiene webcam). Los pesos entrenados se bajan de la EC2 con `scp` y desde ahí se publican al repo vía Git LFS (ver "Qué NO debe ir a GitHub").

## Qué NO debe ir a GitHub

- Datasets completos (`data/raw/`, `data/processed/`)
- Checkpoints intermedios de entrenamiento y pesos sueltos fuera de `models/`
- Entornos virtuales y `node_modules/`
- Todo esto debe estar en `.gitignore`

**Excepción:** los 6 modelos finales de `models/` **sí** se versionan, mediante **Git LFS**
(ver `.gitattributes`), porque el equipo necesita compartirlos. Quien clone el repo debe
tener `git-lfs` instalado y correr `git lfs install` una vez. Las métricas de `metrics/`
van a git normalmente: son texto e imágenes chicas.

## Convenciones de código

- Python 3.10+
- Type hints en funciones públicas
- Un `requirements.txt` por framework si hay conflictos de CUDA/cuDNN entre TF y PyTorch (ver DECISIONS.md)
- Nombres de scripts en snake_case, autoexplicativos
- Cada script de entrenamiento debe guardar métricas (accuracy, F1 por clase, matriz de confusión) en un archivo aparte, no solo imprimirlas en consola

## Estado actual

Ver DECISIONS.md para decisiones ya tomadas y pendientes.
