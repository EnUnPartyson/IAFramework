# DECISIONS.md

Registro de decisiones tomadas durante el proyecto y pendientes por resolver. Actualizar a medida que se avanza — esto ayuda a Claude Code a no proponer cosas ya descartadas.

## Decisiones tomadas

| Fecha | Decisión | Razón |
|---|---|---|
| 2026-07-23 | **(SUPERA las 3 filas siguientes) Cada uno de los 3 modelos se construye 2 veces: una en PyTorch y otra en TensorFlow/Keras.** No es "un framework por modelo" — son 6 pipelines de entrenamiento en total. Requisito no negociable del profesor | Se aclaró que la comparación TF vs PyTorch tiene que darse dentro de cada modelo, no solo entre el Modelo 2 y el Modelo 3 como se había planteado originalmente |
| ~~2026 (superada)~~ | ~~Detector (Modelo 1) solo en PyTorch~~ | Ver decisión de arriba: ahora también lleva versión TensorFlow |
| ~~2026 (superada)~~ | ~~Raza de perro (Modelo 2) solo en TensorFlow/Keras~~ | Ver decisión de arriba: ahora también lleva versión PyTorch |
| ~~2026 (superada)~~ | ~~Raza de gato (Modelo 3) solo en PyTorch~~ | Ver decisión de arriba: ahora también lleva versión TensorFlow |
| | Entrenamiento desde cero, sin transfer learning | Requisito del profesor |
| | Entrenamiento en AWS EC2, inferencia local | EC2 no tiene cámara |
| | OpenCV obligatorio para consumo de cámara | Requisito explícito del profesor (no negociable) |
| 2026-07-23 | Dataset perro/gato (Modelo 1): "Kaggle Cats and Dogs" vía mirror directo de Microsoft (sin credenciales) | Es el dataset que ya proponía ARCHITECTURE.md; al no compartir fuente con Oxford-IIIT Pet (reservado para Modelo 3) se evita data leakage entre el detector y el futuro clasificador de raza de gato |
| 2026-07-23 | Dataset clase "ninguno" (Modelo 1): Food101 + STL10 (sin "cat"/"dog") + Places365 (sin categorías tipo kennel/pet/veterinaria), cuota pareja (1/3 cada una) | Usar solo Food101 le enseñaba al modelo "comida vs mascota" en vez de "hay o no un perro/gato" (shortcut learning). STL10 aporta otros animales reales (pájaro, ciervo, caballo, mono); Places365 aporta paisajes/escenas (playa, montaña, bosque, calle). La cuota pareja evita que la fuente más grande (Food101, ~75k imgs) domine el sampling y diluya la diversidad. Se descartaron CIFAR-10/100 por ser 32x32 (darían un atajo trivial por nitidez) |
| 2026-07-23 | Proporción de desbalance Modelo 1: perro 35% / gato 35% / ninguno 30% | Valor por defecto propuesto en ARCHITECTURE.md; configurable en `PROPORTIONS` al inicio de `data/prepare_data.py` |
| 2026-07-23 | Mitigación de desbalance: weighted `CrossEntropyLoss` (pesos por frecuencia inversa) | Simple y suficiente para un desbalance moderado (35/35/30); no se combina con oversampling para no duplicar imágenes de entrenamiento |
| 2026-07-23 | Tamaño de imagen: 128x128, split 70/15/15 train/val/test, CNN de 4 bloques conv entrenada desde cero | Balance entre tiempo de entrenamiento razonable en EC2 (g4dn.xlarge) y calidad suficiente para distinguir perro/gato/ninguno |
| 2026-07-23 | Búsqueda de hiperparámetros con Optuna (`train/tune_detector_pytorch.py`), separada del entrenamiento final (`train/train_detector_pytorch.py`) | Cada trial entrena pocas épocas (8) para explorar lr/batch_size/dropout/weight_decay rápido; una vez encontrados los mejores valores se corre el entrenamiento completo con esos parámetros vía flags de `train_detector_pytorch.py` |
| 2026-07-23 | Sumar COCO val2017 (imágenes con "dog"/"cat" anotado) a perro/gato, además de Cats&Dogs | Cats&Dogs son fotos de la mascota en primer plano (estilo "foto de producto"); "ninguno" (paisajes de Places365) son fotos de escena amplia sin sujeto. Sin ejemplos de mascota *dentro* de una escena, el modelo podía aprender a distinguir clases por estilo de composición de la foto en vez de por el animal (shortcut learning). Se excluyen imágenes con dog+cat simultáneos (ambiguas para un clasificador de una sola etiqueta). Parseo el JSON de anotaciones directo (sin `pycocotools`, para no depender de un build nativo en Windows) |
| 2026-07-23 | `RandomResizedCrop(scale=(0.5,1.0))` en vez de `Resize` fijo en el transform de train | Refuerza que el modelo tolere al animal en distintos tamaños/posiciones dentro del cuadro, más parecido a cómo lo va a ver la cámara real que a una foto centrada |
| 2026-07-23 | Diagnóstico Grad-CAM (`train/gradcam_detector_pytorch.py`) sobre imágenes de validación, corre después de entrenar | Verifica visualmente que el modelo mira al animal y no al fondo — evidencia concreta para el informe de que no hay shortcut learning por composición de escena |
| 2026-07-23 | Renombrados todos los módulos dependientes de un framework con sufijo `_pytorch`/`_tensorflow` (`train_detector_pytorch.py`, `model_defs_pytorch.py`, `transforms_pytorch.py`, `tune_detector_pytorch.py`, `gradcam_detector_pytorch.py`) | Consecuencia directa del requisito dual-framework: hace falta distinguir a simple vista qué archivo es de qué framework, y ningún archivo debe mezclar imports de `torch` y `tensorflow` (los venvs están separados por conflictos de CUDA) |
| 2026-07-23 | Motores de entrenamiento compartidos (`train/common_pytorch.py` y `train/common_tensorflow.py`); los 6 scripts `train_*_{framework}.py` son wrappers finos | Con 6 pipelines, duplicar el loop de entrenamiento 6 veces garantizaba divergencias; un motor por framework mantiene los hiperparámetros y el flujo idénticos entre modelos, y la equivalencia entre frameworks queda en 2 archivos revisables lado a lado |
| 2026-07-23 | Una sola arquitectura `SimpleCNN` para los 3 modelos (solo cambia `num_classes`), definida en `model_defs_pytorch.py` y espejada capa a capa en `model_defs_tensorflow.py` | La comparación TF vs PyTorch exige arquitecturas idénticas; tener una sola definición por framework (en vez de 3) hace la equivalencia verificable de un vistazo |
| 2026-07-23 | Dataset raza perro (Modelo 2): Stanford Dogs, top-15 razas por cantidad de imágenes (config `DOG_BREEDS_TOP_N` en `prepare_data.py`) | ARCHITECTURE.md ya sugería reducir de 120 a 15-20 razas; entrenar desde cero una CNN chica con 120 clases y ~150 imágenes por clase no es viable. Top-15 por cantidad de imágenes maximiza los datos por clase. Descarga directa de vision.stanford.edu sin credenciales |
| 2026-07-23 | Dataset raza gato (Modelo 3): Oxford-IIIT Pet, las 12 razas de gato (archivos con inicial mayúscula = gato) | Única fuente estándar con descarga directa sin credenciales; el Cat Breeds Dataset de Kaggle requeriría API key, lo que rompería el objetivo de "clonar y correr". ~200 imágenes por raza: dataset chico, esperar accuracy menor que en perros y documentarlo en el informe |
| 2026-07-23 | Selección de modelo, early stopping y scheduler monitorean `val_accuracy` en ambos frameworks | Keras no calcula F1 por época de forma nativa; monitorear F1 en PyTorch y accuracy en TF haría la comparación asimétrica. Se usa accuracy (idéntico en ambos) para las decisiones durante el entrenamiento; F1 por clase y matriz de confusión se siguen reportando sobre test |
| 2026-07-23 | Mejoras aplicadas a los 6 pipelines: `ReduceLROnPlateau` (factor 0.5, paciencia 2) + early stopping (paciencia 6) + `weight_decay` 1e-4 por defecto | Primeras dos mejoras de MEJORAS_FUTURAS.md: mejoran el resultado de CNNs desde cero y evitan sobreajuste/cómputo desperdiciado. Idénticas en ambos frameworks para no romper la comparación |
| 2026-07-23 | Pesos TF en formato `.keras` (no `.h5`) | Formato nativo de Keras 3 (el que trae TF actual); `.h5` es legacy. `.gitignore` ya cubría ambos |
| 2026-07-23 | Orquestador `run_all.sh` (bash, EC2 Ubuntu): crea venvs, instala, descarga, prepara y entrena los 6 modelos en orden, con flags `SKIP_SETUP`/`SKIP_DATA`/`RUN_TUNE` | Objetivo "clonar y correr" en EC2 limpia sin tocar código. Bash y no Python porque debe alternar entre los dos venvs (un proceso Python no puede cambiar de venv a mitad de ejecución). `.gitattributes` fuerza LF en `*.sh` para que no se rompa al clonar desde Windows |
| 2026-07-23 | En Linux se instala torch desde PyPI directo (sin `--index-url` de pytorch.org) | Los wheels de PyPI para Linux ya traen CUDA; el índice cu130 es necesario solo en Windows. Además `--index-url` apuntando solo a pytorch.org rompía la instalación del resto de los paquetes (sklearn, optuna) que no existen en ese índice |
| 2026-07-29 | Smoke test completo ejecutado en local (CPU, datos sintéticos): los 6 pipelines de entrenamiento + Grad-CAM + Optuna + comparador corren de punta a punta (torch 2.13 / TF 2.21) | Verificación real antes de gastar en EC2. Se detectaron y corrigieron 2 bugs: (1) Grad-CAM usaba `register_full_backward_hook`, incompatible con los ReLU inplace — se cambió a hook sobre el tensor de activación; (2) el `.keras` y `save_weights` de un modelo compilado incluyen el estado de Adam (3x el tamaño), inflando la métrica de tamaño de pesos — ahora se mide sobre un modelo fresco sin compilar. Bonus: pesos PyTorch 17.50MB vs TF 17.55MB confirma empíricamente la equivalencia de arquitecturas |
| 2026-08-13 | Los resultados de Optuna se aplican vía `--hparams-from metrics/detector_best_hparams.json`, y `run_all.sh` con `RUN_TUNE=1` se lo pasa a **ambas** versiones del detector (PyTorch y TF) | Antes la búsqueda corría y guardaba el JSON, pero el entrenamiento lo ignoraba y usaba los defaults: el tuning no servía para nada salvo copiar valores a mano. Se pasa a los dos frameworks porque tunear solo PyTorch sesgaría la comparación. Precedencia: valor explícito en CLI > JSON > default (los args tuneables tienen default `None` para poder distinguir "no lo pasó" de "lo pasó igual al default") |
| 2026-08-13 | Fases largas (instalación de dependencias ~3GB, preparación de ~100k imágenes) ahora imprimen progreso; `run_all.sh` verifica al final del setup que torch y TF vean la GPU | Sin salida visible parecían colgadas por decenas de minutos. La verificación temprana de CUDA evita descubrir un problema de GPU recién horas después |
| 2026-07-29 | Verificadas las 7 URLs de datasets (responden 200): Microsoft Cats&Dogs, COCO val2017 + anotaciones, Food101, STL10, Places365 (val_256 + filelist), Stanford Dogs, Oxford-IIIT Pet | Ninguna fuente requiere credenciales; la descarga en EC2 no debería tener sorpresas |

## Pendientes por decidir

- [x] ~~Tipo de instancia EC2 definitivo~~ — decidido 2026-07-23: **g4dn.xlarge on-demand** (~$0.53/h). El pipeline completo (~4-6h) cuesta ~$3 de los $200 de crédito; una CPU grande saldría más cara y tardaría días. Disco: 60-80GB gp3. Correr en tmux, hacer STOP (no terminate) al terminar
- [ ] Cantidad de razas a incluir por especie (todas vs subset) — aplica a Modelo 2 y 3, no bloquea al Modelo 1
- [ ] Cómo repartir el trabajo entre los dos integrantes — con 6 pipelines, una opción natural es 1 persona = 1 framework (todas las versiones PyTorch vs todas las TF) en vez de repartir por modelo
- [ ] Nombre final del producto / enfoque comercial específico (veterinaria vs refugio vs app consumidor)
- [ ] `inference/predict_camera.py`: si carga ambas versiones (TF y PyTorch) para comparar en vivo, o solo la elegida como "final" por modelo — implica tener ambos venvs disponibles en la máquina de inferencia o exportar a un formato común (ONNX)

## Pendiente por construir

- [x] ~~Los 6 pipelines de entrenamiento (3 modelos × 2 frameworks)~~ — hecho 2026-07-23
- [x] ~~Datos para razas de perro y gato~~ — hecho 2026-07-23 (Stanford Dogs top-15 / Oxford-IIIT Pet 12 razas)
- [x] ~~`metrics/comparacion_tf_vs_pytorch.json`~~ — hecho 2026-07-23 (`train/compare_frameworks.py`)
- [x] ~~Orquestador de punta a punta~~ — hecho 2026-07-23 (`run_all.sh`)
- [ ] `inference/predict_camera.py` (script de consumo con OpenCV, obligatorio) — único componente grande faltante
- [ ] Versiones TF de tune (Optuna) y Grad-CAM — hoy solo existen para PyTorch; no bloquean el entrenamiento

## Descartado / no se va a hacer

| Decisión descartada | Razón |
|---|---|
| Transfer learning con modelos preentrenados | El profesor pide entrenar desde cero |
| Jupyter notebooks como entregable final | El profesor exige scripts .py |
| Un solo framework para todo | Se pide comparación explícita TF vs PyTorch |
| Un solo framework por modelo (ej. Modelo 1 solo PyTorch) | Cada modelo debe existir en ambos frameworks, no repartirse un framework distinto por modelo |
