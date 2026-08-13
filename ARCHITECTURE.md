# ARCHITECTURE.md

## Visión general del pipeline

Requisito no negociable: **cada uno de los 3 modelos se entrena 2 veces, una en PyTorch y otra en TensorFlow/Keras.** No hay "el framework de este modelo" — hay 6 pipelines de entrenamiento (3 modelos × 2 frameworks), comparados entre sí dentro de cada modelo.

```
                    ┌───────────────────────────┐
   Imagen/Frame ───▶│      Modelo 1: Detector    │
                    │  (PyTorch  Y  TensorFlow)  │
                    │    perro/gato/ninguno      │
                    └──────────────┬─────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                                    ▼
         "es perro"                             "es gato"
                 │                                    │
                 ▼                                    ▼
   ┌──────────────────────────┐        ┌──────────────────────────┐
   │  Modelo 2: Raza Perro     │        │  Modelo 3: Raza Gato     │
   │  (TensorFlow  Y  PyTorch) │        │  (PyTorch  Y  TensorFlow) │
   └──────────────────────────┘        └──────────────────────────┘
```

Para inferencia (`inference/predict_camera.py`) se elige un framework por corrida (o se corren ambos y se comparan lado a lado) — el pipeline conceptual es el mismo, cambia solo qué par de pesos (`.pt`/`.h5`) se carga en cada etapa.

## Estructura de carpetas

```
proyecto/
├── CLAUDE.md
├── ARCHITECTURE.md
├── DECISIONS.md
├── MEJORAS_FUTURAS.md
├── .gitignore
├── .gitattributes               # fuerza LF en *.sh (se ejecutan en EC2 Linux)
├── run_all.sh                   # orquestador de punta a punta para EC2 (ver README)
├── requirements-torch.txt
├── requirements-tf.txt
├── data/
│   ├── download_dataset.py      # descarga los 7 datasets crudos (URLs publicas, sin credenciales)
│   ├── prepare_data.py          # splits train/val/test de las 3 tareas: detector, dog_breed, cat_breed
│   ├── raw/                     # NO va a git
│   └── processed/               # NO va a git — processed/<tarea>/<split>/<clase>/*.jpg
├── train/
│   ├── common_pytorch.py                # motor de entrenamiento PyTorch (compartido por los 3 modelos)
│   ├── common_tensorflow.py             # motor de entrenamiento TF/Keras (espejo del anterior)
│   ├── train_detector_pytorch.py        # Modelo 1 - PyTorch (wrapper fino sobre el motor)
│   ├── train_detector_tensorflow.py     # Modelo 1 - TensorFlow/Keras
│   ├── train_dog_breed_pytorch.py       # Modelo 2 - PyTorch
│   ├── train_dog_breed_tensorflow.py    # Modelo 2 - TensorFlow/Keras
│   ├── train_cat_breed_pytorch.py       # Modelo 3 - PyTorch
│   ├── train_cat_breed_tensorflow.py    # Modelo 3 - TensorFlow/Keras
│   ├── tune_detector_pytorch.py         # busqueda de hiperparametros (Optuna), version PyTorch
│   ├── gradcam_detector_pytorch.py      # diagnostico Grad-CAM, version PyTorch
│   └── compare_frameworks.py            # consolida las 6 metricas en la tabla comparativa (solo stdlib)
├── models/                      # pesos entrenados, NO va a git
│   ├── <tarea>_pytorch.pt       # tarea = detector | dog_breed | cat_breed
│   └── <tarea>_tensorflow.keras
├── metrics/                     # resultados de entrenamiento (SÍ va a git)
│   ├── <tarea>_<framework>_metrics.json          # 6 archivos
│   ├── <tarea>_<framework>_confusion_matrix.png  # 6 archivos
│   ├── <tarea>_data_distribution.json            # 3 archivos (que dato quedo en que split)
│   ├── gradcam/                                  # heatmaps del detector PyTorch
│   └── comparacion_tf_vs_pytorch.json            # tabla comparativa final
├── inference/
│   └── predict_camera.py        # script de consumo, obligatorio, usa OpenCV (pendiente)
└── utils/
    ├── report_common.py         # metricas/plots/json compartidos — NO importa torch ni tensorflow
    ├── transforms_pytorch.py    # preprocesamiento y augmentation version PyTorch
    ├── transforms_tensorflow.py # carga tf.data + augmentation version TF
    ├── model_defs_pytorch.py    # SimpleCNN PyTorch (nunca importa tensorflow)
    └── model_defs_tensorflow.py # SimpleCNN TF/Keras, espejo capa a capa (nunca importa torch)
```

Nota: ningún archivo `*_pytorch.py` debe importar `tensorflow`, ni ningún `*_tensorflow.py` debe importar `torch` — los venvs están separados por conflictos de CUDA (ver README.md), y mezclar imports rompería esa separación aunque el archivo "funcione" en la laptop de quien lo escribió.

## Datasets propuestos

| Modelo | Dataset sugerido | Notas |
|---|---|---|
| Detector | perro/gato: Kaggle Cats vs Dogs + COCO val2017 (dog/cat en contexto de escena real) · ninguno: Food101 + STL10 + Places365, sin categorías relacionadas a perro/gato, cuota pareja entre las 3 | Aquí se genera el desbalance intencional. COCO evita que el modelo aprenda a distinguir clases por estilo de foto (mascota en primer plano vs escena amplia) en vez de por el animal en sí. "Ninguno" combina comida, otros animales y paisajes/escenas para variedad real |
| Raza perro | Stanford Dogs, top-15 razas por cantidad de imágenes (`DOG_BREEDS_TOP_N` en `prepare_data.py`) | Entrenar desde cero con 120 clases y ~150 imgs/clase no es viable; top-15 maximiza datos por clase. Descarga directa sin credenciales |
| Raza gato | Oxford-IIIT Pet, las 12 razas de gato (archivos con inicial mayúscula) | ~200 imgs/raza: dataset chico, esperar accuracy menor que en perros. Se eligió sobre el Cat Breeds Dataset de Kaggle porque no requiere API key |

## Decisión de desbalance (Modelo 1)

Definir explícitamente la proporción de clases antes de entrenar, por ejemplo:
- Perro: 35%
- Gato: 35%
- Ninguno: 30% (o invertir para desbalancear más fuerte, ej. 45/45/10)

Documentar en el informe:
1. Distribución real usada
2. Técnica de mitigación (weighted `CrossEntropyLoss`, oversampling, o ambas)
3. Métricas: accuracy global + F1 por clase + matriz de confusión (no reportar solo accuracy, es engañoso con desbalance)

## Comparación TF vs PyTorch

La comparación ahora es **dentro de cada modelo** (su versión PyTorch vs su versión TF), no entre modelos distintos. Mismo criterio de arquitectura entre las dos versiones de un mismo modelo para que la comparación sea válida:
- Misma cantidad de capas convolucionales
- Mismo tamaño de filtros/kernels
- Mismo optimizador (ej. Adam) y mismo learning rate
- Mismo batch size
- Mismo split de datos (train/val/test idénticos entre ambas versiones de un modelo)
- Entrenados en el mismo hardware (misma instancia EC2)

Criterios a comparar y documentar en `metrics/comparacion_tf_vs_pytorch.json` (uno por modelo, o uno consolidado con los 3):
- Accuracy / F1 final en el mismo split de test
- Tiempo de entrenamiento total
- Tamaño del archivo de pesos resultante
- Líneas de código del pipeline de entrenamiento
- Facilidad de exportación para inferencia

Con 6 pipelines de entrenamiento en total, conviene que `utils/model_defs_pytorch.py` y `utils/model_defs_tensorflow.py` definan la arquitectura de los 3 modelos (no solo el detector), para que la equivalencia de capas/hiperparámetros entre frameworks quede en un solo lugar por framework en vez de repetida en cada script de train.

## Entorno de ejecución

- **Entrenamiento:** AWS EC2 (instancia con GPU, ver DECISIONS.md para el tipo elegido)
- **Inferencia con cámara:** máquina local (requiere webcam, EC2 no tiene)
