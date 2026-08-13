# MEJORAS_FUTURAS.md

Mejoras posibles para modelos y datos, no bloqueantes para el entregable base. Ninguna está implementada todavía. Marcar con [x] las que se vayan aplicando y anotar en qué modelo(s).

## Prioridad sugerida (barato + alto impacto primero)

1. ~~Scheduler de learning rate~~ — hecho 2026-07-23 en los 6 pipelines
2. ~~Early stopping por paciencia~~ — hecho 2026-07-23 en los 6 pipelines
3. Set de validación "real" con fotos de la webcam que se va a usar en el demo

## Datos

- [ ] **Simular la brecha entrenamiento↔despliegue**: los datasets tienen fotos bien enfocadas y con buena luz; una webcam real da imágenes con blur de movimiento, contraluz, compresión JPEG agresiva. Agregar augmentation que simule esto (`GaussianBlur`, `RandomAdjustSharpness`, compresión JPEG sintética) para no perder accuracy real al pasar de dataset a cámara.
- [ ] **Set de validación "real"**: sacar 20-30 fotos con la webcam del demo final y usarlas como chequeo cualitativo aparte del test set. Barato y es la prueba más honesta de que el sistema funciona en el escenario real.
- [ ] **Minería de negativos duros**: después de la primera corrida, revisar los falsos positivos de "ninguno" (¿qué confunde el modelo con perro/gato?) y sumar más ejemplos parecidos a esos casos.
- [ ] **Muestreo balanceado por batch** (`WeightedRandomSampler`) además de (o en vez de) la loss ponderada — a veces estabiliza mejor el entrenamiento con desbalance.

## Modelo / entrenamiento

- [x] **Scheduler de learning rate** — `ReduceLROnPlateau(factor=0.5, patience=2)` aplicado 2026-07-23 en ambos motores (`common_pytorch.py` / `common_tensorflow.py`), para los 3 modelos.
- [x] **Early stopping por paciencia** — paciencia 6 sobre `val_accuracy`, aplicado 2026-07-23 en ambos motores. También se agregó `weight_decay=1e-4` por defecto como regularización.
- [ ] **Label smoothing** en la loss — ayuda a calibrar mejor las probabilidades, relevante para el modo "forzado" (Modelo 1) que decide por argmax.
- [ ] **Calibración de confianza** (temperature scaling) si el producto final muestra "% de confianza" al usuario — sin esto, una CNN entrenada desde cero suele dar probabilidades mal calibradas (demasiado seguras).

## Evaluación

- [ ] **Múltiples semillas**: correr el entrenamiento 2-3 veces con distinta semilla y reportar accuracy/F1 promedio ± desvío, no un solo número. Más convincente para el informe y menos vulnerable a "tuvimos suerte con el split".
- [ ] **Benchmark de latencia/FPS** en el hardware real de inferencia (la laptop, no la EC2) — es un producto de cámara en vivo, no un clasificador offline.
- [ ] **Prueba de robustez**: imágenes con el animal parcialmente fuera de cuadro u ocluido, ya que la cámara real no siempre va a tener una vista limpia y centrada.

## Relación con otros documentos

- Decisiones ya tomadas y pendientes bloqueantes: ver [DECISIONS.md](DECISIONS.md).
- Diseño del pipeline y estructura de carpetas: ver [ARCHITECTURE.md](ARCHITECTURE.md).
