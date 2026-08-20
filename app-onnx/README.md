# Mascotas Live — inferencia ONNX on-device

Segunda app del proyecto, **separada de `app/`** (que sigue siendo el cliente de la API para
PC/web). Esta corre los modelos **dentro del teléfono** con ONNX Runtime Web y clasifica la
**cámara en vivo**, cuadro a cuadro — la experiencia de `inference/predict_camera.py`
(OpenCV en la laptop) llevada al celular, sin servidor y sin internet.

```
   app/ (la original)                      app-onnx/ (esta)
┌──────────────────────────┐          ┌──────────────────────────┐
│ Cliente liviano          │          │ Modelos ONNX empaquetados │
│ foto → POST /predict     │          │ cámara en vivo → wasm     │
│ requiere el servidor EC2 │          │ 100% offline              │
│ V1 + PRO + comparación   │          │ solo V1 (ver abajo)       │
└──────────────────────────┘          └──────────────────────────┘
```

## Por qué solo los modelos V1

Los tres V1 exportados a ONNX pesan ~32 MB en total y corren en decenas de milisegundos por
cuadro en el CPU del teléfono — aptos para video en vivo. Los PRO (ConvNeXt-Tiny) pesan
~110 MB **cada uno** y tardarían segundos por cuadro en wasm: inviables para empaquetar y
para video. Quien quiera los PRO usa la app original contra la API. `inference/export_onnx.py
--pro` los exporta igualmente si hacen falta para otro consumidor.

## Cómo funciona

- `inference/export_onnx.py` exporta los checkpoints PyTorch a `.onnx` (con verificación
  numérica contra PyTorch, tolerancia 1e-4) y genera `manifest.json` con clases, resolución,
  normalización y umbral de "raza no identificada" — todo lo que el cliente JS necesita.
- `src/pipeline.ts` replica `pipeline_pytorch.py`: preprocesamiento (resize por canvas +
  normalización), softmax con temperatura, cascada detector → raza, umbral de desconocidas.
- `src/App.tsx`: video con `getUserMedia`, clasificación cada ~400 ms (sin encolar si una
  inferencia tarda más), overlay con especie/raza/top-3/latencia, pausa y cambio de cámara.
- Los binarios wasm de onnxruntime-web se copian a `public/ort/` en cada build
  (`scripts/copy-ort.mjs`): la app no toca ningún CDN.

La verificación de paridad está hecha de punta a punta: la misma entrada produce los mismos
logits en PyTorch, en onnxruntime de Python y en onnxruntime-web de JavaScript (diff < 1e-5).

## Desarrollo

```bash
npm install
npm run dev          # http://localhost:5174 — la camara funciona en localhost sin HTTPS
```

Si cambian los modelos: re-exportar y copiar de nuevo a `public/models/`:

```bash
../venv-torch/Scripts/python.exe ../inference/export_onnx.py
cp ../models/onnx/{detector_v1,dog_breed_v1,cat_breed_v1}.onnx ../models/onnx/manifest.json public/models/
```

## Compilar el APK

```bash
npm run build
npx cap sync android
cd android
JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-21.0.6.7-hotspot" ./gradlew assembleDebug
```

**Ojo con el JDK**: el JBR de Android Studio es Java 25 y Gradle 8.14 no lo soporta
("Unsupported class file major version 69") — usar el Temurin 21 del sistema. El proyecto
`android/` no se versiona; al regenerarlo desde cero (`npx cap add android`) hay que volver a:

1. copiar `local.properties` (ruta del SDK) desde `app/android/`,
2. agregar en `AndroidManifest.xml`, junto al permiso de INTERNET:
   ```xml
   <uses-permission android:name="android.permission.CAMERA" />
   <uses-feature android:name="android.hardware.camera" android:required="false" />
   ```
   Sin eso `getUserMedia` falla y la app queda en "No hay acceso a la cámara".

El APK (~68 MB: la app + 32 MB de modelos + ~20 MB del runtime wasm) queda en
`android/app/build/outputs/apk/debug/app-debug.apk`. No necesita cleartext ni configuración
de URL: no usa red.

## Compartirla

El servidor de la EC2 la sirve en `http://44.201.7.59:8000/descargar/live`. Para actualizarla:

```bash
scp -i tu-llave.pem android/app/build/outputs/apk/debug/app-debug.apk \
    ubuntu@44.201.7.59:~/IAFramework/deploy/mascotas-live.apk
ssh -i tu-llave.pem ubuntu@44.201.7.59 "sudo systemctl restart mascotas-api"
```

## Sobre "usar OpenCV"

En Python, OpenCV cumple dos roles en `predict_camera.py`: capturar frames de la cámara y
dibujar el resultado. En el navegador esos dos roles los cumplen las APIs nativas
(`getUserMedia` + canvas), que es lo que usa esta app — meter opencv.js (~8 MB de wasm extra)
solo duplicaría lo que el navegador ya hace. Si más adelante hiciera falta procesamiento de
imagen real (detección de bordes, tracking), ahí sí opencv.js tendría sentido.
