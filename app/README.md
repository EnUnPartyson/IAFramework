# App Ionic — Clasificador de Mascotas

App para probar los modelos desde el celular. Toma una foto (o la elige de la galería),
la manda a la API de Python que corre los modelos, y muestra especie + raza + el modo
"raza no identificada".

## Arquitectura

```
   Celular                          Laptop (misma red WiFi)
┌─────────────┐   POST /predict   ┌──────────────────────────┐
│  App Ionic  │ ────────────────▶ │  inference/server.py     │
│ (Capacitor) │ ◀──────────────── │  (FastAPI + los modelos) │
└─────────────┘    JSON con la    └──────────────────────────┘
                    prediccion
```

Los modelos son `.pt` de PyTorch: no corren en JavaScript, por eso la app es un cliente
liviano y la inferencia pasa en la laptop. La app no necesita GPU ni Python.

## 1. Levantar la API (en la laptop, con los pesos ya bajados de la EC2)

```bash
cd ..
venv-torch/bin/python inference/server.py
```

Al arrancar imprime la IP a usar desde el celular. Si todavía no tenés los pesos
entrenados, se puede desarrollar la app igual con respuestas simuladas:

```bash
venv-torch/bin/python inference/server.py --demo
```

## 2. Correr la app

**En el navegador** (desarrollo rápido, usa la webcam de la laptop):

```bash
npm install     # solo la primera vez
npm run dev
```

**En el celular como app nativa** (usa la cámara real vía Capacitor):

```bash
npm run build
npx cap add android          # solo la primera vez
npx cap sync
npx cap open android         # abre Android Studio para compilar e instalar
```

Requiere Android Studio instalado. Para iOS, `npx cap add ios` y Xcode (solo en Mac).

## 3. Configurar la URL del servidor

En la app, botón de ajustes (arriba a la derecha) → poner la IP que imprimió el servidor,
por ejemplo `http://192.168.1.100:8000`. Queda guardada en el dispositivo.

El celular y la laptop tienen que estar en la **misma red WiFi**. Si no conecta, suele ser
el firewall de Windows bloqueando el puerto 8000 — hay que permitirlo para redes privadas.

## Notas

- Android requiere permitir tráfico HTTP en claro (sin TLS). Capacitor ya lo habilita para
  desarrollo; para producción convendría poner la API detrás de HTTPS.
- La API también tiene documentación interactiva en `http://<ip>:8000/docs`, útil para
  probar el backend desde el navegador del celular sin la app.
