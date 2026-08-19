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

| Dónde corre la app | URL a usar |
|---|---|
| Navegador de la misma máquina que la API | `http://localhost:8000` (es el default, no hay que tocar nada) |
| Celular | La IP que imprime el servidor al arrancar, ej. `http://192.168.1.100:8000` |

Se cambia con el botón de ajustes (arriba a la derecha) y queda guardada en el dispositivo.

**Si desde el navegador local usás la IP de red en vez de `localhost`**, el pedido sale y
vuelve por la pila de red y el firewall de Windows puede cortarlo, con un error que el
navegador reporta confusamente como CORS. Usar `localhost` evita todo eso.

Para el celular, el teléfono y la laptop tienen que estar en la **misma red WiFi**, y hay
que permitir el puerto 8000 en el firewall de Windows para redes privadas.

## Notas

- Android requiere permitir tráfico HTTP en claro (sin TLS). Capacitor ya lo habilita para
  desarrollo; para producción convendría poner la API detrás de HTTPS.
- La API también tiene documentación interactiva en `http://<ip>:8000/docs`, útil para
  probar el backend desde el navegador del celular sin la app.
