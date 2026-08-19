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

**En el celular, por el navegador** (lo más rápido, no necesita Android Studio):

```bash
npm run dev
```

Vite ya está configurado para exponerse en la red, así que imprime dos direcciones: usá la
que dice `Network:`, por ejemplo `http://192.168.100.194:5173`, y abrila en el celular.

La cámara **funciona sobre HTTP sin certificado**: Capacitor usa un `<input capture>` que
abre la cámara nativa del teléfono, no `getUserMedia` (que sí exigiría HTTPS).

Antes hay que abrir los dos puertos en el firewall de Windows, una sola vez, desde una
consola **como administrador**:

```bat
netsh advfirewall firewall add rule name="Clasificador API" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="Clasificador Vite" dir=in action=allow protocol=TCP localport=5173
```

**En el celular como app nativa** (APK instalable):

Requiere Android Studio. Al abrirlo por primera vez descarga el SDK de Android, que son
varios GB — conviene dejarlo terminar antes de seguir.

```bash
npm run build                # compila la app web
npx cap add android          # solo la primera vez: crea el proyecto Android
npx cap sync                 # copia el build al proyecto nativo
npx cap open android         # abre Android Studio
```

En Android Studio: conectá el celular por USB con **depuración USB** activada (en Ajustes →
Opciones de desarrollador; se habilitan tocando 7 veces "Número de compilación" en
Información del teléfono), elegilo en la lista de dispositivos y dale al botón Run.

Cada vez que cambies código: `npm run build && npx cap sync`, y Run de nuevo.

Para iOS hace falta `npx cap add ios` y Xcode, que solo corre en Mac.

### Si la app abre pero no conecta con la API

Android bloquea HTTP sin cifrar desde la versión 9. `capacitor.config.ts` ya trae
`server.cleartext: true` para permitirlo, pero si aun así falla, hay que agregar la IP a
`android/app/src/main/res/xml/network_security_config.xml`:

```xml
<domain-config cleartextTrafficPermitted="true">
    <domain includeSubdomains="true">192.168.100.194</domain>
</domain-config>
```

Después `npx cap sync` y compilar de nuevo. La causa es que la API va por HTTP plano; con
HTTPS el problema desaparecería.

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
