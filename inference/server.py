"""API HTTP que expone el pipeline de clasificacion para la app Ionic.

Corre en la maquina donde esten los pesos (la laptop, no la EC2) y el celular le pega
por la IP de la red local. Los modelos se cargan una sola vez al arrancar.

    venv-torch/bin/python inference/server.py                 # escucha en 0.0.0.0:8000
    venv-torch/bin/python inference/server.py --demo          # sin pesos, respuestas simuladas
    venv-torch/bin/python inference/server.py --forced        # ignora la clase "ninguno"

Probar desde el celular: http://<ip-de-la-laptop>:8000/docs
"""
from __future__ import annotations

import argparse
import io
import random
import socket
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from inference.pipeline_pytorch import PetPipeline, Prediction  # noqa: E402

# APK de la app Ionic, servida desde el mismo servidor para que se pueda compartir con un
# link (ver app/README.md: se compila con VITE_API_URL apuntando a esta misma instancia,
# asi que quien la instala no tiene que configurar nada). Opcional: si no esta, /descargar
# devuelve 404 en vez de romper el arranque del server.
APK_PATH = ROOT_DIR / "deploy" / "clasificador-mascotas.apk"

app = FastAPI(title="Clasificador de Mascotas", version="1.0")

# la app Ionic corre en otro origen (capacitor://localhost o http://localhost:8100)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# puede haber uno o los dos frameworks cargados a la vez: para inferencia en CPU
# torch y tensorflow conviven sin el conflicto de CUDA que obliga a venvs separados
_state: dict = {"pipelines": {}, "demo": False, "error": None}


def _principal():
    """El pipeline que responde /predict cuando no se pide uno puntual."""
    return next(iter(_state["pipelines"].values()), None)

DEMO_RAZAS = ["beagle", "boxer", "pug", "samoyed", "chihuahua"]


def _demo_prediction() -> Prediction:
    especie = random.choice(["perro", "gato", "ninguno"])
    if especie == "ninguno":
        return Prediction(especie=especie, especie_confianza=random.uniform(0.7, 0.99))
    razas = random.sample(DEMO_RAZAS, 3)
    conf = random.uniform(0.3, 0.95)
    return Prediction(
        especie=especie,
        especie_confianza=random.uniform(0.7, 0.99),
        raza=razas[0],
        raza_confianza=conf,
        raza_identificada=conf >= 0.45,
        top_razas=[(razas[0], conf), (razas[1], conf * 0.5), (razas[2], conf * 0.2)],
    )


@app.get("/", response_class=HTMLResponse)
def inicio() -> str:
    """Pagina de estado: es lo que ve quien abre la URL base en el navegador."""
    if _state["demo"]:
        estado, color = "Modo demo (respuestas simuladas)", "#A9700F"
    elif _state["pipelines"]:
        cargados = ", ".join(sorted(_state["pipelines"]))
        estado, color = f"Modelos cargados y listos ({cargados})", "#2E7D53"
    else:
        estado, color = f"Sin modelos: {_state['error'] or 'error desconocido'}", "#B03A2E"

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clasificador de Mascotas - API</title>
<style>
 body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:34rem;margin:0 auto;
      padding:2rem 1.25rem;line-height:1.6;color:#12181A;background:#FBFCFC}}
 h1{{font-size:1.5rem;margin:0 0 .25rem}}
 .estado{{color:{color};font-weight:600;margin:0 0 1.5rem}}
 a{{color:#0F5C63}}
 table{{width:100%;border-collapse:collapse;font-size:.95rem}}
 td{{padding:.5rem 0;border-bottom:1px solid #D3DADA;vertical-align:top}}
 td:first-child{{white-space:nowrap;padding-right:1rem;font-family:ui-monospace,monospace}}
 .pie{{margin-top:1.5rem;font-size:.9rem;color:#4A585B}}
 @media(prefers-color-scheme:dark){{body{{background:#0E1315;color:#E4EAEA}}
   td{{border-color:#2A3538}} a{{color:#5FBDC0}} .pie{{color:#A5B3B5}}}}
</style></head><body>
<h1>Clasificador de Mascotas</h1>
<p class="estado">{estado}</p>
<table>
 <tr><td><a href="/descargar">/descargar</a></td><td><b>Bajar la app para Android (.apk)</b> -- ya viene apuntando a este servidor</td></tr>
 <tr><td><a href="/docs">/docs</a></td><td>Probar la API desde el navegador: subir una foto y ver la prediccion</td></tr>
 <tr><td><a href="/health">/health</a></td><td>Estado del servidor</td></tr>
 <tr><td><a href="/info">/info</a></td><td>Modelos cargados, clases y umbrales</td></tr>
 <tr><td>POST /predict</td><td>Recibe una imagen y devuelve especie, raza y confianza</td></tr>
</table>
<p class="pie">Si llegaste aca desde el celular, la conexion funciona: pone esta misma
direccion en los ajustes de la app.</p>
</body></html>"""


@app.get("/descargar")
def descargar_apk() -> FileResponse:
    """APK de la app para Android. Instalarla pide activar 'apps de origenes desconocidos'
    una vez (Android bloquea instalar fuera de Play Store por defecto)."""
    if not APK_PATH.exists():
        raise HTTPException(status_code=404, detail="la apk todavia no se subio a este servidor")
    return FileResponse(
        APK_PATH, media_type="application/vnd.android.package-archive",
        filename="clasificador-mascotas.apk",
    )


@app.get("/health")
def health() -> dict:
    return {
        "ok": bool(_state["pipelines"]) or _state["demo"],
        "modo": "demo" if _state["demo"] else "modelos reales",
        "frameworks": sorted(_state["pipelines"]),
        "error": _state["error"],
    }


@app.get("/info")
def info() -> dict:
    if _state["demo"]:
        return {"modo": "demo", "razas_demo": DEMO_RAZAS}
    if not _state["pipelines"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "modelos no cargados")
    return {fw: p.describe() for fw, p in _state["pipelines"].items()}


@app.post("/predict")
async def predict(file: UploadFile = File(...), framework: str | None = None) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="archivo vacio")

    if _state["demo"]:
        return _demo_prediction().to_dict()

    pipeline = _state["pipelines"].get(framework) if framework else _principal()
    if pipeline is None:
        detalle = (
            f"framework '{framework}' no cargado; disponibles: {sorted(_state['pipelines'])}"
            if framework
            else _state["error"] or "modelos no cargados"
        )
        raise HTTPException(status_code=503, detail=detalle)

    return pipeline.predict(_abrir(raw)).to_dict()


@app.post("/predict/comparar")
async def predict_comparar(file: UploadFile = File(...)) -> dict:
    """Corre la MISMA imagen por todos los frameworks cargados y devuelve cada resultado.

    Es lo que permite mostrar la comparacion PyTorch vs TensorFlow en vivo.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="archivo vacio")

    if _state["demo"]:
        return {"frameworks": {fw: _demo_prediction().to_dict() for fw in ("pytorch", "tensorflow")}}

    if not _state["pipelines"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "modelos no cargados")

    image = _abrir(raw)
    v1 = {fw: p for fw, p in _state["pipelines"].items() if fw in ("pytorch", "tensorflow")}
    if not v1:
        raise HTTPException(status_code=503, detail="no hay modelos v1 cargados para comparar")
    return {"frameworks": {fw: p.predict(image).to_dict() for fw, p in v1.items()}}


def _abrir(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="el archivo no es una imagen valida")


def _local_ip() -> str:
    """IP de la maquina en la red local, para que el celular sepa a donde pegarle."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no envia nada, solo resuelve la interfaz de salida
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="API de clasificacion de mascotas")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--demo", action="store_true", help="respuestas simuladas, sin cargar pesos")
    parser.add_argument("--forced", action="store_true", help="modo forzado: ignora la clase 'ninguno'")
    parser.add_argument(
        "--sin-tta", dest="sin_tta", action="store_true",
        help="desactiva el test-time augmentation (1 forward en vez de 2, prediccion mas ruidosa)",
    )
    parser.add_argument(
        "--frameworks",
        choices=("pytorch", "tensorflow", "ambos"),
        default="pytorch",
        help="que modelos cargar. 'ambos' habilita /predict/comparar y requiere torch y "
        "tensorflow en el mismo venv (para inferencia en CPU conviven sin problema)",
    )
    parser.add_argument(
        "--sin-pro", dest="sin_pro", action="store_true",
        help="no intentar cargar los modelos pro (por defecto se cargan si existen en models/)",
    )
    args = parser.parse_args()

    _state["demo"] = args.demo
    if args.demo:
        print("MODO DEMO: respuestas simuladas, no se cargan modelos")
    else:
        errores = []
        pedidos = ("pytorch", "tensorflow") if args.frameworks == "ambos" else (args.frameworks,)
        if not args.sin_pro:
            # el modo pro (transfer learning) se suma si sus pesos existen; si todavia no se
            # entrenaron, se avisa y la app simplemente no lo ofrece
            pedidos = pedidos + ("pro",)

        for fw in pedidos:
            try:
                if fw == "pytorch":
                    _state["pipelines"][fw] = PetPipeline(forced=args.forced, tta=not args.sin_tta)
                elif fw == "pro":
                    _state["pipelines"][fw] = PetPipeline(forced=args.forced, tta=not args.sin_tta, pro=True)
                else:
                    # se importa aca y no arriba: si solo se pide PyTorch, no hace falta
                    # tener tensorflow instalado
                    from inference.pipeline_tensorflow import PetPipelineTF

                    _state["pipelines"][fw] = PetPipelineTF(forced=args.forced, tta=not args.sin_tta)
                print(f"  [{fw}] modelos cargados")
            except Exception as exc:  # noqa: BLE001 -- en la nube, un modelo roto no debe tirar el server
                errores.append(f"{fw}: {exc}")
                print(f"  [{fw}] NO cargado -> {type(exc).__name__}: {exc}")

        if not _state["pipelines"]:
            # arranca igual para no bloquear el desarrollo de la app antes de tener pesos
            _state["error"] = " | ".join(errores)
            print("AVISO: no se cargo ningun modelo. /predict devolvera 503.")
            print("Para desarrollar la app sin pesos: python inference/server.py --demo")

    ip = _local_ip()
    print(f"\nAPI escuchando en http://{args.host}:{args.port}")
    print(f"Desde el celular (misma red WiFi): http://{ip}:{args.port}")
    print(f"Documentacion interactiva:         http://{ip}:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
