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
from PIL import Image, UnidentifiedImageError

sys.path.append(str(Path(__file__).resolve().parent.parent))
from inference.pipeline_pytorch import PetPipeline, Prediction  # noqa: E402

app = FastAPI(title="Clasificador de Mascotas", version="1.0")

# la app Ionic corre en otro origen (capacitor://localhost o http://localhost:8100)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {"pipeline": None, "demo": False, "error": None}

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


@app.get("/health")
def health() -> dict:
    return {
        "ok": _state["pipeline"] is not None or _state["demo"],
        "modo": "demo" if _state["demo"] else "modelos reales",
        "error": _state["error"],
    }


@app.get("/info")
def info() -> dict:
    if _state["demo"]:
        return {"modo": "demo", "razas_demo": DEMO_RAZAS}
    if _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "modelos no cargados")
    return _state["pipeline"].describe()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="archivo vacio")

    if _state["demo"]:
        return _demo_prediction().to_dict()

    if _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "modelos no cargados")

    try:
        image = Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="el archivo no es una imagen valida")

    return _state["pipeline"].predict(image).to_dict()


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
    args = parser.parse_args()

    _state["demo"] = args.demo
    if args.demo:
        print("MODO DEMO: respuestas simuladas, no se cargan modelos")
    else:
        try:
            _state["pipeline"] = PetPipeline(forced=args.forced)
            print("Modelos cargados:", _state["pipeline"].describe())
        except FileNotFoundError as exc:
            # arranca igual para no bloquear el desarrollo de la app antes de tener pesos
            _state["error"] = str(exc)
            print(f"\nAVISO: {exc}")
            print("El servidor arranca igual; /predict devolvera 503 hasta que existan los pesos.")
            print("Para desarrollar la app sin pesos: python inference/server.py --demo\n")

    ip = _local_ip()
    print(f"\nAPI escuchando en http://{args.host}:{args.port}")
    print(f"Desde el celular (misma red WiFi): http://{ip}:{args.port}")
    print(f"Documentacion interactiva:         http://{ip}:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
