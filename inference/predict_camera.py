"""Script de consumo obligatorio: captura la camara con OpenCV y corre el pipeline completo.

Requisito del profesor (ver CLAUDE.md): OpenCV es mandatorio para el manejo de camara.
Corre en la maquina local (la EC2 no tiene webcam), con los pesos ya bajados por scp.

    venv-torch/bin/python inference/predict_camera.py              # webcam en vivo
    venv-torch/bin/python inference/predict_camera.py --imagen foto.jpg   # una imagen suelta
    venv-torch/bin/python inference/predict_camera.py --forced     # ignora la clase "ninguno"

Teclas: [q] salir  ·  [espacio] congelar/reanudar  ·  [g] guardar el frame actual
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from inference.pipeline_pytorch import PetPipeline, Prediction  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
COLOR_POR_ESPECIE = {  # BGR
    "perro": (60, 180, 75),
    "gato": (240, 150, 30),
    "ninguno": (120, 120, 120),
}


def _dibujar(frame, pred: Prediction, fps: float) -> None:
    color = COLOR_POR_ESPECIE.get(pred.especie, (200, 200, 200))
    alto, ancho = frame.shape[:2]

    # panel superior semitransparente para que el texto se lea sobre cualquier fondo
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (ancho, 108), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, pred.resumen, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    if pred.top_razas:
        detalle = "  |  ".join(f"{n} {p:.0%}" for n, p in pred.top_razas[:3])
        cv2.putText(frame, detalle, (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    if not pred.raza_identificada and pred.raza is not None:
        cv2.putText(
            frame, "confianza baja: raza no identificada", (14, 96),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 200, 255), 1, cv2.LINE_AA,
        )

    cv2.putText(
        frame, f"{fps:.1f} FPS  [q] salir  [espacio] congelar  [g] guardar",
        (14, alto - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )


def _procesar_imagen(pipeline: PetPipeline, ruta: Path) -> None:
    image = Image.open(ruta)
    inicio = time.perf_counter()
    pred = pipeline.predict(image)
    print(f"\n{ruta.name}: {pred.resumen}  ({(time.perf_counter() - inicio) * 1000:.0f} ms)")
    for nombre, prob in pred.top_razas:
        print(f"   {nombre:28} {prob:.1%}")


def _bucle_camara(pipeline: PetPipeline, indice_camara: int, cada_n_frames: int) -> None:
    captura = cv2.VideoCapture(indice_camara)
    if not captura.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la camara {indice_camara}. Probar otro indice con --camara 1"
        )

    salida_dir = ROOT_DIR / "capturas"
    pred: Prediction | None = None
    congelado = False
    n_frame, fps, t_ultimo = 0, 0.0, time.perf_counter()
    print("Camara abierta. [q] salir  [espacio] congelar  [g] guardar")

    try:
        while True:
            if not congelado:
                ok, frame = captura.read()
                if not ok:
                    print("No se pudo leer el frame; se corta el bucle")
                    break

                # clasificar 1 de cada N frames: el pipeline es mas lento que la camara
                if n_frame % cada_n_frames == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pred = pipeline.predict(Image.fromarray(rgb))
                    ahora = time.perf_counter()
                    fps = cada_n_frames / max(ahora - t_ultimo, 1e-6)
                    t_ultimo = ahora
                n_frame += 1

            vista = frame.copy()
            if pred is not None:
                _dibujar(vista, pred, fps)
            if congelado:
                cv2.putText(
                    vista, "CONGELADO", (vista.shape[1] - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 2, cv2.LINE_AA,
                )
            cv2.imshow("Clasificador de mascotas", vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord(" "):
                congelado = not congelado
            if tecla == ord("g"):
                salida_dir.mkdir(exist_ok=True)
                destino = salida_dir / f"captura_{int(time.time())}.jpg"
                cv2.imwrite(str(destino), vista)
                print(f"Guardado en {destino}")
    finally:
        captura.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasificacion de mascotas con camara (OpenCV)")
    parser.add_argument("--imagen", type=Path, help="clasificar una imagen en vez de usar la camara")
    parser.add_argument("--camara", type=int, default=0, help="indice de la camara (default 0)")
    parser.add_argument("--cada", type=int, default=5, help="clasificar 1 de cada N frames")
    parser.add_argument("--forced", action="store_true", help="modo forzado: ignora la clase 'ninguno'")
    parser.add_argument(
        "--sin-tta", dest="sin_tta", action="store_true",
        help="desactiva el test-time augmentation (1 forward en vez de 2, prediccion mas ruidosa)",
    )
    args = parser.parse_args()

    pipeline = PetPipeline(forced=args.forced, tta=not args.sin_tta)
    print(f"Modelos cargados en {pipeline.device}")

    if args.imagen:
        _procesar_imagen(pipeline, args.imagen)
    else:
        _bucle_camara(pipeline, args.camara, max(1, args.cada))


if __name__ == "__main__":
    main()
