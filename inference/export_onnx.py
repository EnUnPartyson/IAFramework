"""Exporta los modelos PyTorch a ONNX para inferencia on-device (app-onnx/, sin API).

Por defecto exporta los 3 modelos V1 (chicos: ~32 MB en total, aptos para correr en el
navegador del celular con onnxruntime-web). Con --pro exporta los ConvNeXt del modo pro
(~110 MB cada uno: demasiado pesados para empaquetar en una app movil, pero utiles para
otros consumidores de ONNX).

Ademas del .onnx genera models/onnx/manifest.json con todo lo que el cliente JavaScript
necesita para reproducir el preprocesamiento y el pipeline de 2 etapas: clases, resolucion,
normalizacion y umbral de "raza no identificada".

Cada export se VERIFICA contra PyTorch: misma entrada aleatoria, se exige que los logits
coincidan (diferencia maxima < 1e-4).

    venv-torch/bin/python inference/export_onnx.py          # v1 (los 3)
    venv-torch/bin/python inference/export_onnx.py --pro    # ademas, los 3 pro
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from inference.pipeline_pytorch import load_model  # noqa: E402
from utils.transforms_pytorch import (  # noqa: E402
    NORM_IMAGENET_MEAN,
    NORM_IMAGENET_STD,
    NORM_MEAN,
    NORM_STD,
)

OUT_DIR = ROOT_DIR / "models" / "onnx"
TASKS = ("detector", "dog_breed", "cat_breed")
TOLERANCIA = 1e-4


def exportar(task: str, pro: bool) -> dict:
    sufijo = "pro" if pro else "v1"
    loaded = load_model(task, torch.device("cpu"), pro=pro)
    model = loaded.model
    model.eval()

    out_path = OUT_DIR / f"{task}_{sufijo}.onnx"
    ejemplo = torch.randn(1, 3, loaded.img_size, loaded.img_size)
    torch.onnx.export(
        model,
        (ejemplo,),
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # el exportador clasico: estable para estas arquitecturas
    )

    # verificacion numerica contra PyTorch con una entrada nueva
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    prueba = torch.randn(2, 3, loaded.img_size, loaded.img_size)
    with torch.no_grad():
        esperado = model(prueba).numpy()
    obtenido = sess.run(["logits"], {"input": prueba.numpy()})[0]
    diff = float(np.abs(esperado - obtenido).max())
    if diff > TOLERANCIA:
        raise RuntimeError(f"{out_path.name}: divergencia ONNX vs PyTorch = {diff:.2e} (> {TOLERANCIA})")

    mb = out_path.stat().st_size / 1e6
    print(f"  {out_path.name}: {mb:.1f} MB, {len(loaded.class_names)} clases, "
          f"{loaded.img_size}px, diff max {diff:.1e} OK")

    mean, std = (NORM_IMAGENET_MEAN, NORM_IMAGENET_STD) if loaded.norm == "imagenet" else (NORM_MEAN, NORM_STD)
    return {
        "archivo": out_path.name,
        "clases": loaded.class_names,
        "img_size": loaded.img_size,
        "norm_mean": mean,
        "norm_std": std,
        # umbral de "raza no identificada" (None en el detector, que no lo usa)
        "umbral_no_identificada": loaded.unknown_threshold if task != "detector" else None,
        "temperatura": loaded.temperature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta los modelos PyTorch a ONNX")
    parser.add_argument("--pro", action="store_true", help="exportar tambien los modelos pro (pesados)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    variantes = [False] + ([True] if args.pro else [])
    for pro in variantes:
        clave = "pro" if pro else "v1"
        print(f"== Exportando modelos {clave} ==")
        manifest[clave] = {task: exportar(task, pro) for task in TASKS}

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
