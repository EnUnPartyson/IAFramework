"""Porta los pesos de los modelos TensorFlow (.keras) a ONNX, via SimpleCNN de PyTorch.

EXCEPCION DELIBERADA a la regla "nunca mezclar imports de torch y tensorflow": este script
es el puente entre ambos y solo corre en un entorno CPU con los dos instalados (venv-inf de
la EC2). La regla existe por el conflicto de CUDA, que en CPU no aplica.

Por que este camino y no tf2onnx: las arquitecturas TF y PyTorch del proyecto son espejos
capa a capa por diseno (misma cantidad de convs, filtros, BN con el mismo epsilon, mismo
cabezal). Copiar los pesos de Keras al SimpleCNN y exportar con torch.onnx produce modelos
ONNX identicos en formato a los ya empaquetados (NCHW, misma normalizacion): el cliente
JavaScript usa UN solo camino de codigo para ambos frameworks. tf2onnx habria generado
modelos NHWC con la normalizacion adentro y ademas esta atrasado respecto de Keras 3.

Verificaciones en cadena (aborta si alguna falla):
  1. keras vs torch portado: misma imagen, mismos logits (tolerancia 1e-3)
  2. torch portado vs sesion ONNX: idem
Ademas deja un vector de prueba (entrada + logits del detector) para verificar despues la
sesion de onnxruntime-web en JavaScript.

Uso (en la EC2):
    venv-inf/bin/pip install onnx onnxruntime
    venv-inf/bin/python inference/portar_pesos_tf_a_onnx.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utils.model_defs_pytorch import BLOCK_FILTERS, SimpleCNN  # noqa: E402

MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"
OUT_DIR = MODELS_DIR / "onnx"
TASKS = ("detector", "dog_breed", "cat_breed")
TOLERANCIA = 1e-3


def portar(task: str) -> dict:
    with open(METRICS_DIR / f"{task}_tensorflow_metrics.json", encoding="utf-8") as f:
        met = json.load(f)
    hp = met["hiperparametros"]
    head, blocks, img_size = hp["head"], int(hp["blocks"]), int(hp["img_size"])
    clases = met["test"]["clases"]

    keras_model = tf.keras.models.load_model(MODELS_DIR / f"{task}_tensorflow.keras", compile=False)
    tm = SimpleCNN(len(clases), dropout=0.0, head=head, blocks=blocks, img_size=img_size)

    # capas con pesos, en orden. La augmentation (Sequential anidado) y Rescaling no tienen
    # pesos entrenables y quedan fuera de estos filtros.
    k_convs = [l for l in keras_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    k_bns = [l for l in keras_model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    k_denses = [l for l in keras_model.layers if isinstance(l, tf.keras.layers.Dense)]
    t_convs = [m for m in tm.features if isinstance(m, torch.nn.Conv2d)]
    t_bns = [m for m in tm.features if isinstance(m, torch.nn.BatchNorm2d)]
    t_denses = [m for m in tm.classifier if isinstance(m, torch.nn.Linear)]
    assert len(k_convs) == len(t_convs) == blocks, (len(k_convs), len(t_convs), blocks)
    assert len(k_bns) == len(t_bns) == blocks
    assert len(k_denses) == len(t_denses) == 2

    with torch.no_grad():
        for k, t in zip(k_convs, t_convs):
            kernel, bias = k.get_weights()  # (kh, kw, in, out)
            t.weight.copy_(torch.from_numpy(kernel.transpose(3, 2, 0, 1)))
            t.bias.copy_(torch.from_numpy(bias))
        for k, t in zip(k_bns, t_bns):
            gamma, beta, mean, var = k.get_weights()
            t.weight.copy_(torch.from_numpy(gamma))
            t.bias.copy_(torch.from_numpy(beta))
            t.running_mean.copy_(torch.from_numpy(mean))
            t.running_var.copy_(torch.from_numpy(var))

        # primera densa: si el cabezal es flatten, Keras aplana (H, W, C) y PyTorch (C, H, W);
        # hay que reordenar las filas del kernel o la capa mezclaria features de posiciones
        # equivocadas. Con GAP la entrada es (C,) en ambos y no hay reordenamiento.
        kernel, bias = k_denses[0].get_weights()  # (in, 256)
        if head == "flatten":
            fm = img_size // (2 ** blocks)
            canales = BLOCK_FILTERS[blocks - 1]
            kernel = (
                kernel.reshape(fm, fm, canales, kernel.shape[1])
                .transpose(2, 0, 1, 3)
                .reshape(canales * fm * fm, kernel.shape[1])
            )
        t_denses[0].weight.copy_(torch.from_numpy(kernel.T))
        t_denses[0].bias.copy_(torch.from_numpy(bias))

        kernel, bias = k_denses[1].get_weights()
        t_denses[1].weight.copy_(torch.from_numpy(kernel.T))
        t_denses[1].bias.copy_(torch.from_numpy(bias))

    tm.eval()

    # --- verificacion 1: keras vs torch portado, misma imagen ---
    rng = np.random.default_rng(7)
    x255 = rng.uniform(0, 255, (2, img_size, img_size, 3)).astype(np.float32)
    k_logits = np.asarray(keras_model(x255, training=False))
    x_norm = ((x255 / 255.0) - 0.5) / 0.5  # lo que hace el preprocesamiento del cliente
    t_in = torch.from_numpy(x_norm.transpose(0, 3, 1, 2).copy())
    with torch.no_grad():
        t_logits = tm(t_in).numpy()
    diff_kt = float(np.abs(k_logits - t_logits).max())
    if diff_kt > TOLERANCIA:
        raise RuntimeError(f"{task}: keras vs torch portado diverge {diff_kt:.2e}")

    # --- export ONNX + verificacion 2 ---
    out_path = OUT_DIR / f"{task}_v1tf.onnx"
    torch.onnx.export(
        tm, (t_in[:1],), str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    o_logits = sess.run(["logits"], {"input": t_in.numpy()})[0]
    diff_to = float(np.abs(t_logits - o_logits).max())
    if diff_to > TOLERANCIA:
        raise RuntimeError(f"{task}: torch vs onnx diverge {diff_to:.2e}")

    mb = out_path.stat().st_size / 1e6
    print(f"  {out_path.name}: {mb:.1f} MB | keras vs torch {diff_kt:.1e} | torch vs onnx {diff_to:.1e} OK")

    if task == "detector":
        with open(OUT_DIR / "testvec_tf.json", "w", encoding="utf-8") as f:
            json.dump({
                "input_shape": list(t_in[:1].shape),
                "input": t_in[:1].numpy().flatten().round(6).tolist(),
                "logits_esperados": k_logits[:1].flatten().round(5).tolist(),
            }, f)

    umbral = None
    if task != "detector":
        umbral = float(met["raza_no_identificada"]["umbral_sugerido"])
    return {
        "archivo": out_path.name,
        "clases": clases,
        "img_size": img_size,
        "norm_mean": [0.5, 0.5, 0.5],
        "norm_std": [0.5, 0.5, 0.5],
        "umbral_no_identificada": umbral,
        "temperatura": 1.0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== Portando modelos TensorFlow a ONNX (via SimpleCNN) ==")
    meta = {task: portar(task) for task in TASKS}
    with open(OUT_DIR / "manifest_tf.json", "w", encoding="utf-8") as f:
        json.dump({"v1_tf": meta}, f, indent=2, ensure_ascii=False)
    print(f"Metadata: {OUT_DIR / 'manifest_tf.json'}")


if __name__ == "__main__":
    main()
