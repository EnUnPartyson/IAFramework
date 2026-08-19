"""Modelos del modo pro: backbones preentrenados en ImageNet (transfer learning).

El modo pro es la rama posterior al tag v1-presentacion: ahi se levanta la restriccion de
entrenar desde cero y se usa solo PyTorch (el framework con mejor desempeno en la v1).
Los modelos base (SimpleCNN, desde cero, comparados TF vs PyTorch) viven en
model_defs_pytorch.py y no se tocan.

Nunca importar tensorflow aca.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

# arquitecturas soportadas y como se llama su capa clasificadora final.
# Todas con pesos DEFAULT de torchvision (los mas nuevos disponibles por arquitectura).
PRO_ARCHS = ("resnet18", "resnet50", "convnext_tiny", "efficientnet_v2_s")
PRO_ARCH_DEFAULT = "convnext_tiny"


def build_pretrained(arch: str, num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    """Backbone preentrenado con la capa final reemplazada por una para nuestras clases.

    `pretrained=False` existe solo para tests (evita descargar pesos); en entrenamiento
    real siempre va True, que es todo el punto del modo pro.
    """
    if arch not in PRO_ARCHS:
        raise ValueError(f"arch debe ser una de {PRO_ARCHS}, no {arch!r}")

    weights = "DEFAULT" if pretrained else None
    model = models.get_model(arch, weights=weights)

    # cada familia expone la capa final con otro nombre; el reemplazo resetea sus pesos,
    # que es lo que se quiere: la cabeza nueva aprende nuestras clases desde cero
    if arch.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch.startswith("convnext"):
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        # convnext ya trae LayerNorm+Flatten en classifier[0:2]; el dropout se inserta antes del Linear
        model.classifier = nn.Sequential(
            model.classifier[0], model.classifier[1], nn.Dropout(dropout), model.classifier[2]
        )
    elif arch.startswith("efficientnet"):
        # efficientnet ya trae Dropout en classifier[0]; se ajusta su p y se cambia el Linear
        model.classifier[0].p = dropout
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def head_parameters(model: nn.Module, arch: str) -> list[nn.Parameter]:
    """Parametros de la cabeza clasificadora (los unicos que se entrenan en el warmup)."""
    head = model.fc if arch.startswith("resnet") else model.classifier
    return list(head.parameters())


def freeze_backbone(model: nn.Module, arch: str, frozen: bool) -> None:
    """Congela (o descongela) todo menos la cabeza clasificadora.

    Fase 1 del fine-tuning: con el backbone congelado, la cabeza recien inicializada
    aprende sin que sus gradientes gigantes del arranque destruyan los pesos preentrenados.
    """
    head_ids = {id(p) for p in head_parameters(model, arch)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = not frozen
