"""Prepara los splits del modo PRO en data/processed/<tarea>_pro/. No descarga ni entrena.

Reutiliza los colectores de prepare_data.py (mismo codigo, misma verificacion de imagenes,
mismo SEED) y suma las fuentes nuevas de download_dataset_pro.py:

  - detector_pro : fuentes base + AFHQ (perro/gato de alta calidad; "salvaje" va a la clase
                   "ninguno" como negativo dificil: animales que parecen mascota y no lo son)
  - dog_breed_pro: mismas 3 fuentes de la v1 pero top-30 razas (con transfer learning
                   alcanzan menos imagenes por clase) y guardado a mayor resolucion
  - cat_breed_pro: Oxford + el dataset de PetFinder (~67 etiquetas). Se excluyen las
                   etiquetas de PELAJE que no son razas (domestic short hair, tabby, calico,
                   ...): son las mas numerosas del dataset y contaminarian el top-N

Los splits de la v1 (data/processed/<tarea>/) no se tocan: la presentacion queda intacta.

Uso:
    python data/prepare_data_pro.py                    # las 3 tareas
    python data/prepare_data_pro.py --tasks cat_breed
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

_spec = importlib.util.spec_from_file_location("prepare_data", Path(__file__).resolve().parent / "prepare_data.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

RAW_DIR = base.RAW_DIR
PROCESSED_ROOT = base.PROCESSED_ROOT
SEED = base.SEED

# resoluciones de guardado del pro: se entrena a 224 (la nativa de los preentrenados),
# asi que se guarda con margen para que RandomResizedCrop recorte de verdad
DETECTOR_PRO_STORED = 224
BREED_PRO_STORED = 256

DOG_BREEDS_PRO_TOP_N = 30
CAT_BREEDS_PRO_TOP_N = 25
# tope por raza: PetFinder es muy desparejo (miles de "siamese" vs cientos de otras);
# sin tope, una raza dominaria el dataset y el peso del desbalance se iria a los extremos
MAX_POR_RAZA = 3000

# etiquetas del dataset de PetFinder que describen PELAJE o condicion, no raza
NO_RAZAS = frozenset({
    "domestic_short_hair", "domestic_medium_hair", "domestic_long_hair",
    "tabby", "tortoiseshell", "calico", "tuxedo", "torbie",
    "dilute_calico", "dilute_tortoiseshell",
    "extra-toes_cat_-_hemingway_polydactyl", "extra-toes_cat_(hemingway_polydactyl)",
})


def _norm(nombre: str) -> str:
    return nombre.strip().lower().replace(" ", "_")


def _collect_afhq() -> dict[str, list[Path]]:
    """AFHQ: carpetas cat/dog/wild (en train/ y val/; se juntan, los splits los hace este repo)."""
    root = RAW_DIR / "afhq"
    out: dict[str, list[Path]] = {"cat": [], "dog": [], "wild": []}
    if not root.exists():
        print("  AVISO: falta data/raw/afhq (correr data/download_dataset_pro.py); se sigue sin AFHQ")
        return out
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and p.name in out):
        out[d.name].extend(sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")))
    print(f"  AFHQ: {len(out['dog'])} perros, {len(out['cat'])} gatos, {len(out['wild'])} salvajes")
    return out


def _collect_petfinder_cats() -> dict[str, list[Path]]:
    """Dataset de razas de gato de PetFinder: carpetas <Raza>/*.jpg (se autodetecta la raiz)."""
    root = RAW_DIR / "cat_breeds_extra"
    breeds: dict[str, list[Path]] = {}
    if not root.exists():
        print("  AVISO: falta data/raw/cat_breeds_extra (correr data/download_dataset_pro.py)")
        return breeds
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        images = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
        if len(images) < 30:  # carpetas auxiliares o razas demasiado chicas
            continue
        nombre = _norm(d.name)
        if nombre in NO_RAZAS:
            continue
        breeds.setdefault(nombre, []).extend(images)
    print(f"  PetFinder: {len(breeds)} razas de gato utilizables (excluidas las etiquetas de pelaje)")
    return breeds


def prepare_detector_pro() -> dict[str, dict[str, int]]:
    rng = random.Random(SEED)

    cats, dogs = base._collect_cats_dogs()
    coco_dogs, coco_cats = base._collect_coco_pet_in_context(rng)
    boost_dogs, boost_cats = base._collect_breed_reinforcement(rng)
    afhq = _collect_afhq()
    dogs = dogs + coco_dogs + boost_dogs + afhq["dog"]
    cats = cats + coco_cats + boost_cats + afhq["cat"]

    anchor = min(len(cats), len(dogs))
    none_needed = int(anchor * 2 * base.PROPORTIONS["ninguno"] / (base.PROPORTIONS["perro"] + base.PROPORTIONS["gato"]))
    # los salvajes de AFHQ entran a "ninguno" y descuentan cuota de las fuentes base
    wild = list(afhq["wild"])
    none_candidates = base._collect_none_candidates(max(0, none_needed - len(wild)), rng) + wild

    rng.shuffle(cats)
    rng.shuffle(dogs)
    rng.shuffle(none_candidates)

    class_paths = {
        "gato": list(cats[:anchor]),
        "perro": list(dogs[:anchor]),
        "ninguno": none_candidates,
    }
    return base._materialize_splits(class_paths, PROCESSED_ROOT / "detector_pro", size=DETECTOR_PRO_STORED)


def _cap_por_raza(breeds: dict[str, list[Path]], rng: random.Random) -> None:
    for nombre, paths in breeds.items():
        if len(paths) > MAX_POR_RAZA:
            rng.shuffle(paths)
            breeds[nombre] = paths[:MAX_POR_RAZA]


def prepare_dog_breed_pro() -> dict[str, dict[str, int]]:
    rng = random.Random(SEED)
    breeds: dict[str, list[Path]] = {}
    base._add_imagenet_style_dirs(breeds, RAW_DIR / "stanford_dogs" / "Images")
    base._add_imagenet_style_dirs(breeds, RAW_DIR / "tsinghua_dogs")
    for nombre, paths in base._oxford_breeds(want_cats=False).items():
        breeds.setdefault(nombre, []).extend(paths)
    _cap_por_raza(breeds, rng)
    base.BREED_STORED_SIZE = BREED_PRO_STORED
    return base._top_breeds_to_splits(breeds, DOG_BREEDS_PRO_TOP_N, "dog_breed_pro")


def prepare_cat_breed_pro() -> dict[str, dict[str, int]]:
    rng = random.Random(SEED)
    breeds = _collect_petfinder_cats()
    for nombre, paths in base._oxford_breeds(want_cats=True).items():
        breeds.setdefault(nombre, []).extend(paths)
    _cap_por_raza(breeds, rng)
    base.BREED_STORED_SIZE = BREED_PRO_STORED
    return base._top_breeds_to_splits(breeds, CAT_BREEDS_PRO_TOP_N, "cat_breed_pro")


TASKS = {
    "detector": prepare_detector_pro,
    "dog_breed": prepare_dog_breed_pro,
    "cat_breed": prepare_cat_breed_pro,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara los splits del modo pro (no toca los de la v1)")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    args = parser.parse_args()

    resumen = {}
    for task in args.tasks:
        print(f"== Preparando {task}_pro ==")
        resumen[task] = TASKS[task]()

    out = base.METRICS_DIR / "distribucion_datos_pro.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"Distribucion guardada en {out}")


if __name__ == "__main__":
    main()
