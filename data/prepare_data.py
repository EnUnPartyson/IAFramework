from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Union

from PIL import Image, UnidentifiedImageError
from torchvision.datasets import STL10, Places365

NoneSource = Union[Path, Image.Image]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"

IMG_SIZE = 128
SEED = 42
# Desbalance intencional del Modelo 1 (ver DECISIONS.md)
PROPORTIONS = {"perro": 0.35, "gato": 0.35, "ninguno": 0.30}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
# cuota pareja entre fuentes de "ninguno" -- si se samplea del pool combinado sin esto, Food101
# (75k imagenes) domina sobre STL10 (10k) y diluye la diversidad que se buscaba
NONE_SOURCE_WEIGHTS = {"food101": 1 / 3, "stl10": 1 / 3, "places365": 1 / 3}
PLACES365_EXCLUDE_KEYWORDS = ("dog", "cat", "kennel", "pet", "veterinar")

# Razas: se eligen las N con MAS imagenes disponibles. Perros combina 3 fuentes
# (Stanford + Oxford + Tsinghua, ~600+ imgs/raza en el top); gatos solo tiene Oxford
# (~200 imgs/raza), por eso el numero de clases viable es distinto.
DOG_BREEDS_TOP_N = 20
CAT_BREEDS_TOP_N = 12

# Razas fuera del top-N: se guarda una muestra como "desconocidas" para evaluar el
# modo "raza no identificada" (umbral de confianza) contra razas nunca vistas.
UNKNOWN_SAMPLE_SIZE = 300


def _progress(done: int, total: int, label: str, every: int = 2000) -> None:
    if done % every == 0 or done == total:
        print(f"    {label}: {done}/{total}", flush=True)


def _valid_images(paths: list[Path], label: str = "verificando") -> list[Path]:
    valid = []
    for i, path in enumerate(paths, 1):
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append(path)
        except (UnidentifiedImageError, OSError):
            continue
        _progress(i, len(paths), label)
    return valid


def _collect_cats_dogs() -> tuple[list[Path], list[Path]]:
    pet_images = RAW_DIR / "cats_dogs" / "PetImages"
    print("  Verificando imagenes de Cats&Dogs (descarta las corruptas, tarda unos minutos)...")
    cats = _valid_images(sorted((pet_images / "Cat").glob("*.jpg")), "gatos verificados")
    dogs = _valid_images(sorted((pet_images / "Dog").glob("*.jpg")), "perros verificados")
    return cats, dogs


def _collect_coco_pet_in_context(rng: random.Random) -> tuple[list[Path], list[Path]]:
    # Fotos de perro/gato dentro de una escena real (no primer plano de estudio como Cats&Dogs), para
    # que el detector no aprenda a distinguir clases por estilo de composicion en vez de por el animal.
    ann_path = RAW_DIR / "coco" / "annotations" / "instances_val2017.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco_index = json.load(f)

    category_ids = {c["name"]: c["id"] for c in coco_index["categories"]}
    dog_image_ids = {a["image_id"] for a in coco_index["annotations"] if a["category_id"] == category_ids["dog"]}
    cat_image_ids = {a["image_id"] for a in coco_index["annotations"] if a["category_id"] == category_ids["cat"]}
    both = dog_image_ids & cat_image_ids  # imagenes con ambas especies: ambiguas para un clasificador de 1 etiqueta

    id_to_filename = {img["id"]: img["file_name"] for img in coco_index["images"]}
    coco_images_dir = RAW_DIR / "coco" / "val2017"

    dog_paths = [coco_images_dir / id_to_filename[i] for i in (dog_image_ids - both)]
    cat_paths = [coco_images_dir / id_to_filename[i] for i in (cat_image_ids - both)]
    rng.shuffle(dog_paths)
    rng.shuffle(cat_paths)
    return dog_paths, cat_paths


def _collect_food101_candidates(quota: int, rng: random.Random) -> list[Path]:
    food_images = RAW_DIR / "food101" / "food-101" / "images"
    paths = sorted(food_images.glob("*/*.jpg"))
    rng.shuffle(paths)
    return paths[:quota]


def _collect_stl10_non_pet_candidates(quota: int, rng: random.Random) -> list[Image.Image]:
    # STL10 trae "cat" y "dog" entre sus 10 clases; se excluyen para no contaminar la clase "ninguno"
    # con los mismos animales que el detector tiene que reconocer.
    print("  Leyendo STL10 (excluye sus clases cat/dog)...", flush=True)
    all_items: list[Image.Image] = []
    for split in ("train", "test"):
        dataset = STL10(root=str(RAW_DIR / "stl10"), split=split, download=False)
        exclude = {dataset.classes.index("cat"), dataset.classes.index("dog")}
        for img, label in dataset:
            if label not in exclude:
                all_items.append(img)
    rng.shuffle(all_items)
    return all_items[:quota]


def _collect_places365_candidates(quota: int, rng: random.Random) -> list[Image.Image]:
    # Paisajes/escenas reales (playa, montana, bosque, calle, etc.). Se descartan por nombre las
    # categorias relacionadas con perros/gatos/veterinarias para no contaminar "ninguno".
    dataset = Places365(root=str(RAW_DIR / "places365"), split="val", small=True, download=False)
    excluded_idx = {
        i for i, name in enumerate(dataset.classes)
        if any(kw in name.lower() for kw in PLACES365_EXCLUDE_KEYWORDS)
    }

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    print(f"  Leyendo hasta {quota} escenas de Places365...", flush=True)
    images: list[Image.Image] = []
    for idx in indices:
        if len(images) >= quota:
            break
        img, label = dataset[idx]
        if label not in excluded_idx:
            images.append(img)
            _progress(len(images), quota, "escenas leidas")
    return images


def _collect_none_candidates(total_needed: int, rng: random.Random) -> list[NoneSource]:
    quota_food101 = int(total_needed * NONE_SOURCE_WEIGHTS["food101"])
    quota_stl10 = int(total_needed * NONE_SOURCE_WEIGHTS["stl10"])
    quota_places365 = total_needed - quota_food101 - quota_stl10

    return [
        *_collect_food101_candidates(quota_food101, rng),
        *_collect_stl10_non_pet_candidates(quota_stl10, rng),
        *_collect_places365_candidates(quota_places365, rng),
    ]


def _split_counts(total: int) -> dict[str, int]:
    train = int(total * SPLIT_RATIOS["train"])
    val = int(total * SPLIT_RATIOS["val"])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def _save_one(img: Image.Image, out_dir: Path, index: int) -> None:
    img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(out_dir / f"{index:05d}.jpg", quality=90)


def _save_resized(sources: list[NoneSource], out_dir: Path, label: str = "") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if sources:
        print(f"  Guardando {len(sources)} imagenes en {label or out_dir.name}...", flush=True)
    saved = 0
    for source in sources:
        try:
            if isinstance(source, Path):
                with Image.open(source) as img:
                    _save_one(img, out_dir, saved)
            else:
                _save_one(source, out_dir, saved)
            saved += 1
        except (UnidentifiedImageError, OSError):
            continue
    return saved


def _materialize_splits(class_paths: dict[str, list[NoneSource]], out_root: Path) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for class_name, paths in class_paths.items():
        counts = _split_counts(len(paths))
        offset = 0
        distribution[class_name] = {}
        for split, count in counts.items():
            split_paths = paths[offset:offset + count]
            offset += count
            out_dir = out_root / split / class_name
            saved = _save_resized(split_paths, out_dir, f"{class_name}/{split}")
            distribution[class_name][split] = saved
    return distribution


def prepare_detector_data() -> dict[str, dict[str, int]]:
    rng = random.Random(SEED)

    cats, dogs = _collect_cats_dogs()
    coco_dogs, coco_cats = _collect_coco_pet_in_context(rng)
    dogs = dogs + coco_dogs
    cats = cats + coco_cats

    # ancla = tamano disponible de perro/gato; "ninguno" se deriva de PROPORTIONS para no hardcodear 35/35/30
    anchor = min(len(cats), len(dogs))
    none_needed = int(anchor * 2 * PROPORTIONS["ninguno"] / (PROPORTIONS["perro"] + PROPORTIONS["gato"]))
    none_candidates = _collect_none_candidates(none_needed, rng)

    rng.shuffle(cats)
    rng.shuffle(dogs)
    rng.shuffle(none_candidates)

    class_paths: dict[str, list[NoneSource]] = {
        "gato": list(cats[:anchor]),
        "perro": list(dogs[:anchor]),
        "ninguno": none_candidates,
    }
    return _materialize_splits(class_paths, PROCESSED_ROOT / "detector")


def _oxford_breeds(want_cats: bool) -> dict[str, list[Path]]:
    """Oxford-IIIT Pet: archivos "Raza_123.jpg"; inicial mayuscula = gato, minuscula = perro."""
    images_root = RAW_DIR / "oxford_pets" / "images"
    breeds: dict[str, list[Path]] = {}
    for path in sorted(images_root.glob("*.jpg")):
        raw_name = path.stem.rsplit("_", 1)[0]
        if raw_name[0].isupper() == want_cats:
            breeds.setdefault(raw_name.lower(), []).append(path)
    return breeds


def _top_breeds_to_splits(breeds: dict[str, list[Path]], top_n: int, out_name: str) -> dict[str, dict[str, int]]:
    rng = random.Random(SEED)
    ranked = sorted(breeds.items(), key=lambda kv: len(kv[1]), reverse=True)
    top, excluded = ranked[:top_n], ranked[top_n:]
    print(f"  Razas elegidas (las {top_n} con mas imagenes): " + ", ".join(f"{n} ({len(p)})" for n, p in top))

    class_paths: dict[str, list[NoneSource]] = {}
    for breed_name, paths in top:
        valid = _valid_images(paths, f"{breed_name} verificadas")
        rng.shuffle(valid)
        class_paths[breed_name] = list(valid)
    distribution = _materialize_splits(class_paths, PROCESSED_ROOT / out_name)

    # muestra de razas excluidas -> "desconocidas": evalua el modo "raza no identificada"
    if excluded:
        pool: list[Path] = [p for _, paths in excluded for p in paths]
        rng.shuffle(pool)
        sample = _valid_images(pool[: UNKNOWN_SAMPLE_SIZE * 2], "desconocidas verificadas")[:UNKNOWN_SAMPLE_SIZE]
        saved = _save_resized(sample, PROCESSED_ROOT / out_name / "unknown" / "desconocida", f"{out_name}/unknown")
        distribution["_desconocidas"] = {"unknown": saved}
        print(f"  Muestra de razas desconocidas ({len(excluded)} razas excluidas): {saved} imagenes")
    return distribution


def _add_imagenet_style_dirs(breeds: dict[str, list[Path]], root: Path) -> int:
    """Suma carpetas estilo "nXXXXXXXX-nombre_de_raza" (Stanford y Tsinghua usan ese formato)."""
    found = 0
    for breed_dir in sorted(root.rglob("n*-*")):
        if not breed_dir.is_dir():
            continue
        images = sorted(breed_dir.glob("*.jpg")) + sorted(breed_dir.glob("*.jpeg"))
        if not images:
            continue
        breed_name = breed_dir.name.split("-", 1)[1].lower().replace(" ", "_")
        breeds.setdefault(breed_name, []).extend(images)
        found += 1
    return found


def prepare_dog_breed_data() -> dict[str, dict[str, int]]:
    # Tres fuentes combinadas por nombre de raza normalizado: Stanford (~20k), Tsinghua (~70k)
    # y las razas de perro de Oxford-IIIT Pet (~5k). Las razas presentes en varias fuentes
    # suman todos sus ejemplos, por eso el top-N queda con ~600+ imagenes por raza.
    breeds: dict[str, list[Path]] = {}
    n_stanford = _add_imagenet_style_dirs(breeds, RAW_DIR / "stanford_dogs" / "Images")
    n_tsinghua = _add_imagenet_style_dirs(breeds, RAW_DIR / "tsinghua_dogs")
    print(f"  Carpetas de raza encontradas: Stanford={n_stanford}, Tsinghua={n_tsinghua}")
    if n_tsinghua == 0:
        print("  AVISO: no se encontro Tsinghua Dogs; corre data/download_dataset.py para sumarlo")

    for breed_name, paths in _oxford_breeds(want_cats=False).items():
        breeds.setdefault(breed_name, []).extend(paths)

    return _top_breeds_to_splits(breeds, DOG_BREEDS_TOP_N, "dog_breed")


def prepare_cat_breed_data() -> dict[str, dict[str, int]]:
    return _top_breeds_to_splits(_oxford_breeds(want_cats=True), CAT_BREEDS_TOP_N, "cat_breed")


def _report(task: str, distribution: dict[str, dict[str, int]]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"{task}_data_distribution.json", "w", encoding="utf-8") as f:
        json.dump(distribution, f, indent=2, ensure_ascii=False)
    total = sum(sum(splits.values()) for splits in distribution.values())
    print(f"[{task}] {len(distribution)} clases, {total} imagenes totales")
    for class_name, splits in distribution.items():
        print(f"  {class_name}: {sum(splits.values())} {splits}")


def main() -> None:
    print("Preparando datos del detector (Modelo 1)...")
    _report("detector", prepare_detector_data())
    print("Preparando datos de raza de perro (Modelo 2)...")
    _report("dog_breed", prepare_dog_breed_data())
    print("Preparando datos de raza de gato (Modelo 3)...")
    _report("cat_breed", prepare_cat_breed_data())


if __name__ == "__main__":
    main()
