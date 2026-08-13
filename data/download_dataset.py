from __future__ import annotations

from pathlib import Path

from torchvision.datasets import STL10, Food101, Places365
from torchvision.datasets.utils import download_and_extract_archive

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Mirror oficial de Microsoft del dataset "Kaggle Cats and Dogs" (Asirra), sin login requerido.
CATS_DOGS_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
CATS_DOGS_DIR = RAW_DIR / "cats_dogs"
FOOD101_DIR = RAW_DIR / "food101"
STL10_DIR = RAW_DIR / "stl10"
PLACES365_DIR = RAW_DIR / "places365"
COCO_DIR = RAW_DIR / "coco"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Modelo 2 (raza perro): Stanford Dogs, descarga directa sin credenciales.
STANFORD_DOGS_DIR = RAW_DIR / "stanford_dogs"
STANFORD_DOGS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"

# Modelo 3 (raza gato): Oxford-IIIT Pet, descarga directa sin credenciales.
# Los archivos con inicial mayuscula son gatos (12 razas); los de minuscula son perros y se ignoran.
OXFORD_PETS_DIR = RAW_DIR / "oxford_pets"
OXFORD_PETS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"


def download_cats_dogs() -> None:
    if (CATS_DOGS_DIR / "PetImages").exists():
        print(f"cats_dogs ya existe en {CATS_DOGS_DIR}, se omite descarga")
        return
    CATS_DOGS_DIR.mkdir(parents=True, exist_ok=True)
    download_and_extract_archive(CATS_DOGS_URL, download_root=str(CATS_DOGS_DIR))


def download_food101() -> None:
    # Fuente de la clase "ninguno": fotos reales sin perros/gatos, resolucion comparable a PetImages.
    FOOD101_DIR.mkdir(parents=True, exist_ok=True)
    Food101(root=str(FOOD101_DIR), split="train", download=True)


def download_stl10() -> None:
    # Segunda fuente de "ninguno": aporta otros animales/objetos/vehiculos reales (avion, pajaro, auto,
    # ciervo, caballo, mono, barco, camion) para que el detector no aprenda solo "no es comida".
    # Las clases "cat"/"dog" de STL10 se descartan en prepare_data.py, no aca.
    STL10_DIR.mkdir(parents=True, exist_ok=True)
    STL10(root=str(STL10_DIR), split="train", download=True)
    STL10(root=str(STL10_DIR), split="test", download=True)


def download_places365() -> None:
    # Tercera fuente de "ninguno": paisajes/escenas reales (playa, montana, bosque, desierto, calle, etc.)
    # para que el detector tambien vea fondos variados, no solo objetos en primer plano.
    # Categorias tipo "kennel"/"pet_shop"/"veterinarians_office" se descartan en prepare_data.py.
    PLACES365_DIR.mkdir(parents=True, exist_ok=True)
    Places365(root=str(PLACES365_DIR), split="val", small=True, download=True)


def download_coco_val() -> None:
    # Fotos de perro/gato "en contexto" (dentro de una escena real), a diferencia de Cats&Dogs que
    # son primer plano de la mascota. Solo se usa el split val2017 (mucho mas liviano que train2017).
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    if (COCO_DIR / "val2017").exists():
        print(f"coco val2017 ya existe en {COCO_DIR}, se omite descarga de imagenes")
    else:
        download_and_extract_archive(COCO_VAL_IMAGES_URL, download_root=str(COCO_DIR))

    if (COCO_DIR / "annotations" / "instances_val2017.json").exists():
        print("coco annotations ya existen, se omite descarga")
    else:
        download_and_extract_archive(COCO_ANNOTATIONS_URL, download_root=str(COCO_DIR))


def download_stanford_dogs() -> None:
    if (STANFORD_DOGS_DIR / "Images").exists():
        print(f"stanford_dogs ya existe en {STANFORD_DOGS_DIR}, se omite descarga")
        return
    STANFORD_DOGS_DIR.mkdir(parents=True, exist_ok=True)
    download_and_extract_archive(STANFORD_DOGS_URL, download_root=str(STANFORD_DOGS_DIR))


def download_oxford_pets() -> None:
    if (OXFORD_PETS_DIR / "images").exists():
        print(f"oxford_pets ya existe en {OXFORD_PETS_DIR}, se omite descarga")
        return
    OXFORD_PETS_DIR.mkdir(parents=True, exist_ok=True)
    download_and_extract_archive(OXFORD_PETS_URL, download_root=str(OXFORD_PETS_DIR))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Descargando Cats vs Dogs (clases perro/gato)...")
    download_cats_dogs()
    print("Descargando COCO val2017 (perro/gato en contexto)...")
    download_coco_val()
    print("Descargando Food101 (fuente de la clase 'ninguno')...")
    download_food101()
    print("Descargando STL-10 (segunda fuente de la clase 'ninguno')...")
    download_stl10()
    print("Descargando Places365 (tercera fuente de la clase 'ninguno': paisajes/escenas)...")
    download_places365()
    print("Descargando Stanford Dogs (Modelo 2: raza perro)...")
    download_stanford_dogs()
    print("Descargando Oxford-IIIT Pet (Modelo 3: raza gato)...")
    download_oxford_pets()
    print("Descarga completa.")


if __name__ == "__main__":
    main()
