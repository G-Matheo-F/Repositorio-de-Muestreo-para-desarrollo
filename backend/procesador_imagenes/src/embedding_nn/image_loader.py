from pathlib import Path
import cv2
import numpy as np


def cargar_imagenes_procesadas(ruta_dataset: Path, image_size):
    X = []
    rutas = []
    clases = []

    carpeta_base = ruta_dataset / "imagenes_procesadas"

    for carpeta_clase in carpeta_base.iterdir():
        if not carpeta_clase.is_dir():
            continue

        for carpeta_img in carpeta_clase.iterdir():
            if not carpeta_img.is_dir():
                continue

            for img_path in carpeta_img.glob("preprocesada_*.jpg"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, image_size)
                img = img.astype(np.float32) / 255.0

                X.append(img.flatten())
                rutas.append(str(img_path))
                clases.append(carpeta_clase.name)

    return np.array(X), rutas, clases
