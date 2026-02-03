import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .image_loader import cargar_imagenes_procesadas
from .cnn_bottleneck_model import construir_cnn_bottleneck
from .config import (
    IMAGE_SIZE,
    EMBEDDING_DIM,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE
)


def generar_embeddings_dataset(ruta_dataset):
    print(f"Generando embeddings para {ruta_dataset.name}")

    X, rutas, clases = cargar_imagenes_procesadas(
        ruta_dataset,
        IMAGE_SIZE
    )

    if X.shape[0] == 0:
        print("No se encontraron imágenes procesadas")
        return

    # Codificar clases (entrenamiento supervisado)
    encoder_clases = LabelEncoder()
    y = encoder_clases.fit_transform(clases)

    model, embedding_model = construir_cnn_bottleneck(
        IMAGE_SIZE,
        EMBEDDING_DIM,
        LEARNING_RATE
    )

    model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        shuffle=True,
        verbose=1
    )

    embeddings = embedding_model.predict(X, verbose=1)

    carpeta_salida = ruta_dataset / "embeddings_nn"
    carpeta_salida.mkdir(exist_ok=True)

    df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )

    df.insert(0, "clase", clases)
    df.insert(0, "rutaImagen", rutas)

    ruta_csv = carpeta_salida / "embeddings.csv"
    df.to_csv(ruta_csv, index=False)

    print(f"✔ Embeddings guardados en {ruta_csv}")
