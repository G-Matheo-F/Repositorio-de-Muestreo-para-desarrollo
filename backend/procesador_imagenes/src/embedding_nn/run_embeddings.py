from pathlib import Path
from .embedding_service import generar_embeddings_dataset


def run():
    ruta_results = Path(__file__).resolve().parents[2] / "results"

    for dataset in ruta_results.iterdir():
        if dataset.is_dir():
            generar_embeddings_dataset(dataset)

