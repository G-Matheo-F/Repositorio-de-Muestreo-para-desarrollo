"""
Procesador principal de características
"""

import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path

from ..extraccionCaracteristicas import (
    ExtractorMomentos,
    ExtractorSift,
    ExtractorHog
)


class ProcesadorCaracteristicas:
    """
    Procesa imágenes y extrae características con múltiples algoritmos
    (entrada: matrices ya preprocesadas).
    """

    def __init__(self, tamanoObjetivo: tuple = (224, 224)):
        """
        Inicializa el procesador de características

        Args:
            tamanoObjetivo: Tamaño objetivo referencial para consistencia
        """
        self.tamanoObjetivo = tamanoObjetivo
        self.extractorMomentos = ExtractorMomentos()
        self.extractorSift = ExtractorSift()
        self.extractorHog = ExtractorHog()
        self.resultados = {}

    def extraerDesdePreprocesadas(self, rutaImagen: str,
                                   imagenGrayscale: np.ndarray,
                                   imagenBinaria: Optional[np.ndarray],
                                   algoritmos: List[str]) -> Dict:
        """
        Extrae características usando imágenes ya preprocesadas (sin volver a leer ni preprocesar).
        - Momentos: usa imagenBinaria si está disponible; en su defecto usa imagenGrayscale.
        - SIFT y HOG: usan imagenGrayscale.
        """
        if algoritmos is None:
            algoritmos = ['momentos', 'sift', 'hog']

        resultado = {
            'rutaImagen': str(rutaImagen),
            'caracteristicas': {},
            'analisisImagen': None,
            'estado': 'exito'
        }
        try:
            # Momentos
            if 'momentos' in algoritmos:
                base = imagenBinaria if imagenBinaria is not None else imagenGrayscale
                base_uint8 = (base * 255).astype(np.uint8) if base.dtype in (np.float32, np.float64) and base.max() <= 1 else base.astype(np.uint8)
                datosMomentos = self.extractorMomentos.extraerTodosMomentos(base_uint8)
                resultado['caracteristicas']['momentos'] = {
                    'momentosCrudos': datosMomentos['momentosCrudos'],
                    'momentosCentrales': datosMomentos['momentosCentrales'],
                    'momentosNormalizados': datosMomentos['momentosNormalizados'],
                    'momentosHu': datosMomentos['momentosHu'].tolist(),
                    'momentosZernike': datosMomentos['momentosZernike'].tolist(),
                    'tamanoVector': len(datosMomentos['vectorCaracteristicas']),
                    'vectorCaracteristicas': datosMomentos['vectorCaracteristicas'].tolist()
                }

            # SIFT
            if 'sift' in algoritmos:
                vecSift = self.extractorSift.extraerCaracteristicas(imagenGrayscale)
                resultado['caracteristicas']['sift'] = {
                    'tamanoVector': len(vecSift),
                    'vectorCaracteristicas': vecSift.tolist()
                }

            # HOG
            if 'hog' in algoritmos:
                vecHog = self.extractorHog.extraerCaracteristicas(imagenGrayscale)
                resultado['caracteristicas']['hog'] = {
                    'tamanoVector': len(vecHog),
                    'vectorCaracteristicas': vecHog.tolist()
                }

        except Exception as e:
            resultado['estado'] = 'error'
            resultado['error'] = str(e)

        return resultado
    
    def _calcularEstadisticas(self, resultados: Dict) -> Dict:
        """Calcula estadísticas de los resultados"""
        estad = {'exitosas': 0, 'fallidas': 0, 'tiposError': {}}

        for img in resultados['imagenes']:
            if img['estado'] == 'exito':
                estad['exitosas'] += 1
            else:
                estad['fallidas'] += 1
                err = img.get('error', 'desconocido')
                estad['tiposError'][err] = estad['tiposError'].get(err, 0) + 1
        return estad
    
    def guardarResultados(self, rutaSalida: str, formato: str = 'json'):
        """
        Guarda los resultados procesados
        
        Args:
            rutaSalida: Ruta de salida
            formato: Formato de salida ('json', 'npz')
        """
        Path(rutaSalida).parent.mkdir(parents=True, exist_ok=True)
        
        if formato == 'json':
            with open(rutaSalida, 'w') as f:
                json.dump(self.resultados, f, indent=2)
            print(f"Resultados guardados en {rutaSalida}")
        
        elif formato == 'npz':
            # Convertir a arrays numpy para formato comprimido
            npData = {}
            for i, img in enumerate(self.resultados.get('imagenes', [])):
                if img.get('estado') == 'exito':
                    feats = img.get('caracteristicas', {})
                    for algo, data in feats.items():
                        clave = f"img_{i}_{algo}"
                        if 'vectorCaracteristicas' in data:
                            npData[clave] = np.array(data['vectorCaracteristicas'])
            
            np.savez_compressed(rutaSalida, **npData)
            print(f"Resultados guardados en {rutaSalida}")
