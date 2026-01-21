"""
Módulo de extracción de características
"""

from .extractorMomentos import ExtractorMomentos
from .extractorSift import ExtractorSift
from .extractorHog import ExtractorHog

__all__ = ['ExtractorMomentos', 'ExtractorSift', 'ExtractorHog']
