"""SigmaRx7: Synthetic healthcare ETL/ELT pipeline for FHIR data processing.

This package provides a comprehensive pipeline for processing synthetic healthcare data,
including HL7/FHIR ingestion, medication normalization via RxNorm, overlap detection,
generic recommendations, payer rule alignment, and ML-based denial prediction.
"""

__version__ = "0.1.0"
__author__ = "SigmaRx7 Team"
__email__ = "team@sigma-rx7.com"

from .core.pipeline import SigmaRx7Pipeline
from .core.config import Config

__all__ = ["SigmaRx7Pipeline", "Config"]