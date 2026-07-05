"""Task layer (Stage 4-6): screening / extraction / synthesis over the indexed corpus.

Each task exposes its core per-item logic + data loaders; the top-level experiment
scripts (screening_h1.py, full_extraction_101.py, synthesis_demo.py, ...) orchestrate
sampling, ablation sweeps and metric aggregation around these modules.
"""

from . import screening, extraction, synthesis

__all__ = ["screening", "extraction", "synthesis"]
