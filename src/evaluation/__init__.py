"""Evaluation layer (Stage 6): normalization + metrics + three-arm comparison.

normalize.py is implemented (canonical scoring, used by the web app). metrics.py
and compare.py stay skeletons until the manual reference arm is finalized.
"""

from .normalize import (
    DOMAIN_VALUES,
    HYBRID_VALUES,
    canonical_domain,
    canonical_hybrid_set,
    field_score,
    lenient_score,
)

__all__ = [
    "DOMAIN_VALUES",
    "HYBRID_VALUES",
    "canonical_domain",
    "canonical_hybrid_set",
    "field_score",
    "lenient_score",
]
