"""Observed-data metrics for the pilot.

Inputs are per-item report probabilities -- the fraction of an item's samples whose verdict
was "incorrect". Items whose every sample failed to parse arrive as NaN and are excluded
rather than treated as zero, because a parse failure is missing data, not a "correct" verdict.
"""

from __future__ import annotations

import numpy as np


def far(report_probs: np.ndarray) -> float:
    """False-alarm rate on verified-correct items: the mean report probability. Primary DV."""
    return float(np.nanmean(report_probs))


def instability_index(report_probs: np.ndarray) -> float:
    """N1, the per-item verdict instability diagnostic.

    Gate 1 requires < 0.25: a near-coin-flip auditor makes every downstream contrast unmeasurable.
    **Distinct from N2**, the FAR-difference null band -- different units, never to be compared.
    """
    p = np.asarray(report_probs, dtype=float)
    return float(np.nanmean(2.0 * np.minimum(p, 1.0 - p)))
