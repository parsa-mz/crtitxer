"""Detection-side metrics for the labelled-incorrect arm.

A false-alarm rate on verified-correct traces cannot distinguish three things that all lower it:
better calibration, uniform leniency, and genuinely improved discrimination. Only the second is bad
news. These metrics supply the other half of the confusion matrix, the decisive pair being
`sensitivity_index` (d'), which responds to discrimination and is blind to a pure threshold shift,
and `criterion` (c), the reverse.

ProcessBench marks a flawed trace with ``label`` = the **0-based** first bad step; the audit schema
reports ``first_error_step`` **1-based**. That conversion lives in `localisation_accuracy` and
nowhere else, because an off-by-one does not raise -- it silently reports localisation near zero,
which looks like a finding.
"""

from __future__ import annotations

import numpy as np

# Standard log-linear correction for rates at 0 or 1, which would otherwise send d' to infinity and
# drop the condition out of every comparison. Applied as (x*n + 0.5) / (n + 1).
_CORRECTION = 0.5


def _z(rate: float, n: int) -> float:
    from scipy.stats import norm

    adjusted = (rate * n + _CORRECTION) / (n + 2 * _CORRECTION)
    return float(norm.ppf(adjusted))


def localisation_accuracy(
    reported_steps: list[int | None],
    gold_labels: list[int],
) -> float:
    """Share of *localisable* attempts that name the right step.

    ``gold_labels`` are ProcessBench's 0-based indices, so a report of ``k`` (1-based) is correct
    when ``k == label + 1``. Items where no step was reported are excluded rather than scored wrong:
    counting them would fold the detection rate into this metric. NaN when nothing was localisable.
    """
    if len(reported_steps) != len(gold_labels):
        raise ValueError(
            f"reported_steps and gold_labels must align; got {len(reported_steps)} and "
            f"{len(gold_labels)}"
        )
    hits = [
        int(step == label + 1)
        for step, label in zip(reported_steps, gold_labels, strict=True)
        if step is not None
    ]
    return float(np.mean(hits)) if hits else float("nan")


def sensitivity_index(detection: float, far: float, n_signal: int, n_noise: int) -> float:
    """d' = z(detection) - z(FAR): discrimination, invariant to a pure threshold shift."""
    return _z(detection, n_signal) - _z(far, n_noise)


def criterion(detection: float, far: float, n_signal: int, n_noise: int) -> float:
    """c = -(z(detection) + z(FAR))/2: where the threshold sits, higher being more conservative."""
    return -0.5 * (_z(detection, n_signal) + _z(far, n_noise))


def balanced_accuracy(detection: float, far: float) -> float:
    """(TPR + TNR) / 2. Chance is 0.5, including for an auditor that flags everything."""
    return 0.5 * (detection + (1.0 - far))


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Binned |confidence - accuracy|, weighted by bin occupancy.

    Whether a shifted threshold is tracked by the schema's self-reported confidence is a separate
    question from whether the verdict moves. Empty bins contribute nothing, so the score does not
    drift with ``n_bins`` on fixed data.
    """
    conf = np.asarray(confidence, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if conf.shape != acc.shape:
        raise ValueError(f"confidence and correct must align; got {conf.shape} and {acc.shape}")
    keep = ~(np.isnan(conf) | np.isnan(acc))
    conf, acc = conf[keep], acc[keep]
    if conf.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so a confidence of exactly 1.0 lands in the top bin rather than out of range
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        if not in_bin.any():
            continue
        total += in_bin.mean() * abs(conf[in_bin].mean() - acc[in_bin].mean())
    return float(total)
