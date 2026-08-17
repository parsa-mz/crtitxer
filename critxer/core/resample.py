"""Cluster-aware resampling for contrasts whose items share a frozen episode.

Every R4 cell cycles 465 targets over a pool of 50 frozen episodes, so each episode is reused for
about 9.3 targets. Item resampling is right for the ladder, where each item is an independent trace,
but wrong here: two targets sharing an episode see the same prior exchange, so treating them as
independent understates the variance of every R4 contrast. Resample episodes as whole units instead,
keeping each drawn episode's targets intact; it reduces to the item bootstrap when there is no
reuse, which `tests/test_resample.py` pins.

A bootstrap rather than a mixed-effects logistic model: no distributional assumption, the same
machinery as every other interval in the paper, and it cannot silently fail to converge.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.05


class Streams:
    """An independent RNG per (model, contrast), so an interval depends only on its own identity.

    Sharing one generator makes every p-value a function of how many draws were consumed before
    it, which moved numbers across Holm thresholds in both analyses here with the data untouched.

    Issuing the same identity twice **raises**: seeding from a name is only reproducible if the
    names are unique, and the way that breaks is a copy-pasted call site, which recouples two
    contrasts while looking correct in review.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._issued: set[tuple[str, str]] = set()

    def __call__(self, model: str, contrast: str) -> np.random.Generator:
        key = (model, contrast)
        if key in self._issued:
            raise SystemExit(
                f"stream {model}/{contrast} was already issued; two call sites share one identity, "
                "so their intervals are not independent. Give each its own name."
            )
        self._issued.add(key)
        return np.random.default_rng([self._seed, *[ord(ch) for ch in f"{model}/{contrast}"]])


def interval_from_reps(
    reps: np.ndarray,
    effect: float,
    alpha: float = ALPHA,
    n_boot: int | None = None,
) -> dict:
    """Percentile interval and the p-value that agrees with it.

    ``p`` is the achieved significance level *of the interval beside it*: twice the smaller tail
    mass either side of zero, which is below ``alpha`` exactly when the interval excludes zero. A
    null-centred p-value is a different test and disagrees on skewed replicate distributions, so a
    table drawing bold from one and stars from the other contradicts itself. ``p_centred`` is kept
    so the change is auditable, but nothing should test against it.
    """
    if n_boot is None:
        n_boot = int(np.size(reps))
    below = float((reps <= 0.0).mean())
    return {
        "effect": float(effect),
        "lo": float(np.quantile(reps, alpha / 2)),
        "hi": float(np.quantile(reps, 1 - alpha / 2)),
        "p": max(2.0 * min(below, 1.0 - below), 1.0 / n_boot),
        "p_centred": max(float((np.abs(reps - effect) >= abs(effect)).mean()), 1.0 / n_boot),
    }


def _draw_clusters(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One cluster-level resample, returned as item indices into the original arrays.

    Draws ``k`` clusters with replacement from the ``k`` present and concatenates each drawn
    cluster's members **intact**. Clusters are the sampling unit; their contents are not resampled.
    """
    c = np.asarray(clusters)
    labels = np.unique(c)
    if labels.size < 2:
        raise ValueError(f"need at least 2 clusters to resample; got {labels.size}")
    members = [np.flatnonzero(c == lab) for lab in labels]
    return np.concatenate([members[k] for k in rng.integers(0, labels.size, size=labels.size)])


def cluster_indices(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One cluster resample, as indices rather than as a mean.

    d' is a function of two rates on two *disjoint* arms, so it cannot use `cluster_bootstrap`.
    Exposing the draw keeps both on one implementation.
    """
    return _draw_clusters(clusters, rng)


def paired_bootstrap(
    diff: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 20000,
    alpha: float = ALPHA,
) -> dict:
    """Resample items only. Correct for the ladder, and kept here so both can be reported."""
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        raise ValueError("no non-NaN differences to bootstrap")
    reps = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    n = int(d.size)
    return {**interval_from_reps(reps, float(d.mean()), alpha, n_boot), "n": n, "n_clusters": n}


def cluster_bootstrap(
    diff: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 20000,
    alpha: float = ALPHA,
) -> dict:
    """Resample whole clusters with replacement. The estimator every R4 contrast uses.

    ``clusters`` labels each item with the episode it was paired to. Cluster sizes are near equal
    by construction (the pool is cycled), but the implementation does not rely on that.
    """
    d = np.asarray(diff, dtype=float)
    c = np.asarray(clusters)
    if d.shape != c.shape:
        raise ValueError(f"diff and clusters must align; got {d.shape} and {c.shape}")
    keep = ~np.isnan(d)
    d, c = d[keep], c[keep]
    labels = np.unique(c)
    if labels.size < 2:
        raise ValueError(f"need at least 2 clusters to resample; got {labels.size}")
    reps = np.array([d[_draw_clusters(c, rng)].mean() for _ in range(n_boot)])
    return {
        **interval_from_reps(reps, float(d.mean()), alpha, n_boot),
        "n": int(d.size),
        "n_clusters": int(labels.size),
    }


def two_stage_bootstrap(
    diff: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 20000,
    alpha: float = ALPHA,
) -> dict:
    """DEPRECATED and miscalibrated. Kept only so the numbers it produced remain reproducible.

    Resampling items *within* each drawn cluster is the error: the drawn cluster's mean is already
    the quantity whose variability is being propagated, so re-drawing its members adds variance
    the sampling distribution does not have. Coverage at this project's geometry is 0.98 against a
    nominal 0.95, where `cluster_bootstrap` gives 0.94, and it withdrew a real effect on that basis.

    **No caller in this repository may use it.** `tests/test_resample.py` pins the miscalibration.
    """
    d = np.asarray(diff, dtype=float)
    c = np.asarray(clusters)
    if d.shape != c.shape:
        raise ValueError(f"diff and clusters must align; got {d.shape} and {c.shape}")
    keep = ~np.isnan(d)
    d, c = d[keep], c[keep]
    # Grouped as index arrays once, so the resampling loop does no searching.
    labels = np.unique(c)
    if labels.size < 2:
        raise ValueError(
            f"two_stage_bootstrap needs at least 2 clusters to resample; got {labels.size}"
        )
    members = [np.flatnonzero(c == lab) for lab in labels]

    reps = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, labels.size, size=labels.size)
        picked = [
            members[k][rng.integers(0, members[k].size, size=members[k].size)] for k in drawn
        ]
        reps[b] = d[np.concatenate(picked)].mean()
    return {
        **interval_from_reps(reps, float(d.mean()), alpha, n_boot),
        "n": int(d.size),
        "n_clusters": int(labels.size),
    }


def episode_ids_for(n_items: int, n_episodes: int) -> np.ndarray:
    """Reconstruct which episode each target was paired to.

    `cli/run/r4.py` gives target *i* (in persisted `item_ids` order) the episode at
    ``i % len(pool)``; cells generated before this module existed did not record it. Reconstructed
    here and pinned by tests rather than re-derived per call site, because a silently wrong
    reconstruction produces confident and wrongly-clustered intervals.

    New runs persist ``episode_ids`` directly and should prefer that.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1; got {n_episodes}")
    return np.arange(n_items) % n_episodes
