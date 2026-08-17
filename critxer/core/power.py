"""Power and null-calibration simulation for the pilot.

The pilot's primary DV is a per-item *sampled* report probability, not a greedy verdict. Two
questions, answered before any GPU time is spent:

  1. How many items and samples-per-item are needed to detect the pre-registered 3pp minimum
     effect on false-alarm rate at 80% power?
  2. Is the gate-3 decision procedure calibrated -- does it fire on 5% of null studies rather
     than more? An uncalibrated gate would kill or spare the project for the wrong reason.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

_EPS = 1e-9


def _additive(p: np.ndarray, c: float) -> np.ndarray:
    return np.clip(p + c, 0.0, 1.0)


def _logit(p: np.ndarray, b: float) -> np.ndarray:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return 1.0 / (1.0 + np.exp(-(np.log(q / (1.0 - q)) + b)))


_MECHANISMS = {"additive": (_additive, 1.0), "logit": (_logit, 60.0)}


def apply_effect(p: np.ndarray, delta: float, mechanism: str) -> np.ndarray:
    """Shift per-item propensities so mean FAR moves by exactly ``delta``.

    ``additive`` moves every item equally (optimistic); ``logit`` shifts log-odds, so saturated
    items barely move (realistic, and harder to detect). Both solve for the parameter that lands
    the *mean* shift on ``delta``, so they compare at equal effect size, not equal parameter.
    """
    if delta == 0.0:
        return p.copy()
    if mechanism not in _MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected {set(_MECHANISMS)}")

    transform, span = _MECHANISMS[mechanism]
    target = p.mean() + delta
    if not 0.0 < target < 1.0:
        raise ValueError(f"mean FAR {p.mean():.3f} shifted by {delta} leaves [0, 1]")

    def gap(param: float) -> float:
        return float(transform(p, param).mean() - target)

    lo, hi = (0.0, span) if delta > 0 else (-span, 0.0)
    return transform(p, brentq(gap, lo, hi, xtol=1e-12))


def latent_propensities(
    rng: np.random.Generator,
    n_items: int,
    baseline_far: float,
    concentration: float,
) -> np.ndarray:
    """Per-item latent report propensities, Beta-distributed with mean ``baseline_far``.

    Items are not interchangeable: some traces invite a false alarm and some never do.
    ``concentration`` is the Beta a+b -- low values spread the pool, high values cluster it.
    """
    a = baseline_far * concentration
    b = (1.0 - baseline_far) * concentration
    return rng.beta(a, b, size=n_items)


def observe(rng: np.random.Generator, p: np.ndarray, n_samples: int) -> np.ndarray:
    """Draw per-item binary verdicts, shape (n_items, n_samples); 1 means an error was reported."""
    return (rng.random((len(p), n_samples)) < p[:, None]).astype(np.int8)


def paired_bootstrap_rejects(
    rng: np.random.Generator,
    obs_a: np.ndarray,
    obs_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> bool:
    """Does the item-clustered bootstrap CI on mean(FAR_b - FAR_a) exclude zero?

    The pilot's primary inference: robust to the dependence induced by measuring every
    condition on the same items.
    """
    diff = obs_b.mean(axis=1) - obs_a.mean(axis=1)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return bool(lo > 0 or hi < 0)


def paired_normal_rejects(
    obs_a: np.ndarray,
    obs_b: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray | bool:
    """Closed-form stand-in for :func:`paired_bootstrap_rejects`, ~1000x cheaper.

    Used only for the sample-size sweep; the reported pilot inference stays the bootstrap.
    ``test_power.py`` pins the two to the same verdict on >=95% of studies rather than trusting the
    asymptotics. Accepts a leading batch axis, so a whole sweep cell evaluates in one call.
    """
    diff = obs_b.mean(axis=-1) - obs_a.mean(axis=-1)
    n = diff.shape[-1]
    se = diff.std(axis=-1, ddof=1) / np.sqrt(n)
    z = _norm_quantile(1.0 - alpha / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.abs(diff.mean(axis=-1)) / se
    return np.where(se > 0, stat > z, False)


def _norm_quantile(q: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(q))


def n2_band(
    obs: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> float:
    """The N2 null band: how far a same-condition FAR difference wanders by chance.

    Splits an item's samples into disjoint halves and bootstraps the difference over items,
    returning the (1-alpha) quantile of |FAR(A) - FAR(B)|. A real effect must exceed it.
    """
    if obs.shape[1] < 2:
        raise ValueError(
            "n2_band needs at least 2 samples per item to form disjoint halves; "
            f"got {obs.shape[1]}. This is why greedy decoding cannot be the primary "
            "DV -- with one sample there is no null band to test against."
        )
    half = obs.shape[1] // 2
    far_a = obs[:, :half].mean(axis=1)
    far_b = obs[:, half : 2 * half].mean(axis=1)
    diff = far_a - far_b

    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    # A split-half difference uses n/2 samples per side, so its per-item variance is 2s^2/(n/2)
    # while the cross-condition contrast it guards is 2s^2/n -- exactly twice as wide. Without
    # this rescaling only ~0.3% of complete-null studies cleared their own band instead of ~5%,
    # the same class of unit error as comparing a FAR shift to a verdict flip rate.
    half = obs.shape[1] // 2
    scale = np.sqrt(obs.shape[1] / half)  # = sqrt(2) at n=8
    return float(np.quantile(np.abs(boot), 1 - alpha) / scale)


def power(
    rng: np.random.Generator,
    n_items: int,
    n_samples: int,
    baseline_far: float,
    concentration: float,
    delta: float,
    mechanism: str,
    reps: int = 1000,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> float:
    """Fraction of simulated studies whose paired bootstrap CI excludes zero.

    At ``delta=0`` this is the false-positive rate and should sit near ``alpha``. The two conditions
    share each item's latent propensity, mirroring the real design.
    """
    rejections = 0
    for _ in range(reps):
        p = latent_propensities(rng, n_items, baseline_far, concentration)
        obs_a = observe(rng, p, n_samples)
        obs_b = observe(rng, apply_effect(p, delta, mechanism), n_samples)
        rejections += paired_bootstrap_rejects(rng, obs_a, obs_b, n_boot, alpha)
    return rejections / reps


def solve_effect_param(p_reference: np.ndarray, delta: float, mechanism: str) -> float:
    """The transform parameter that shifts ``p_reference``'s mean FAR by ``delta``."""
    if delta == 0.0:
        return 0.0
    if mechanism not in _MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected {set(_MECHANISMS)}")
    transform, span = _MECHANISMS[mechanism]
    target = p_reference.mean() + delta
    if not 0.0 < target < 1.0:
        raise ValueError(f"mean FAR {p_reference.mean():.3f} shifted by {delta} leaves [0, 1]")
    lo, hi = (0.0, span) if delta > 0 else (-span, 0.0)
    return float(brentq(lambda x: float(transform(p_reference, x).mean() - target), lo, hi))


def power_fast(
    rng: np.random.Generator,
    n_items: int,
    n_samples: int,
    baseline_far: float,
    concentration: float,
    delta: float,
    mechanism: str,
    reps: int = 2000,
    alpha: float = 0.05,
) -> float:
    """Vectorised power via the normal fast path -- all reps in one batched call.

    The effect parameter is solved once on a large reference draw and held fixed across reps: it is
    a property of the population, and each rep is a fresh sample from it.
    """
    a, b = baseline_far * concentration, (1.0 - baseline_far) * concentration
    param = solve_effect_param(rng.beta(a, b, size=200_000), delta, mechanism)
    transform, _ = _MECHANISMS[mechanism]

    p = rng.beta(a, b, size=(reps, n_items))
    obs_a = (rng.random((reps, n_items, n_samples)) < p[..., None]).astype(np.int8)
    shifted = transform(p, param)[..., None]
    obs_b = (rng.random((reps, n_items, n_samples)) < shifted).astype(np.int8)
    return float(np.mean(paired_normal_rejects(obs_a, obs_b, alpha)))


def min_items_for_power(
    rng: np.random.Generator,
    n_samples: int,
    baseline_far: float,
    concentration: float,
    delta: float,
    mechanism: str,
    target: float = 0.80,
    reps: int = 2000,
    alpha: float = 0.05,
    ceiling: int = 20_000,
) -> int | None:
    """Fewest items reaching ``target`` power, by doubling then bisecting.

    Returns None if ``ceiling`` still falls short, so an infeasible cell is visible in the table
    rather than silently clamped to something that merely looks large.
    """
    def at(n: int) -> float:
        return power_fast(rng, n, n_samples, baseline_far, concentration, delta,
                          mechanism, reps, alpha)

    lo, hi = 50, 100
    while at(hi) < target:
        lo, hi = hi, hi * 2
        if hi > ceiling:
            return None
    while lo < hi:
        mid = (lo + hi) // 2
        if at(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _two_sided_p(obs_a: np.ndarray, obs_b: np.ndarray) -> np.ndarray:
    """Paired two-sided p-values for mean(FAR_b - FAR_a), batched over a leading axis."""
    from scipy.stats import norm

    diff = obs_b.mean(axis=-1) - obs_a.mean(axis=-1)
    n = diff.shape[-1]
    se = diff.std(axis=-1, ddof=1) / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(diff.mean(axis=-1)) / se
    return np.where(se > 0, 2.0 * (1.0 - norm.cdf(z)), 1.0)


def holm_rejects_any(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Does Holm-Bonferroni reject at least one hypothesis? Batched over rows.

    Holm rather than Bonferroni: uniformly more powerful at the same family-wise error rate, and
    gate 3 asks only whether *any* condition moved, which is what Holm's first step controls.
    """
    # Holm is step-down: if the smallest p-value fails alpha/k, the procedure stops and rejects
    # nothing, so a later p-value cannot revive the family. The previous
    # `(ordered <= thresholds).any()` allowed exactly that -- p = [0.011, 0.012, 0.9, 0.9, 0.9]
    # returned True when correct Holm rejects nothing.
    pvals = np.atleast_2d(pvals)
    k = pvals.shape[-1]
    return pvals.min(axis=-1) <= alpha / k


def family_false_pass_rate(
    rng: np.random.Generator,
    n_conditions: int,
    n_items: int,
    n_samples: int,
    baseline_far: float,
    concentration: float,
    reps: int = 2000,
    alpha: float = 0.05,
    correction: str = "holm",
) -> float:
    """How often an "any of K conditions moved" gate fires when nothing moved.

    Simulated rather than read off a Bonferroni table because every condition shares the same R0
    baseline, so the K tests are correlated. ``correction="none"`` shows the naive gate's inflation.
    """
    a, b = baseline_far * concentration, (1.0 - baseline_far) * concentration
    p = rng.beta(a, b, size=(reps, n_items))

    def draw() -> np.ndarray:
        return (rng.random((reps, n_items, n_samples)) < p[..., None]).astype(np.int8)

    control = draw()
    pvals = np.stack([_two_sided_p(control, draw()) for _ in range(n_conditions)], axis=-1)

    if correction == "none":
        return float(np.mean((pvals <= alpha).any(axis=-1)))
    if correction == "holm":
        return float(np.mean(holm_rejects_any(pvals, alpha)))
    raise ValueError(f"unknown correction {correction!r}")


def fit_concentration(baseline_far: float, target_n1: float) -> float:
    """Beta concentration whose implied N1 instability matches ``target_n1``.

    Fit before powering anything: pass the measured FAR and N1 for the model in question. A
    hardcoded concentration implies an N1 that can fail this project's own gate 1, which makes every
    sample-size conclusion an artefact of a falsified generative model.
    """
    rng = np.random.default_rng(0)

    def implied(c: float) -> float:
        p = latent_propensities(rng, 40_000, baseline_far, c)
        return float(np.mean(2.0 * np.minimum(p, 1.0 - p)))

    # N1 *increases* with concentration: a tight Beta clusters every item near the mean, which
    # is where per-item verdicts are least deterministic. Measured: c=0.05 -> 0.019,
    # c=0.25 -> 0.080, c=8.0 -> 0.358.
    lo, hi = 1e-4, 500.0
    if implied(lo) >= target_n1:
        return lo
    if implied(hi) <= target_n1:
        return hi
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if implied(mid) > target_n1:
            hi = mid
        else:
            lo = mid
    return (lo * hi) ** 0.5
