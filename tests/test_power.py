"""Tests for the pilot power / null-calibration simulation."""

from __future__ import annotations

import numpy as np
import pytest

from critxer.core.metrics import instability_index
from critxer.core.power import (
    apply_effect,
    family_false_pass_rate,
    fit_concentration,
    holm_rejects_any,
    latent_propensities,
    min_items_for_power,
    n2_band,
    observe,
    paired_bootstrap_rejects,
    paired_normal_rejects,
    power,
)


def test_n2_band_rejects_a_single_sample_per_item():
    """One sample per item cannot be split in half, so there is no null band.

    This is the structural reason T=0 cannot be the primary DV: gate 3
    would have nothing to test a cross-condition difference against.
    """
    rng = np.random.default_rng(0)
    obs = observe(rng, p=np.full(50, 0.2), n_samples=1)

    with pytest.raises(ValueError, match="at least 2 samples"):
        n2_band(obs, rng=rng)


def test_paired_test_is_calibrated_under_the_null():
    """With no real effect the test must fire at about alpha, not more.

    An uncalibrated decision procedure would kill or spare the project for the wrong
    reason. Tolerance is loose because this is itself a Monte Carlo estimate: 300 reps
    at alpha=0.05 has a standard error of ~1.3pp.
    """
    rng = np.random.default_rng(1)
    reps = 300
    rejections = 0
    for _ in range(reps):
        p = latent_propensities(rng, n_items=300, baseline_far=0.15, concentration=8.0)
        obs_a = observe(rng, p, n_samples=8)
        obs_b = observe(rng, p, n_samples=8)  # same latent p => true effect is zero
        rejections += paired_bootstrap_rejects(rng, obs_a, obs_b, n_boot=500)

    false_positive_rate = rejections / reps
    assert 0.02 <= false_positive_rate <= 0.09, f"FPR={false_positive_rate:.3f}"


@pytest.mark.parametrize("mechanism", ["additive", "logit"])
@pytest.mark.parametrize("baseline_far", [0.05, 0.15, 0.30])
def test_effect_achieves_the_requested_mean_shift(mechanism, baseline_far):
    """Both mechanisms must shift the *mean* FAR by exactly delta.

    Otherwise every power number is quietly reported against the wrong effect size.
    The logit mechanism needs a solve to hit the target, so this is where a bug hides.
    """
    rng = np.random.default_rng(2)
    p = latent_propensities(rng, n_items=4000, baseline_far=baseline_far, concentration=8.0)

    shifted = apply_effect(p, delta=0.03, mechanism=mechanism)

    assert shifted.mean() - p.mean() == pytest.approx(0.03, abs=1e-4)


def test_logit_effect_concentrates_on_mid_range_items():
    """A logit shift moves mid-range items most and near-certain items least.

    An additive shift moves them all equally. This is why the two mechanisms give
    different power and why the conservative one must drive the sample-size decision.
    Values chosen to avoid the 0/1 boundary so no clipping confounds the comparison.
    """
    p = np.array([0.001, 0.5, 0.9])

    logit_shift = apply_effect(p, delta=0.03, mechanism="logit") - p
    additive_shift = apply_effect(p, delta=0.03, mechanism="additive") - p

    assert logit_shift[1] > logit_shift[2] > logit_shift[0]
    assert additive_shift == pytest.approx(np.full(3, 0.03), abs=1e-9)


def test_additive_effect_redistributes_shift_away_from_saturated_items():
    """When an item cannot absorb its share of an additive shift, the rest take it up.

    An item at 0.999 has only 0.001 of headroom, so holding the *mean* shift at delta
    forces a larger move onto the unsaturated items. Pinned because it means "additive"
    is not literally uniform near the boundary, and a reader of the power table should
    know the shift is defined on the mean rather than per item.
    """
    p = np.array([0.001, 0.5, 0.999])

    shift = apply_effect(p, delta=0.03, mechanism="additive") - p

    assert shift.mean() == pytest.approx(0.03, abs=1e-9)
    assert shift[2] == pytest.approx(0.001, abs=1e-9)
    assert shift[0] > 0.03 and shift[1] > 0.03


def test_power_increases_with_more_items():
    """More items must buy more power, all else equal. Sanity check on the whole chain."""
    rng = np.random.default_rng(4)
    kwargs = dict(
        n_samples=8, baseline_far=0.15, concentration=8.0, delta=0.03,
        mechanism="logit", reps=200, n_boot=400,
    )

    small = power(np.random.default_rng(5), n_items=150, **kwargs)
    large = power(rng, n_items=1200, **kwargs)

    assert large > small


def test_power_collapses_to_alpha_when_there_is_no_effect():
    """At delta=0 the power function must return roughly alpha, not more.

    Integration-level guard on the same property test 2 checks for the bare test: if
    this drifts high, the gate would kill or spare the project on noise.
    """
    rate = power(
        np.random.default_rng(6), n_items=400, n_samples=8, baseline_far=0.15,
        concentration=8.0, delta=0.0, mechanism="logit", reps=300, n_boot=400,
    )

    assert rate <= 0.09, f"false-positive rate {rate:.3f}"


def test_normal_approximation_agrees_with_the_bootstrap():
    """The fast path must reach the same verdict as the bootstrap on the same data.

    The full sweep needs ~180 power estimates; at 1000 bootstrap resamples each that is
    tens of minutes. The bootstrap CI on a mean is asymptotically the normal CI with
    SE = sd(d)/sqrt(N), so the sweep uses the closed form -- but only because this test
    pins the two to the same decisions rather than assuming the asymptotics hold at our n.
    """
    rng = np.random.default_rng(7)
    agree = 0
    trials = 200
    for _ in range(trials):
        p = latent_propensities(rng, n_items=400, baseline_far=0.15, concentration=8.0)
        obs_a = observe(rng, p, n_samples=8)
        # Sweep across the null and a detectable effect so agreement is tested on both
        # sides of the decision boundary, not just where both trivially say "no".
        delta = 0.0 if rng.random() < 0.5 else 0.04
        obs_b = observe(rng, apply_effect(p, delta, "logit"), n_samples=8)

        boot = paired_bootstrap_rejects(rng, obs_a, obs_b, n_boot=1000)
        fast = paired_normal_rejects(obs_a, obs_b)
        agree += boot == fast

    assert agree / trials >= 0.95, f"agreement {agree / trials:.3f}"


def test_smaller_effects_need_more_items():
    """The sample-size search must be monotone in effect size.

    This is the property the whole compute budget hangs on: if 3pp did not demand more
    items than 5pp, the search is broken and the budget derived from it is meaningless.
    """
    common = dict(
        n_samples=8, baseline_far=0.15, concentration=8.0, mechanism="logit",
        target=0.80, reps=400,
    )

    n_for_5pp = min_items_for_power(np.random.default_rng(8), delta=0.05, **common)
    n_for_3pp = min_items_for_power(np.random.default_rng(8), delta=0.03, **common)

    assert n_for_3pp > n_for_5pp


def test_naive_any_of_k_gate_inflates_the_false_pass_rate():
    """Gate 3 compares 5 conditions to R0 and passes if ANY clears. That is 5 tests.

    Without multiplicity control the family-wise false-pass rate is far above 5%, so a
    project with no real effect anywhere would be spared roughly a quarter of the time.
    This is the same class of bug as rl-project's condition-4 false pass.
    """
    rate = family_false_pass_rate(
        np.random.default_rng(9), n_conditions=5, n_items=600, n_samples=8,
        baseline_far=0.15, concentration=8.0, reps=2000, correction="none",
    )

    assert rate > 0.15, f"expected inflation, got {rate:.3f}"


def test_holm_correction_restores_the_family_wise_rate():
    """With Holm across the 5 comparisons the gate fires at about alpha again."""
    rate = family_false_pass_rate(
        np.random.default_rng(10), n_conditions=5, n_items=600, n_samples=8,
        baseline_far=0.15, concentration=8.0, reps=2000, correction="holm",
    )

    assert rate <= 0.075, f"family-wise rate {rate:.3f}"


def test_n2_band_is_calibrated_against_the_contrast_it_guards():
    """The band must match the variance of an 8-vs-8 contrast, not a 4-vs-4 split-half.

    A split-half difference has variance 2s^2/4 while the cross-condition contrast it guards has
    2s^2/8 -- twice as wide. Measured before the fix: only ~0.3% of complete-null studies cleared
    their own band instead of ~5%, so it was not the 95th-percentile band it was advertised as.
    That is the same class of unit error as comparing a FAR shift to a verdict flip rate.
    """
    rng = np.random.default_rng(77)
    cleared = 0
    trials = 400
    for _ in range(trials):
        p = latent_propensities(rng, n_items=400, baseline_far=0.18, concentration=0.25)
        band = n2_band(observe(rng, p, n_samples=8), rng=rng, n_boot=600)
        a, b = observe(rng, p, n_samples=8), observe(rng, p, n_samples=8)
        cleared += abs(a.mean(axis=1).mean() - b.mean(axis=1).mean()) > band
    rate = cleared / trials

    assert 0.02 <= rate <= 0.10, f"band clears {rate:.3f} of null studies, want ~0.05"


def test_holm_requires_the_smallest_p_to_clear_alpha_over_k():
    """`(ordered <= thresholds).any()` skipped Holm's step-down stop.

    Holm rejects nothing unless the smallest p-value clears alpha/k; a later p-value cannot
    revive the family. Verified failing case: p = [0.011, 0.012, 0.9, 0.9, 0.9] at alpha=0.05
    returned True, when correct step-down Holm rejects nothing (0.011 > 0.05/5 = 0.01).
    """
    pvals = np.array([[0.011, 0.012, 0.9, 0.9, 0.9]])

    assert not holm_rejects_any(pvals, alpha=0.05)[0]
    assert holm_rejects_any(np.array([[0.009, 0.9, 0.9, 0.9, 0.9]]), alpha=0.05)[0]


def test_fitted_concentration_reproduces_the_measured_instability():
    """Guards against powering the study on a generative model the data rules out."""
    c = fit_concentration(baseline_far=0.195, target_n1=0.086)
    p = latent_propensities(np.random.default_rng(1), 40_000, 0.195, c)

    assert instability_index(p) == pytest.approx(0.086, abs=0.01)
    assert c < 1.0, f"measured dispersion implies c<1, got {c:.3f} (8.0 was assumed)"
