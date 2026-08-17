"""Tests for observed-data metrics."""

from __future__ import annotations

import numpy as np
import pytest

from critxer.core.metrics import far, instability_index


def test_deterministic_items_have_zero_instability():
    """N1 is the gate-1 diagnostic: 0 means every item's verdict is stable."""
    assert instability_index(np.array([0.0, 1.0, 0.0, 1.0])) == pytest.approx(0.0)


def test_coin_flip_items_have_maximal_instability():
    """1 means every item is a coin flip, so per-item verdicts carry no signal."""
    assert instability_index(np.array([0.5, 0.5])) == pytest.approx(1.0)


def test_instability_is_symmetric_about_one_half():
    """An item reported 1/8 of the time is exactly as unstable as one reported 7/8."""
    assert instability_index(np.array([0.125])) == pytest.approx(
        instability_index(np.array([0.875]))
    )


def test_far_is_the_mean_report_probability():
    assert far(np.array([0.0, 0.25, 0.5])) == pytest.approx(0.25)


def test_metrics_ignore_items_with_no_usable_samples():
    """An item whose every sample failed to parse is NaN and must not poison the mean."""
    assert far(np.array([0.2, np.nan, 0.4])) == pytest.approx(0.3)
    assert instability_index(np.array([0.5, np.nan])) == pytest.approx(1.0)
