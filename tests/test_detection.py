"""Tests for detection-side metrics on the labelled-incorrect arm.

A false-alarm rate alone cannot tell a better-calibrated auditor from a uniformly more lenient one,
since both lower FAR; these metrics add the other half.

The test that matters most is the 1-based/0-based one. ProcessBench's ``label`` is the 0-based first
bad step, the audit schema reports ``first_error_step`` 1-based, and an off-by-one would not crash:
it would silently report localisation near zero and we would probably believe it.
"""

from __future__ import annotations

import numpy as np
import pytest

from critxer.core.detection import (
    balanced_accuracy,
    criterion,
    expected_calibration_error,
    localisation_accuracy,
    sensitivity_index,
)


class TestLocalisationUsesOneBasedSteps:
    """ProcessBench label 0 means "step 1 is wrong", which the model reports as 1."""

    def test_a_correct_localisation_scores_one(self):
        got = localisation_accuracy(reported_steps=[1], gold_labels=[0])

        assert got == 1.0

    def test_treating_the_label_as_one_based_would_score_zero(self):
        """The failure mode this test exists to catch, pinned explicitly.

        With gold label 3 (0-based) the right answer is step 4. If a caller compared against 3 it
        would score this attempt wrong, and the metric would look uniformly terrible rather than
        broken.
        """
        assert localisation_accuracy(reported_steps=[4], gold_labels=[3]) == 1.0
        assert localisation_accuracy(reported_steps=[3], gold_labels=[3]) == 0.0

    def test_items_with_no_reported_step_are_excluded_not_counted_wrong(self):
        """A model that said "correct" has nothing to localise; that is a miss, not a mislocation.

        Scoring it as a localisation failure would conflate two different errors and make the metric
        move whenever the detection rate moved.
        """
        # step 1 vs label 0 hits; step 3 vs label 1 (gold step 2) misses; the None is skipped.
        got = localisation_accuracy(reported_steps=[1, None, 3], gold_labels=[0, 1, 1])

        assert got == pytest.approx(0.5)

    def test_all_missing_gives_nan_rather_than_zero(self):
        assert np.isnan(localisation_accuracy(reported_steps=[None, None], gold_labels=[0, 1]))


class TestSignalDetectionSeparatesThresholdFromDiscrimination:
    """The reason FAR alone is not enough: d' moves with discrimination, c with the threshold."""

    def test_a_pure_threshold_shift_leaves_d_prime_unchanged(self):
        """Both rates moving together is leniency, and that is what d' must be blind to."""
        strict = sensitivity_index(detection=0.84, far=0.16, n_signal=500, n_noise=500)
        lenient = sensitivity_index(detection=0.69, far=0.07, n_signal=500, n_noise=500)

        assert strict == pytest.approx(lenient, abs=0.05)

    def test_the_criterion_moves_with_that_shift(self):
        strict = criterion(detection=0.84, far=0.16, n_signal=500, n_noise=500)
        lenient = criterion(detection=0.69, far=0.07, n_signal=500, n_noise=500)

        assert lenient > strict

    def test_better_discrimination_raises_d_prime(self):
        worse = sensitivity_index(detection=0.60, far=0.40, n_signal=500, n_noise=500)
        better = sensitivity_index(detection=0.90, far=0.10, n_signal=500, n_noise=500)

        assert better > worse

    def test_perfect_rates_are_corrected_rather_than_infinite(self):
        """Without a correction a ceiling rate gives d' = inf and the condition drops out."""
        got = sensitivity_index(detection=1.0, far=0.0, n_signal=200, n_noise=200)

        assert np.isfinite(got)
        assert got > 4.0

    def test_chance_performance_is_zero(self):
        assert sensitivity_index(detection=0.5, far=0.5, n_signal=100, n_noise=100) == (
            pytest.approx(0.0)
        )


class TestBalancedAccuracy:
    def test_it_averages_the_two_rates(self):
        assert balanced_accuracy(detection=0.8, far=0.2) == pytest.approx(0.8)

    def test_flagging_everything_scores_chance(self):
        """A model that always says "incorrect" has FAR 1 and detection 1, and knows nothing."""
        assert balanced_accuracy(detection=1.0, far=1.0) == pytest.approx(0.5)


class TestExpectedCalibrationError:
    def test_a_perfectly_calibrated_set_scores_zero(self):
        # 0.9 confidence on 10 items of which 9 are right, 0.1 on 10 of which 1 is right.
        conf = np.array([0.9] * 10 + [0.1] * 10)
        correct = np.array([1] * 9 + [0] + [1] + [0] * 9)

        assert expected_calibration_error(conf, correct, n_bins=2) == pytest.approx(0.0, abs=1e-9)

    def test_confident_and_wrong_scores_high(self):
        conf = np.full(10, 0.95)
        correct = np.zeros(10)

        assert expected_calibration_error(conf, correct, n_bins=5) == pytest.approx(0.95)

    def test_empty_bins_do_not_contribute(self):
        """Otherwise bin count would change the score on identical data."""
        conf = np.array([0.5, 0.5])
        correct = np.array([1, 0])

        few = expected_calibration_error(conf, correct, n_bins=2)
        many = expected_calibration_error(conf, correct, n_bins=20)
        assert few == pytest.approx(many)


# --- the reasoning-enabled detection arm --------------------------------------------------------
#
# The clean arm can be run with thinking on (`run-detection` had no such flag, so the criterion-vs-
# discrimination question could not be asked in the setting deployed judges actually use). Budget
# and thinking are stamped on every record because a thinking-enabled cell joined to the main arm
# is a two-way confound, and nothing downstream can detect it after the fact.

class TestReasoningArmFlags:
    def test_defaults_match_the_main_study(self):
        """Absent the flags, a run must be byte-for-byte the main arm's configuration."""
        from critxer.cli.run.detection import MAX_TOKENS, build_parser

        args = build_parser().parse_args(["--endpoint", "m=M@http://127.0.0.1:9000"])

        assert args.thinking is False
        assert args.max_tokens == MAX_TOKENS

    def test_thinking_needs_an_explicit_budget_to_be_useful(self):
        from critxer.cli.run.detection import build_parser

        args = build_parser().parse_args(
            ["--endpoint", "m=M@http://127.0.0.1:9000", "--thinking", "--max-tokens", "8192"])

        assert args.thinking is True
        assert args.max_tokens == 8192
