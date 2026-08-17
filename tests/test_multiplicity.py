"""Tests for Holm-Bonferroni step-down over a declared family.

This lived as an untested helper inside ``critxer analyse-ladder`` while the detection analysis
reported uncorrected p-values, which is how the paper came to claim a d' improvement at p = 0.028
inside a family of three. Two properties matter and neither is obvious from the formula: the
threshold depends on a hypothesis's *rank*, and the procedure *stops* at the first failure rather
than continuing to test later hypotheses on their own merits.
"""

from __future__ import annotations

import pytest

from critxer.core.multiplicity import holm, short_family_message


def test_the_smallest_p_is_tested_against_alpha_over_k():
    decided = holm({"a": 0.01, "b": 0.02, "c": 0.03}, alpha=0.05)

    assert decided["a"]["threshold"] == pytest.approx(0.05 / 3)


def test_thresholds_relax_as_the_rank_increases():
    decided = holm({"a": 0.001, "b": 0.002, "c": 0.003}, alpha=0.06)

    assert [decided[k]["threshold"] for k in ("a", "b", "c")] == pytest.approx([0.02, 0.03, 0.06])


def test_a_family_of_one_is_uncorrected():
    """Declaring a family of one is not a way to smuggle a lone test past correction."""
    decided = holm({"a": 0.04}, alpha=0.05)

    assert decided["a"]["threshold"] == pytest.approx(0.05)
    assert decided["a"]["reject"]


def test_the_real_case_the_paper_got_wrong():
    """AS delta-d' across three models: the largest p sinks the family's third hypothesis.

    Measured values. Uncorrected, the 27B's 0.028 reads as significant; inside its declared family
    of three it is tested against 0.05/2 and fails.
    """
    decided = holm({"27B": 0.028, "35B": 0.83, "ministral": 0.92}, alpha=0.05)

    assert decided["27B"]["threshold"] == pytest.approx(0.05 / 3)
    assert not decided["27B"]["reject"]


def test_a_step_down_stops_at_the_first_failure():
    """The whole of "step-down": a later hypothesis cannot reject on its own merits.

    Ranks are 0.001, 0.03, 0.04 against thresholds 0.0167, 0.025, 0.05. The first rejects. The
    second fails. The third would clear 0.05 comfortably if tested independently -- and must not,
    because the procedure already stopped. Testing each rank on its own threshold would be
    Holm-shaped and wrong, and is the bug this test exists to catch.
    """
    decided = holm({"a": 0.001, "b": 0.03, "c": 0.04}, alpha=0.05)

    assert decided["a"]["reject"]
    assert not decided["b"]["reject"]
    assert decided["c"]["threshold"] == pytest.approx(0.05)
    assert not decided["c"]["reject"]


def test_ties_do_not_let_both_members_through_on_the_looser_threshold():
    """Two identical p-values occupy two ranks; the stricter one governs both."""
    decided = holm({"a": 0.03, "b": 0.03}, alpha=0.05)

    assert not decided["a"]["reject"]
    assert not decided["b"]["reject"]


def test_tuple_keys_survive_so_a_family_can_be_keyed_by_model_and_contrast():
    decided = holm({("27B", "AS"): 0.001, ("35B", "AS"): 0.002}, alpha=0.05)

    assert decided[("27B", "AS")]["reject"]


def test_an_empty_family_decides_nothing_rather_than_dividing_by_zero():
    assert holm({}, alpha=0.05) == {}


class TestAnUnderPopulatedFamilyHasToSaySo:
    """A Holm threshold is alpha/k, so a family missing members tests survivors too leniently.

    This happened twice. The detection analysis swept only AS across F2-F5, so `prior-context` had
    nine members at F1 and three elsewhere and the same cell was tested at 0.0167 at one wording and
    0.0056 at another. The R4 analysis had the same hole and no guard at all: AN, AX and AXN existed
    only at F1, so its claim family was eighteen members at the primary wording and twelve at the
    other four -- and twelve members is a threshold of alpha/12, which is *easier* to clear. The
    declaration is the thing under review, so a family built from whatever is on disk must announce
    the gap instead of correcting against it quietly.
    """

    def test_a_fully_populated_family_is_silent(self):
        assert short_family_message("r4-claims", got=18, expected=18, present=["AS"]) is None

    def test_a_short_family_names_both_thresholds_and_what_is_present(self):
        msg = short_family_message("r4-claims", got=12, expected=18,
                                   present=["episode_vs_filler", "audit_only_vs_filler"])

        assert msg is not None
        assert "r4-claims" in msg
        assert "12" in msg and "18" in msg
        assert "episode_vs_filler" in msg
        assert "MORE lenient" in msg

    def test_a_family_larger_than_declared_is_also_reported(self):
        """Membership drift in the other direction is a declaration error too, not a safe one.

        Over-population cannot inflate a survivor, but it means the family being corrected is not
        the family that was declared -- so it is the declaration or the code that is wrong, and
        either way the run should not pass silently.
        """
        msg = short_family_message("r4-claims", got=21, expected=18, present=["interaction"])

        assert msg is not None
        assert "21" in msg and "18" in msg

    def test_an_empty_family_is_reported_rather_than_treated_as_complete(self):
        """got=0 is falsy, and a truthiness-based guard would have skipped this case."""
        assert short_family_message("r4-claims", got=0, expected=18, present=[]) is not None
