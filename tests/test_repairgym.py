"""Tests for repairgym fault injection and its blocking validation."""

from __future__ import annotations

import pytest

from critxer.core.repairgym import (
    Claim,
    arithmetic_claims,
    final_answer,
    first_inconsistent_claim,
    injectable_steps,
    is_eligible_source,
    validate_injection,
)


def claim(operands, ops, result):
    return Claim(tuple(operands), tuple(ops), result)


class TestArithmeticClaims:
    """Extraction has to be conservative: a false claim would reject a valid injection."""

    def test_finds_a_plain_claim(self):
        assert arithmetic_claims("She buys 4 more: 3 + 4 = 7.") == [claim([3, 4], ["+"], 7)]

    @pytest.mark.parametrize(
        "text",
        [r"In total, Randy has \(10 + 7 + 6 = 23\) cookies now.",
         "Total quarts picked per hour = 6 + 5 + 10 + 8 = 29 quarts.",
         "2 + 3 * 4 = 14",
         r"$100 - 20 - 30 = 50$"],
        ids=["three_terms", "four_terms", "precedence", "repeated_minus"],
    )
    def test_evaluates_n_ary_chains(self, text):
        """Chained arithmetic is everywhere in these traces.

        Reading only the last two operands of `10 + 7 + 6 = 23` gives `7 + 6 = 23`, which is
        false. Measured against the 1,179 human-verified-correct clean seeds, that bug flagged
        17.56% of them as inconsistent -- enough to reject valid injections wholesale in the
        acceptance gate.
        """
        claims = arithmetic_claims(text)

        assert claims, f"no claim extracted from {text!r}"
        assert all(c.holds for c in claims), [c for c in claims if not c.holds]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"$3 \times 4 = 12$", claim([3, 4], ["*"], 12)),
            (r"\(6 \cdot 7 = 42\)", claim([6, 7], ["*"], 42)),
            (r"$144 \div 12 = 12$", claim([144, 12], ["/"], 12)),
            ("100 - 38 = 62", claim([100, 38], ["-"], 62)),
            ("0.5 * 4 = 2.0", claim([0.5, 4], ["*"], 2.0)),
            ("1,000 + 1 = 1001", claim([1000, 1], ["+"], 1001)),
        ],
    )
    def test_handles_latex_decimals_and_thousands_separators(self, text, expected):
        assert arithmetic_claims(text) == [expected]

    def test_finds_several_claims_in_one_step(self):
        claims = arithmetic_claims("First 2 + 3 = 5, then 5 * 4 = 20.")

        assert claims == [claim([2, 3], ["+"], 5), claim([5, 4], ["*"], 20)]

    @pytest.mark.parametrize(
        "text",
        ["a + b = c", "x + 1 = 5", "Let b_1 + 2 = 5", "2^3 = 8", "the answer is 7"],
        ids=["symbolic", "one_symbol", "subscript", "exponent", "prose"],
    )
    def test_ignores_anything_not_a_purely_numeric_binary_claim(self, text):
        """Exponents and symbols are out of scope: we only verify what we can evaluate."""
        assert arithmetic_claims(text) == []


class TestConsistency:
    def test_accepts_arithmetically_sound_steps(self):
        assert first_inconsistent_claim(["3 + 4 = 7", "7 * 2 = 14"]) is None

    def test_reports_the_first_unsound_claim(self):
        bad = first_inconsistent_claim(["3 + 4 = 7", "7 * 2 = 15", "15 - 1 = 14"])

        assert bad == (1, claim([7, 2], ["*"], 15))

    def test_tolerates_floating_point_rounding(self):
        """Traces round; 1/3 = 0.333 must not read as an error."""
        assert first_inconsistent_claim(["1 / 3 = 0.333"]) is None

    def test_ignores_division_by_zero_rather_than_crashing(self):
        assert first_inconsistent_claim(["5 / 0 = 0"]) is None


ORIGINAL = ["Start with 10 apples.", "Buy 5 more: 10 + 5 = 15.", "Eat 3: 15 - 3 = 12.",
            "The answer is 12."]


class TestValidateInjection:
    """Acceptance is blocking: anything that fails is dropped, never repaired."""

    def test_accepts_a_consistently_propagated_local_fault(self):
        injected = ["Start with 10 apples.", "Buy 5 more: 10 + 5 = 16.", "Eat 3: 16 - 3 = 13.",
                    "The answer is 13."]

        result = validate_injection(ORIGINAL, injected, step_k=2, original_answer="12")

        assert result.ok, result.reasons

    def test_rejects_a_changed_prefix(self):
        """Steps before the fault must be byte-identical, or the fault is not localised at k."""
        injected = ["Start with 11 apples.", "Buy 5 more: 11 + 5 = 16.", "Eat 3: 16 - 3 = 13.",
                    "The answer is 13."]

        result = validate_injection(ORIGINAL, injected, step_k=2, original_answer="12")

        assert not result.ok
        assert any("prefix" in r for r in result.reasons)

    def test_rejects_an_unchanged_fault_step(self):
        result = validate_injection(ORIGINAL, list(ORIGINAL), step_k=2, original_answer="12")

        assert not result.ok
        assert any("step 2 unchanged" in r for r in result.reasons)

    def test_rejects_unpropagated_downstream_arithmetic(self):
        """The whole point of propagation: no visible inconsistency after the fault step.

        An un-propagated fault leaves a second anomaly downstream, which changes the fault's
        detectability rather than its repair cost, so the two families stop being comparable.
        """
        injected = ["Start with 10 apples.", "Buy 5 more: 10 + 5 = 16.", "Eat 3: 15 - 3 = 12.",
                    "The answer is 12."]

        result = validate_injection(ORIGINAL, injected, step_k=2, original_answer="12")

        assert not result.ok
        assert any("inconsistent" in r for r in result.reasons)

    def test_rejects_a_fault_that_leaves_the_answer_unchanged(self):
        """If the final answer still matches ground truth, nothing detectable was injected."""
        injected = ["Start with 10 apples.", "Buy 5 more: 10 + 5 = 16.", "Eat 3: 16 - 4 = 12.",
                    "The answer is 12."]

        result = validate_injection(ORIGINAL, injected, step_k=2, original_answer="12")

        assert not result.ok
        assert any("answer" in r for r in result.reasons)

    def test_rejects_a_changed_step_count(self):
        injected = ["Start with 10 apples.", "Buy 5 more: 10 + 5 = 16.", "Eat 3: 16 - 3 = 13."]

        result = validate_injection(ORIGINAL, injected, step_k=2, original_answer="12")

        assert not result.ok
        assert any("step count" in r for r in result.reasons)

    def test_rejects_an_out_of_range_fault_step(self):
        with pytest.raises(ValueError, match="step_k"):
            validate_injection(ORIGINAL, list(ORIGINAL), step_k=99, original_answer="12")


class TestInjectableSteps:
    """Position is now the repair-cost manipulation, so the old exclusions are re-derived.

    The first/last exclusion existed to stop fault position confounding fault *family*. With
    family demoted and position promoted, that reason is gone -- but the two ends are not
    symmetric:

    * **Step 1 stays excluded.** There would be no untouched prefix, so nothing anchors the
      claim that the fault is localised at k, and the whole trace becomes a rewrite.
    * **The last step is now allowed.** It has zero downstream steps, which is the cheapest
      possible repair -- a genuinely informative end of the continuous repair-cost range, not
      a degenerate case.
    """

    def test_includes_the_last_step(self):
        steps = ["Start.", "10 + 5 = 15.", "15 - 3 = 12.", "12 + 1 = 13."]

        assert 4 in injectable_steps(steps)

    def test_still_excludes_the_first_step(self):
        steps = ["3 + 4 = 7.", "7 + 1 = 8.", "8 + 1 = 9.", "9 + 1 = 10."]

        assert 1 not in injectable_steps(steps)

    def test_only_returns_steps_carrying_a_numeric_claim(self):
        steps = ["Start.", "Reason abstractly.", "10 + 5 = 15.", "Conclude."]

        assert injectable_steps(steps) == [3]


class TestSourceEligibility:
    def test_accepts_a_trace_the_extractor_reads_as_consistent(self):
        assert is_eligible_source(ORIGINAL, min_steps=4)

    def test_rejects_a_trace_the_extractor_cannot_verify_cleanly(self):
        """The extractor mis-reads multi-`=` chains, so such traces are excluded as sources.

        Measured: 5.43% of the 1,179 human-verified-correct clean seeds trip the extractor,
        mostly `4 * 9 * 5 = 36 * 5 = 180` chains and symbolic equations. Excluding them makes
        any inconsistency in an *injected* trace attributable to the injection by construction,
        which is stronger than tuning the extractor until it stops complaining.
        """
        trace = ["Setup.", "LCM = 4 * 9 * 5 = 36 * 5 = 180.", "Check.", "The answer is 180."]

        assert not is_eligible_source(trace, min_steps=4)

    def test_rejects_a_trace_with_too_few_steps(self):
        assert not is_eligible_source(["a", "b", "The answer is 1."], min_steps=4)

    def test_rejects_a_trace_whose_only_claim_is_in_the_first_step(self):
        """Step 1 has no untouched prefix to anchor localisation, so it is not injectable."""
        assert not is_eligible_source(
            ["3 + 4 = 7.", "Reason abstractly.", "Conclude.", "Done."], min_steps=4
        )


class TestControlCharacterCorruption:
    r"""LaTeX inside a JSON string gets silently mangled, and it parses as valid JSON.

    A model writing `\times` in a JSON string emits a legal `\t` escape, so `json.loads`
    yields TAB + "imes". Guided decoding cannot prevent it -- the output *is* valid JSON.
    Measured on the injection pilot before this guard existed. It must never pass silently,
    because it marks injected items with corruption clean items never have.
    """

    def test_rejects_a_tab_introduced_by_a_mangled_latex_command(self):
        injected = [*ORIGINAL[:1], "Buy 5 more: $10 \times 5 = 16$.".replace("\\t", "\t"),
                    "Eat 3: 16 - 3 = 13.", "The answer is 13."]

        result = validate_injection(ORIGINAL, injected, step_k=2, original_answer="12")

        assert not result.ok
        assert any("control character" in r for r in result.reasons)

    def test_allows_a_plain_newline_which_is_legitimate_multi_line_text(self):
        """Newlines are not the corruption signature and must not be rejected.

        With plain-text generation there is no escape layer, so an embedded newline is just a
        multi-line step. Treating it as corruption rejected 45% of otherwise-valid candidates.
        """
        injected = [*ORIGINAL[:1], "Buy 5 more:\n10 + 5 = 16.", "Eat 3: 16 - 3 = 13.",
                    "The answer is 13."]

        result = validate_injection(ORIGINAL, injected, step_k=2, original_answer="12")

        assert result.ok, result.reasons

    def test_allows_a_control_character_the_original_already_had(self):
        original = ["Start.", "A\ttabbed step: 10 + 5 = 15.", "Eat 3: 15 - 3 = 12.", "Answer 12."]
        injected = ["Start.", "A\ttabbed step: 10 + 5 = 16.", "Eat 3: 16 - 3 = 13.", "Answer 13."]

        result = validate_injection(original, injected, step_k=2, original_answer="12")

        assert result.ok, result.reasons


class TestAnswerComparison:
    """`_normalise_answer` stripped all non-digits from the whole last step.

    "So the total is 15 - 3 = 12, the answer is 12." became "15-31212", so an injected trace
    whose final answer was still ground truth validated clean. Verified against the live
    validator before this fix. The check only ever fired when the last step was answer-only,
    which is exactly the shape the original test used.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("So the total is 15 - 3 = 12, the answer is 12.", "12"),
            (r"Therefore the answer is \boxed{362}.", "362"),
            ("The answer is 7", "7"),
            ("Hence x = 1,024.", "1024"),
            ("Thus the total comes to 3.5 dollars.", "3.5"),
        ],
        ids=["trailing_arithmetic", "boxed", "bare", "thousands", "decimal"],
    )
    def test_extracts_the_final_answer_not_every_digit(self, text, expected):
        assert final_answer(text) == expected

    def test_rejects_a_fault_whose_answer_is_still_ground_truth(self):
        original = ["Start.", "10 + 5 = 15.", "Then 15 - 3 = 12.",
                    "So the total is 15 - 3 = 12, the answer is 12."]
        injected = ["Start.", "10 + 5 = 16.", "Then 16 - 4 = 12.",
                    "So the total is 16 - 4 = 12, the answer is 12."]

        result = validate_injection(original, injected, 2, original[-1])

        assert not result.ok
        assert any("answer" in r for r in result.reasons)


class TestPropagationIsCheckedRegardlessOfRewriting:
    """The guard did `if injected[j] != original[j]: continue`.

    The generator rewrites every step from k onward, so the guard almost never applied --
    prepending "Now, " was enough to smuggle a stale pre-fault value through, and only 9
    not-propagated rejections appeared across a 363-source run. Consistent propagation is the
    load-bearing claim of the injection design, so it has to be checked on rewritten steps too.
    """

    def test_catches_a_stale_value_in_a_rewritten_step(self):
        original = ["Start.", "10 + 5 = 15.", "15 - 3 = 12.", "Answer 12."]
        injected = ["Start.", "10 + 5 = 16.", "Now, 15 - 3 = 12.", "Answer 99."]

        result = validate_injection(original, injected, 2, "Answer 12.")

        assert not result.ok
        assert any("not propagated" in r for r in result.reasons)

    def test_accepts_a_properly_propagated_rewrite(self):
        original = ["Start.", "10 + 5 = 15.", "15 - 3 = 12.", "Answer 12."]
        injected = ["Start.", "10 + 5 = 16.", "Now, 16 - 3 = 13.", "Answer 13."]

        assert validate_injection(original, injected, 2, "Answer 12.").ok


class TestStepKMustNotBeGutted:
    """The manual audit found step k shrunk by >30% in 10.2% of injected traces.

    Two distinct harms, and the arithmetic checks see neither:

    * A systematically shorter step at k is itself a cue, in the same class as the emphasis and
      step-prefix cues `critxer.core.inject.normalise_step` already corrects.
    * Deletion can change the problem rather than inject a fault. In one audited item the
      injector dropped an entire case from a maximisation, so the answer changed because the
      candidate set shrank -- not because any step was wrong. That item passes every arithmetic
      check and is not a single-step-attributable fault.
    """

    def _trace(self, step_k_text: str) -> list[str]:
        return ["Start with 12 apples.", step_k_text, "So the answer is \\boxed{7}."]

    def test_a_gutted_step_k_is_rejected(self):
        original = self._trace(
            "Next, we subtract the five apples that were eaten: 12 - 5 = 7 apples remain."
        )
        injected = ["Start with 12 apples.", "12 - 5 = 6", "So the answer is \\boxed{6}."]

        v = validate_injection(original, injected, 2, "So the answer is \\boxed{7}.")

        assert not v.ok
        assert any("shorter" in r for r in v.reasons), v.reasons

    def test_a_value_substitution_of_similar_length_is_accepted(self):
        """The check must not fire on the normal case: same prose, one number changed."""
        original = self._trace(
            "Next, we subtract the five apples that were eaten: 12 - 5 = 7 apples remain."
        )
        injected = [
            "Start with 12 apples.",
            "Next, we subtract the five apples that were eaten: 12 - 5 = 6 apples remain.",
            "So the answer is \\boxed{6}.",
        ]

        v = validate_injection(original, injected, 2, "So the answer is \\boxed{7}.")

        assert v.ok, v.reasons


class TestRejectionReasonsAreClassifiedByTheirOwnCheck:
    """Each validator check must map to its own class, not to a substring of another's.

    The classifier is a first-match-wins scan over keys, so a general key placed before a specific
    one swallows it. Two did: "prefix" caught "echoed step prefix", and "unchanged" caught the
    not-propagated reason, which also contains the word. Both merged distinct checks into one bucket
    in the rejection breakdown -- silently, because the count still added up to the total.
    """

    # Verbatim from `validate_injection`, formatted with plausible values.
    REASONS = (
        ("prefix before step 3 was modified", "prefix"),
        ("step 4 unchanged, so no fault was injected", "unchanged"),
        ("step 4 is 45% shorter than the original (10 vs 22 chars)", "shorter"),
        ("step 6 is unchanged but still uses pre-fault value 12: inconsistent with the injected "
         "step, i.e. the fault was not propagated", "not propagated"),
        ("step 3 introduced control character(s) ['\\t']: ", "control character"),
        ("step 5 gained an echoed step prefix absent from the original: a formatting cue",
         "echoed step prefix"),
        ("final answer still matches ground truth", "final answer"),
        ("step count changed from 6 to 7", "step count"),
        # Two different checks both end "fault was not propagated", so this one is only
        # distinguishable by the more specific key being scanned first.
        ("inconsistent arithmetic at step 5: 2 + 2 = 5 -- fault was not propagated",
         "inconsistent arithmetic"),
        ("something no check emits", "other"),
    )

    @pytest.mark.parametrize(("reason", "expected"), REASONS)
    def test_each_reason_maps_to_its_own_class(self, reason, expected):
        from critxer.core.repairgym import reason_class

        assert reason_class(reason) == expected

    def test_every_declared_class_is_reachable(self):
        """A key shadowed by an earlier one can never be returned, so the class is dead."""
        from critxer.core.repairgym import REJECTION_CLASSES, reason_class

        reached = {reason_class(r) for r, _ in self.REASONS}
        unreachable = set(REJECTION_CLASSES) - reached
        assert not unreachable, f"no reason maps to {sorted(unreachable)}"
