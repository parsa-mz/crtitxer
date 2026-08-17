"""Tests for audit prompt assembly and response parsing."""

from __future__ import annotations

import json

import pytest

from critxer.core.audit import (
    AUDIT_SCHEMA,
    CONDITIONS,
    CONTROLS,
    LADDER,
    WORDINGS,
    AuditItem,
    audit_records,
    build_audit_messages,
    parse_audit,
)

ITEM = AuditItem(
    item_id="gsm8k-200",
    problem="Jane has 3 apples and buys 4 more. How many does she have?",
    steps=["Jane starts with 3 apples.", "She buys 4 more: 3 + 4 = 7.", "The answer is 7."],
)
VARIANTS = sorted(set(CONDITIONS) - {"R0"})


def _rendered(condition: str, family: str) -> str:
    return "\n".join(m["content"] for m in build_audit_messages(ITEM, condition, family))


@pytest.mark.parametrize("family", WORDINGS)
@pytest.mark.parametrize("condition", VARIANTS)
def test_conditions_differ_from_baseline_only_by_the_future_arrangement(condition, family):
    """The whole design rests on this: nothing but the future-arrangement sentence varies.

    If a condition also changes the task description, output schema, whitespace, or step
    text, the experiment is confounded with the output-format effect Jin & Chen 2026 already
    demonstrated, and no result from it means anything. Compared as exact strings rather
    than token lists so whitespace drift cannot slip through.
    """
    baseline = _rendered("R0", family)
    variant = _rendered(condition, family)
    slot = CONDITIONS[condition]

    assert slot, f"{condition} must define a non-empty future-arrangement sentence"
    assert variant.replace(f"{slot} ", "", 1) == baseline


@pytest.mark.parametrize("family", WORDINGS)
def test_every_family_places_the_slot_before_the_audit_instruction(family):
    """A family that dropped the slot would silently turn that condition into R0."""
    rendered = _rendered("R2", family)

    assert CONDITIONS["R2"] in rendered


def test_steps_are_numbered_from_one():
    """first_error_step is 1-based, so the prompt must present steps that way."""
    rendered = _rendered("R0", "F1")

    assert "Step 1: Jane starts with 3 apples." in rendered
    assert "Step 3: The answer is 7." in rendered


def test_unknown_condition_and_family_are_rejected():
    with pytest.raises(ValueError, match="unknown condition"):
        build_audit_messages(ITEM, "R99", "F1")
    with pytest.raises(ValueError, match="unknown template family"):
        build_audit_messages(ITEM, "R0", "F99")


def test_parses_a_well_formed_incorrect_verdict():
    raw = json.dumps({
        "verdict": "incorrect", "first_error_step": 2, "confidence": 0.8,
        "error_type": "arithmetic", "evidence": "3 + 4 is 7, not 8.",
    })

    audit = parse_audit(raw, n_steps=3)

    assert audit is not None
    assert audit.reported_error
    assert audit.first_error_step == 2


def test_correct_verdict_forces_step_and_type_to_null():
    """Models often fill these in anyway; normalising here keeps FAR unambiguous."""
    raw = json.dumps({
        "verdict": "correct", "first_error_step": 2, "confidence": 0.9,
        "error_type": "arithmetic", "evidence": "Looks fine.",
    })

    audit = parse_audit(raw, n_steps=3)

    assert audit is not None
    assert not audit.reported_error
    assert audit.first_error_step is None
    assert audit.error_type is None


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        "[]",
        json.dumps({"verdict": "maybe", "first_error_step": 1, "confidence": 0.5,
                    "error_type": "arithmetic", "evidence": "x"}),
        json.dumps({"verdict": "incorrect", "first_error_step": 1, "confidence": 1.5,
                    "error_type": "arithmetic", "evidence": "x"}),
        json.dumps({"verdict": "incorrect", "first_error_step": 1, "confidence": None,
                    "error_type": "arithmetic", "evidence": "x"}),
        json.dumps({"verdict": "incorrect", "first_error_step": 1, "confidence": 0.5,
                    "error_type": "arithmetic", "evidence": 42}),
    ],
    ids=["garbage", "list", "bad_verdict", "confidence_high", "null_confidence",
         "non_string_evidence"],
)
def test_unusable_responses_return_none_rather_than_raising(bad):
    """A parse failure is data about the model, reported as a rate.

    Raising would abort a 190k-generation run over one malformed line. Booleans are
    rejected explicitly because ``isinstance(True, int)`` is True in Python.
    """
    assert parse_audit(bad, n_steps=3) is None


@pytest.mark.parametrize(
    "step", [None, 0, 9, True, "two"],
    ids=["null", "zero", "out_of_range", "bool", "string"],
)
def test_unusable_localization_keeps_the_verdict(step):
    """An unusable first_error_step must not discard the verdict.

    FAR -- the primary metric -- needs only the verdict. Gemma-4 really does return
    verdict="incorrect" with first_error_step=null, and the schema permits it. Rejecting
    the whole response would throw away primary-metric data, and would bias FAR if models
    omit the step precisely when they are least sure.
    """
    raw = json.dumps({
        "verdict": "incorrect", "first_error_step": step, "confidence": 0.5,
        "error_type": "arithmetic", "evidence": "x",
    })

    audit = parse_audit(raw, n_steps=3)

    assert audit is not None
    assert audit.reported_error
    assert audit.first_error_step is None
    assert not audit.localization_usable


def test_valid_localization_is_marked_usable():
    raw = json.dumps({
        "verdict": "incorrect", "first_error_step": 2, "confidence": 0.5,
        "error_type": "arithmetic", "evidence": "x",
    })

    audit = parse_audit(raw, n_steps=3)

    assert audit is not None and audit.localization_usable and audit.first_error_step == 2


def test_correct_verdict_has_no_localization_to_use():
    raw = json.dumps({
        "verdict": "correct", "first_error_step": None, "confidence": 0.5,
        "error_type": None, "evidence": "x",
    })

    audit = parse_audit(raw, n_steps=3)

    assert audit is not None and not audit.reported_error
    assert not audit.localization_usable


def test_unusable_error_type_keeps_the_verdict():
    """Same rule as localization: a bad secondary field never discards the verdict."""
    raw = json.dumps({
        "verdict": "incorrect", "first_error_step": 2, "confidence": 0.5,
        "error_type": "not_a_real_type", "evidence": "x",
    })

    audit = parse_audit(raw, n_steps=3)

    assert audit is not None
    assert audit.reported_error
    assert audit.error_type is None
    assert audit.first_error_step == 2


def test_schema_bounds_evidence_length():
    """Guided decoding constrains legal tokens, not when generation stops.

    Without a maxLength the model writes long evidence, hits max_tokens mid-string, and the
    JSON is truncated into a parse failure. That loss is not random: long evidence
    correlates with hard items and with "incorrect" verdicts, so discarding truncated
    samples biases FAR and detection rate downward. Bounding the string lets the constrained
    decoder close the object instead.
    """
    evidence = AUDIT_SCHEMA["properties"]["evidence"]

    assert evidence.get("maxLength"), "evidence must be length-bounded"
    assert evidence["maxLength"] <= 400


@pytest.mark.parametrize("control", CONTROLS)
def test_ladder_excludes_the_instrument_controls(control):
    """The controls validate the instrument; none is a claim about responsibility."""
    assert control in CONDITIONS
    assert control not in LADDER


def test_the_two_framing_primes_are_exact_mirrors():
    """PC and PCL must differ only in base-rate direction and which loss is the worse one.

    The ladder predicts FAR *falls* under repair responsibility, so a strictness-only control
    tests the wrong direction: a model near the bottom of its FAR range can respond upward and
    still have no room to move down. PCL is the direction-matched control, and it is only
    interpretable as one if it is the same manipulation reversed rather than a differently
    worded prompt.
    """
    pc, pcl = CONDITIONS["PC"], CONDITIONS["PCL"]

    assert "very often contain" in pc
    assert "very rarely contain" in pcl
    assert "for missing an error than for reporting one" in pc
    assert "for reporting an error that turns out to be benign than for missing one" in pcl
    assert abs(len(pc) - len(pcl)) <= 4, "near-equal length keeps token count out of the effect"


def test_the_hard_control_states_a_decision_rule_rather_than_a_framing():
    """PCH separates 'the DV cannot move' from 'framing cannot move it'.

    PC and PCL are framings and may legitimately produce small effects. PCH is an explicit
    threshold instruction, so a null there is a dead dependent variable -- a measurement
    failure, not a finding. The two must not be conflated, hence the separate condition.
    """
    pch = CONDITIONS["PCH"]

    assert "incorrect" in pch, "PCH must name the verdict it forces toward"
    assert "criticised" not in pch, "PCH is an instruction, not a reputational framing"


def test_r3_and_r3u_differ_only_in_conditionality():
    """R3 - R3u is the paper's identifying contrast.

    Without R3u, R3 > R0 is indistinguishable from Jin & Chen 2026's published finding that
    asking for a fix changes verdicts. The two sentences must therefore differ only in whether
    the repair is contingent on the verdict.
    """
    r3, r3u = CONDITIONS["R3"], CONDITIONS["R3u"]

    assert "only if your audit reports an error" in r3
    assert "whether or not your audit reports an error" in r3u
    assert r3.replace("and only if your audit reports an error", "") == r3u.replace(
        "whether or not your audit reports an error", ""
    ).replace("a corrected solution", "the corrected solution")


def test_r0p_isolates_generic_future_workload_from_repair_specifically():
    """R0p was cut for redundancy, and the ladder run showed the cut was wrong.

    R1 was meant to control for "future work happens at all", but R1 itself moves FAR -- it is an
    effect, not a control. Without a condition promising future work *unrelated to repair*, that
    rise cannot be told apart from any stated future task raising the flagging rate, which makes the
    ladder a story about workload rather than responsibility. So R0p must promise work that is
    real, comparable in weight, and unrelated to the audit.
    """
    r0p = CONDITIONS["R0p"]

    assert r0p, "R0p must define a future arrangement"
    assert "correct" not in r0p and "repair" not in r0p, "R0p must not mention repair"
    assert "R0p" in LADDER, "R0p is a rung, not an instrument control"
    # Length-comparable to R1, or a token-count difference confounds the contrast it exists for.
    assert abs(len(r0p) - len(CONDITIONS["R1"])) <= 25


# --- per-generation capture (E3) ----------------------------------------------------------------
#
# The clean arm persists probabilities and flags only, so a "false alarm" cannot be inspected: no
# one -- us included -- can tell a model applying a stricter-but-defensible justification standard
# from one that is simply wrong. `unjustified_step` is a permitted error_type and Ministral's
# baseline FAR is 0.684, which is exactly the shape a stricter standard would produce. Hand-auditing
# needs the evidence string and the claimed step, per generation.

def _raw(**over) -> str:
    base = {"verdict": "incorrect", "first_error_step": 2, "confidence": 0.8,
            "error_type": "arithmetic", "evidence": "step 2 adds wrong"}
    return json.dumps(base | over)


def test_records_carry_the_fields_the_summary_throws_away():
    rows = audit_records([[_raw()]], [ITEM])

    assert rows == [{"item_id": ITEM.item_id, "sample": 0, "verdict": "incorrect",
                     "first_error_step": 2, "confidence": 0.8, "error_type": "arithmetic",
                     "evidence": "step 2 adds wrong", "parsed": True}]


def test_a_parse_failure_is_recorded_rather_than_dropped():
    """Dropping it would silently shrink the denominator the false-alarm rate is read against."""
    rows = audit_records([["not json", _raw()]], [ITEM])

    assert [r["parsed"] for r in rows] == [False, True]
    assert rows[0]["verdict"] is None
    assert rows[0]["sample"] == 0 and rows[1]["sample"] == 1


def test_a_missing_choice_is_recorded_too():
    """`None` is what the backend returns for a failed choice; it must not vanish either."""
    rows = audit_records([[None]], [ITEM])

    assert rows[0]["parsed"] is False and rows[0]["evidence"] is None


def test_a_correct_verdict_carries_no_step_or_type():
    """Same normalisation as `parse_audit`, or the hand-audit sample is polluted by non-flags."""
    rows = audit_records([[_raw(verdict="correct")]], [ITEM])

    assert rows[0]["first_error_step"] is None and rows[0]["error_type"] is None


def test_every_item_appears_even_when_all_of_its_generations_fail():
    rows = audit_records([[None, None]], [ITEM])

    assert len(rows) == 2 and {r["item_id"] for r in rows} == {ITEM.item_id}


def test_mismatched_lengths_raise_rather_than_silently_truncating():
    """zip() without strict= would drop the tail and the capture would under-report."""
    with pytest.raises(ValueError):
        audit_records([[_raw()], [_raw()]], [ITEM])
