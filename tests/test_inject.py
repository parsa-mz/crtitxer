"""Tests for injection prompt construction and splicing."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from critxer.core import backend
from critxer.core.inject import (
    FAMILY_BRIEFS,
    SENTINEL,
    build_injection_messages,
    inject_at,
    normalise_step,
    parse_suffix,
    splice,
    suffix_schema,
)
from critxer.core.repairgym import validate_injection

ORIGINAL = ["Start with 10 apples.", "Buy 5 more: 10 + 5 = 15.", "Eat 3: 15 - 3 = 12.",
            "The answer is 12."]


def test_splice_keeps_the_prefix_byte_identical():
    """Prefix integrity is guaranteed structurally, not requested politely.

    Asking a model to rewrite a whole trace and then checking the prefix survived wastes
    generations on avoidable rejections. We only ever ask for steps k..n.
    """
    spliced = splice(ORIGINAL, step_k=3, suffix=["Eat 3: 15 - 4 = 11.", "The answer is 11."])

    assert spliced[:2] == ORIGINAL[:2]
    assert spliced[2:] == ["Eat 3: 15 - 4 = 11.", "The answer is 11."]


def test_splice_rejects_a_suffix_of_the_wrong_length():
    """A wrong-length suffix would silently change the step count."""
    with pytest.raises(ValueError, match="suffix length"):
        splice(ORIGINAL, step_k=3, suffix=["only one step"])


@pytest.mark.parametrize("family", sorted(FAMILY_BRIEFS))
def test_prompt_asks_only_for_the_suffix(family):
    messages = build_injection_messages(ORIGINAL, step_k=3, family=family)
    rendered = "\n".join(m["content"] for m in messages)

    assert "Step 3" in rendered and "Step 4" in rendered
    assert "2 steps" in rendered  # steps 3..4 of a 4-step trace


@pytest.mark.parametrize("family", sorted(FAMILY_BRIEFS))
def test_prompt_demands_consistent_propagation(family):
    """Both families must propagate, or repair cost stops being the only difference."""
    rendered = "\n".join(
        m["content"] for m in build_injection_messages(ORIGINAL, step_k=3, family=family)
    )

    assert "propagat" in rendered.lower()


def test_families_differ_in_what_they_ask_for():
    local = FAMILY_BRIEFS["local"]
    structural = FAMILY_BRIEFS["structural"]

    assert local != structural
    assert "value" in local.lower()
    assert "method" in structural.lower()


def test_structural_brief_forbids_a_bare_operator_swap():
    """Gate-2 calibration measured 0.94-0.97 detection for structural faults -- ceilinged.

    The cause was operator swaps (`R = 4B` -> `R = 4 + B`), which are glaring once the problem
    statement is in context. The brief must ask for a plausible wrong method instead, or the
    family carries no usable variance for the repairability-tilt interaction.
    """
    brief = FAMILY_BRIEFS["structural"].lower()

    assert "do not simply swap" in brief
    assert "plausib" in brief


def test_suffix_schema_pins_the_step_count():
    """Constrained decoding enforces the length so the step-count check cannot fail."""
    schema = suffix_schema(n_suffix=3)

    assert schema["properties"]["steps"]["minItems"] == 3
    assert schema["properties"]["steps"]["maxItems"] == 3


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown family"):
        build_injection_messages(ORIGINAL, step_k=2, family="nonsense")


class TestNormalisation:
    """Injected items must not be distinguishable from clean ones on formatting.

    The audit prompt numbers steps itself, so a model that echoes "Step 7: " into the step
    text produces "Step 7: Step 7: ..." on injected items and nothing like it on clean ones.
    That is a detectable cue unrelated to the fault, which breaks the comparison the whole
    design rests on.
    """

    @pytest.mark.parametrize(
        "raw",
        ["Step 7: Now that we know B = 5.", "step 7 : Now that we know B = 5.",
         "**Step 7:** Now that we know B = 5.", "Step 7. Now that we know B = 5."],
        ids=["plain", "spaced", "bold", "period"],
    )
    def test_strips_an_echoed_step_prefix(self, raw):
        assert normalise_step(raw, original="x") == "Now that we know B = 5."

    def test_leaves_a_genuine_reference_to_another_step_alone(self):
        """"From Step 3 we know..." is real content, not an echoed prefix."""
        text = "From Step 3 we know B = 5."

        assert normalise_step(text, original="x") == text

    def test_collapses_newlines_when_the_original_step_was_single_line(self):
        """Line structure must not become a cue in either direction.

        Ministral writes multi-line steps. If the original step was one line and the injected
        one is three, that is a visible layout difference unrelated to the fault -- so match
        the original's structure. Where the original was itself multi-line, leave it alone.
        """
        assert normalise_step("a\n b\n  c", original="one line") == "a b c"
        assert normalise_step("a\nb", original="orig\nhas\nlines") == "a\nb"

    def test_splice_normalises_the_suffix(self):
        spliced = splice(ORIGINAL, step_k=3, suffix=["Step 3: Eat 3: 15 - 4 = 11.",
                                                    "Step 4: The answer is 11."])

        assert spliced[2] == "Eat 3: 15 - 4 = 11."
        assert spliced[3] == "The answer is 11."


def test_validation_rejects_a_step_prefix_that_survived():
    """Defence in depth: if normalisation is ever bypassed, the gate still catches it."""
    injected = [*ORIGINAL[:2], "Step 3: Eat 3: 15 - 4 = 11.", "The answer is 11."]

    result = validate_injection(ORIGINAL, injected, step_k=3, original_answer="12")

    assert not result.ok
    assert any("step prefix" in r for r in result.reasons)


class TestSuffixParsing:
    r"""Plain text with a sentinel, not JSON.

    Asking for JSON meant the model wrote `\times` inside a string, JSON parsed the legal
    `\t` escape, and the step arrived as TAB + "imes". Measured on the first pilot: **50% of
    accepted injections were corrupted this way**, invisibly, because the JSON was valid.
    Sentinel-delimited plain text has no escaping layer to get wrong.
    """

    def test_parses_sentinel_delimited_steps(self):
        raw = "first step\n---STEP---\nsecond step\n---STEP---\nthird step"

        assert parse_suffix(raw, 3) == ["first step", "second step", "third step"]

    def test_preserves_backslashes_verbatim(self):
        raw = r"$R = 4 \times 5 = 20$" + "\n---STEP---\ndone"

        parsed = parse_suffix(raw, 2)

        assert parsed is not None
        assert r"\times" in parsed[0]
        assert "\t" not in parsed[0]

    def test_returns_none_on_the_wrong_step_count(self):
        """Count is checked here since plain text cannot pin it the way a schema did."""
        assert parse_suffix("only one\n---STEP---\ntwo", 3) is None

    def test_tolerates_surrounding_whitespace_and_blank_lines(self):
        raw = "\n  first  \n\n---STEP---\n\n  second \n"

        assert parse_suffix(raw, 2) == ["first", "second"]

    def test_returns_none_when_a_step_is_empty(self):
        assert parse_suffix("first\n---STEP---\n   \n", 2) is None


class TestEmphasisIsNotACue:
    """The manual audit found bold markdown in 16.3% of injected traces and 0% of originals.

    `normalise_step` already strips echoed step numbers and matches line structure, on the stated
    principle that formatting which appears only on injected items marks them. Markdown emphasis
    is the same class of cue and was missed: an auditor could learn "the bolded step is the
    injected one" and its detection rate would rise for reasons unrelated to the fault, directly
    inflating the injected arm's DV.
    """

    def test_bold_is_stripped_when_the_original_has_none(self):
        out = normalise_step("Total is **59 minutes**, so 0.98 hours.", "Total is 60 minutes.")

        assert "**" not in out
        assert "59 minutes" in out

    def test_italic_underscores_are_stripped_when_the_original_has_none(self):
        out = normalise_step("The sum is _22_ exactly.", "The sum is 20 exactly.")

        assert "_22_" not in out
        assert "22" in out

    def test_bold_survives_when_the_original_itself_uses_bold(self):
        """Stripping unconditionally would make *absence* of bold the cue instead.

        Same reasoning as the line-structure rule: match the original rather than impose a house
        style, so the correction cannot become a reverse cue on genuinely bold source traces.
        """
        out = normalise_step("So **x = 9** follows.", "So **x = 6** follows.")

        assert "**x = 9**" in out


class TestInjectAtCollectsWhyEachAttemptFailed:
    """One implementation, and the rejection reasons are part of its contract.

    It existed twice: `run_injection_set`'s collected every reason, and `run_position_calibration`'s
    was the same function with that collection removed. The reasons are what the acceptance audit's
    rejection breakdown is built from, so a caller that does not want them should discard them
    rather than run a second implementation that cannot produce them.
    """

    EP = backend.Endpoint("inj", "some/model", "http://x/v1")
    STEPS = ["2 + 2 = 4.", "4 * 3 = 12.", "12 - 2 = 10.", "So the answer is 10."]

    def _run(self, contents: list[str], attempts: int = 3):
        """Serve one canned generation per attempt, in order."""
        seen_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            i = min(len(seen_requests), len(contents) - 1)
            seen_requests.append(1)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": contents[i]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            })

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                return await inject_at(http, self.EP, self.STEPS, 2, "10", attempts)

        return asyncio.run(go()), len(seen_requests)

    def test_a_suffix_with_the_wrong_step_count_is_recorded_not_silently_retried(self):
        # One step where three are expected (steps 2..4).
        (injected, reasons), calls = self._run(["only one step"], attempts=2)

        assert injected is None
        assert calls == 2, "it should have retried"
        assert reasons.count("step count mismatch in suffix") == 2

    def test_an_accepted_injection_returns_the_spliced_steps_and_the_reasons_so_far(self):
        # Reaching 12 downstream would be rejected: 12 is step 2's PRE-fault value, so the
        # validator reads it as the fault not having propagated.
        good = SENTINEL.join(["4 * 3 = 15.", "15 - 2 = 13.", "So the answer is 13."])
        (injected, reasons), _ = self._run(["only one step", good], attempts=3)

        assert injected is not None
        assert injected[0] == self.STEPS[0], "the prefix must be untouched"
        assert "step count mismatch in suffix" in reasons, "earlier failures are still reported"

    def test_an_empty_generation_is_recorded_as_its_own_reason(self):
        (injected, reasons), _ = self._run([""], attempts=1)

        assert injected is None
        assert reasons == ["step count mismatch in suffix"] or reasons == ["no generation"]
