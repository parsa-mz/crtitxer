"""Tests for the R4 2x2.

The four cells cross placement (assistant turn / user turn) with attribution (self / peer).
Everything except placement and a length-matched label must be byte-identical across all four,
and the target audit request must be byte-identical to R0's. If any of that drifts, the
off-diagonal cells stop isolating placement from label and the paper's core contrast is dead.
"""

from __future__ import annotations

import itertools

import pytest

from critxer.core.audit import AuditItem, build_audit_messages
from critxer.core.r4 import (
    CELLS,
    LABELS,
    REPAIR_REQUEST,
    FillerEpisode,
    WarmupEpisode,
    build_audit_only_messages,
    build_filler_messages,
    build_neutral_messages,
    build_r4_messages,
    check_pool_fingerprints,
    injected_candidates,
    natural_candidates,
    neutral_map,
    select_cells,
    warmup_content,
)

TARGET = AuditItem("gsm8k-200", "Jane has 3 apples and buys 4 more. How many?",
                   ["Jane starts with 3.", "3 + 4 = 7.", "The answer is 7."])
WARMUP = WarmupEpisode(
    item=AuditItem("gsm8k-999", "Bob has 10 pears and eats 2. How many?",
                   ["Bob starts with 10.", "10 - 2 = 8.", "The answer is 8."]),
    audit='{"verdict": "correct", "first_error_step": null, "confidence": 0.9, '
          '"error_type": null, "evidence": "All steps check out."}',
    repair="No repair needed; the solution is correct.",
)


def _target_request(messages: list[dict]) -> str:
    """The last user turn is always the target audit request."""
    return [m for m in messages if m["role"] == "user"][-1]["content"]


@pytest.mark.parametrize("cell", CELLS)
def test_r0_request_appears_verbatim_as_the_tail_of_the_final_turn(cell):
    """R4 adds history; it must not reword the ask.

    Exact equality with R0's turn is impossible by design -- R4's whole point is that prior
    context exists, and a handoff sentence is needed to mark where the earlier episode ends.
    What must hold is that the R0 request is present *verbatim* and is the last thing the model
    reads, so the present task is textually unchanged.
    """
    expected = build_audit_messages(TARGET, "R0", "F1")[1]["content"]

    assert _target_request(build_r4_messages(TARGET, WARMUP, cell, "F1")).endswith(expected)


def test_the_final_turn_is_identical_across_all_four_cells():
    """The invariant that actually matters for the 2x2: cells differ only in history."""
    tails = {
        _target_request(build_r4_messages(TARGET, WARMUP, cell, "F1")) for cell in CELLS
    }

    assert len(tails) == 1


@pytest.mark.parametrize(("a", "b"), list(itertools.combinations(CELLS, 2)))
def test_cells_differ_only_by_placement_and_label(a, b):
    """Strip the two labels and every cell must carry exactly the same text.

    Compared as a concatenation of all message contents, so a cell that moved content between
    turns still has to account for every character of it.
    """
    def stripped(cell: str) -> str:
        text = "\n".join(m["content"] for m in build_r4_messages(TARGET, WARMUP, cell, "F1"))
        for label in LABELS.values():
            text = text.replace(label, "")
        return " ".join(text.split())

    assert stripped(a) == stripped(b)


def test_attribution_labels_are_length_matched():
    """An unequal label length would itself be a token-count difference between cells."""
    assert len(LABELS["self"]) == len(LABELS["peer"])


@pytest.mark.parametrize("cell", CELLS)
def test_warmup_content_is_byte_identical_across_cells(cell):
    """The episode text is fixed; only where it sits and who it is credited to varies."""
    rendered = "\n".join(m["content"] for m in build_r4_messages(TARGET, WARMUP, cell, "F1"))

    assert WARMUP.audit in rendered
    assert WARMUP.repair in rendered


@pytest.mark.parametrize("cell", ["AS", "AO"])
def test_assistant_placement_puts_the_episode_in_assistant_turns(cell):
    messages = build_r4_messages(TARGET, WARMUP, cell, "F1")
    assistant = "\n".join(m["content"] for m in messages if m["role"] == "assistant")

    assert WARMUP.audit in assistant
    assert WARMUP.repair in assistant


@pytest.mark.parametrize("cell", ["US", "UO"])
def test_user_placement_keeps_the_episode_out_of_assistant_turns(cell):
    """This is the Khullar et al. manipulation: off-policy content sits in a user turn."""
    messages = build_r4_messages(TARGET, WARMUP, cell, "F1")

    assert not [m for m in messages if m["role"] == "assistant"]
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    assert WARMUP.audit in user


@pytest.mark.parametrize(("cell", "expected"), [("AS", "self"), ("AO", "peer"),
                                               ("US", "self"), ("UO", "peer")])
def test_each_cell_uses_its_own_attribution_label(cell, expected):
    rendered = "\n".join(m["content"] for m in build_r4_messages(TARGET, WARMUP, cell, "F1"))
    other = "peer" if expected == "self" else "self"

    assert LABELS[expected] in rendered
    assert LABELS[other] not in rendered


def test_warmup_content_is_shared_by_construction():
    """One helper renders the episode, so no cell can accidentally render it differently."""
    variants = {
        "\n".join(warmup_content(WARMUP, attribution)) for attribution in ("self", "peer")
    }

    assert len(variants) == 2  # they differ, but only by label
    stripped = {
        text.replace(LABELS["self"], "").replace(LABELS["peer"], "") for text in variants
    }
    assert len(stripped) == 1


def test_unknown_cell_is_rejected():
    with pytest.raises(ValueError, match="unknown cell"):
        build_r4_messages(TARGET, WARMUP, "ZZ", "F1")


class TestFillerControlsForContextRatherThanEpisode:
    """All four 2x2 cells dropped FAR ~3.5-4.2pp and sit within 0.7pp of each other.

    Measured on qwen3.6-35B-A3B, 465 targets: AS 0.1970, AO 0.1953, US 0.1936, UO 0.1897 against
    R0 at 0.2316. The placement contrast is +0.45pp and the label contrast +0.28pp -- both null.
    What the four cells share, and R0 lacks, is *any prior exchange at all*.

    So the effect is either the episode being an audit->repair, or merely context preceding the
    task. Nothing in the 2x2 distinguishes those, and the same omission already cost R0p once.
    The filler cell is a prior exchange of comparable length on a non-audit task, in the
    same position as the assistant-placement cells.
    """

    FILLER = FillerEpisode(
        request="Summarise the following problem statement in a few sentences.\n\nBob has pears.",
        response="A short restatement of a word problem about counting pears.",
    )

    def test_the_filler_cell_carries_no_audit_and_no_repair(self):
        """If the filler exchange contained a verdict it would be an episode, not a control.

        Scoped to the *prior exchange*, not the whole prompt: the system turn names the audit
        schema in every condition including R0, so a whole-prompt check would be vacuous.
        """
        messages = build_filler_messages(TARGET, self.FILLER, "F1")
        prior = "\n".join(m["content"] for m in messages[1:3])

        assert "verdict" not in prior
        assert REPAIR_REQUEST not in prior

    def test_the_filler_sits_in_the_same_turns_as_assistant_placement(self):
        messages = build_filler_messages(TARGET, self.FILLER, "F1")
        assistant = [m for m in messages if m["role"] == "assistant"]

        assert len(assistant) == 1
        assert assistant[0]["content"] == self.FILLER.response

    def test_the_target_request_is_identical_to_the_2x2_cells(self):
        """Otherwise the filler is not comparable to the cells it is controlling for."""
        filler_tail = _target_request(build_filler_messages(TARGET, self.FILLER, "F1"))
        cell_tail = _target_request(build_r4_messages(TARGET, WARMUP, "AS", "F1"))

        assert filler_tail == cell_tail


class TestAuditOnlySeparatesTheVerdictFromTheRepair:
    """The filler rules out length and non-audit content; it does not rule out the verdict.

    86-88% of the generated warmup episodes have the model reporting "correct" (27B 6/50 flagged an
    error, 35B 7/50), and on the 27B 49 of 50 "repairs" are a second emission of the audit JSON.
    So every 2x2 cell puts one or two assertions of ``"verdict": "correct"`` in the context of a
    model about to emit that same field. An episode lowering FAR is therefore equally consistent
    with verdict priming -- the model agrees with itself -- and with the paper's claim that having
    performed a repair changes the diagnosis. The AF filler cannot separate them because it removes
    the verdict and the repair together.

    AV keeps the audit and drops only the repair. AS - AV is then the repair's own contribution,
    which is what "enacted responsibility" has to mean if it means anything.
    """

    def test_the_prior_exchange_keeps_the_audit_and_drops_the_repair(self):
        messages = build_audit_only_messages(TARGET, WARMUP, "F1")
        prior = "\n".join(m["content"] for m in messages[1:-1])

        assert WARMUP.audit in prior
        assert WARMUP.repair not in prior
        assert REPAIR_REQUEST not in prior

    def test_the_audit_sits_in_the_only_assistant_turn(self):
        """Matched to assistant placement, so AS - AV is not also a placement contrast."""
        messages = build_audit_only_messages(TARGET, WARMUP, "F1")
        assistant = [m for m in messages if m["role"] == "assistant"]

        assert len(assistant) == 1
        assert WARMUP.audit in assistant[0]["content"]

    def test_it_is_attributed_to_self_like_the_AS_cell(self):
        rendered = "\n".join(m["content"] for m in build_audit_only_messages(TARGET, WARMUP, "F1"))

        assert LABELS["self"] in rendered
        assert LABELS["peer"] not in rendered

    def test_the_target_request_is_identical_to_the_2x2_cells(self):
        av_tail = _target_request(build_audit_only_messages(TARGET, WARMUP, "F1"))
        cell_tail = _target_request(build_r4_messages(TARGET, WARMUP, "AS", "F1"))

        assert av_tail == cell_tail

    def test_AS_is_AV_plus_exactly_the_repair_turns(self):
        """The contrast is only interpretable if nothing else differs between them.

        Compared as a concatenation of all turns, so content that merely moved still has to
        account for every character.
        """
        def flat(messages: list[dict]) -> list[str]:
            return [" ".join(m["content"].split()) for m in messages]

        av = flat(build_audit_only_messages(TARGET, WARMUP, "F1"))
        as_ = flat(build_r4_messages(TARGET, WARMUP, "AS", "F1"))
        added = [" ".join(REPAIR_REQUEST.split()),
                 " ".join(f"{LABELS['self']} {WARMUP.repair}".split())]

        assert as_ == [*av[:-1], *added, av[-1]]


class TestInjectedCandidatesSupplyIncorrectVerdictEpisodes:
    """AV holds the verdict and drops the repair; this holds the structure and moves the verdict.

    An episode built on a *faulty* trace makes the model report "incorrect" and then perform a
    real correction, so the AX cell is a genuine audit->repair whose verdict is the opposite of
    the 86-88% majority. If verdict priming drives the effect, AX should push FAR *up*; if
    experienced repair does, AX should behave like AS.

    Candidates come from the injected pairs, which `allocation.json` holds disjoint from both the
    clean targets and the warmup arm -- so an AX episode can never be a trace the model is later
    asked to judge cold.
    """

    PAIRS = [
        {"id": "gsm8k-1", "problem": "P1", "original": ["a", "b"],
         "early": {"k": 1, "steps": ["a-bad", "b1"], "downstream": 1},
         "late": {"k": 2, "steps": ["a", "b-bad"], "downstream": 0}},
        {"id": "gsm8k-2", "problem": "P2", "original": ["c", "d"],
         "early": {"k": 1, "steps": ["c-bad", "d1"], "downstream": 1},
         "late": {"k": 2, "steps": ["c", "d-bad"], "downstream": 0}},
    ]

    def test_the_early_arm_is_exhausted_before_the_late_arm(self):
        """One fault per source trace while any trace is still unused.

        Drawing both arms of a trace before moving on would build a pool in which two episodes
        are near-duplicates of each other, narrowing the episode variation the pool exists to
        provide.
        """
        got = injected_candidates(self.PAIRS, limit=3)

        assert [i.item_id for i in got] == ["gsm8k-1::early", "gsm8k-2::early", "gsm8k-1::late"]

    def test_the_injected_steps_are_used_not_the_original(self):
        early = injected_candidates(self.PAIRS, limit=1)[0]

        assert early.steps == ["a-bad", "b1"]
        assert early.problem == "P1"

    def test_the_limit_caps_the_pool(self):
        assert len(injected_candidates(self.PAIRS, limit=2)) == 2

    def test_asking_for_more_than_exist_returns_all_of_them(self):
        """Silently returning a short pool is right here; the run reports the achieved n."""
        assert len(injected_candidates(self.PAIRS, limit=99)) == 4


class TestNaturalFaultCandidatesForTheAXCell:
    """The AX pool built from ProcessBench's own labelled-incorrect traces.

    `injected_candidates` needs an injector, and the injector was Ministral -- so Ministral cannot
    take AX or AXN without auditing faults it authored, and the verdict-priming test rests on one
    clean model. ProcessBench already ships expert-annotated faulty traces, which removes both the
    injector and the synthetic-fault caveat and readmits Ministral.
    """

    TRACES = [
        {"id": "gsm8k-1", "split": "gsm8k", "problem": "P1", "steps": ["a", "b"], "gold_label": 1},
        {"id": "gsm8k-2", "split": "gsm8k", "problem": "P2", "steps": ["c", "d"], "gold_label": 0},
        {"id": "math-3", "split": "math", "problem": "P3", "steps": ["e", "f"], "gold_label": 1},
        {"id": "omnimath-4", "split": "omnimath", "problem": "P4", "steps": ["g", "h"],
         "gold_label": 0},
    ]

    def test_it_takes_one_source_at_a_time_before_repeating_one(self):
        """Round-robin across splits, because the available pool is not balanced.

        1,246 traces are available and 565 of them are omnimath; taking them in file order would
        build a pool that is mostly one source, so episode variation would confound source with
        everything else. `injected_candidates` solves the analogous problem by exhausting `early`
        before `late`.
        """
        got = natural_candidates(self.TRACES, limit=4)

        assert [i.item_id for i in got] == ["gsm8k-1", "math-3", "omnimath-4", "gsm8k-2"]

    def test_the_trace_text_is_carried_through(self):
        first = natural_candidates(self.TRACES, limit=1)[0]

        assert first.problem == "P1"
        assert first.steps == ["a", "b"]

    def test_excluded_ids_never_enter_the_pool(self):
        """The invariant this function exists to protect.

        A warmup item that is also a detection-arm target means the model audited *and repaired*
        the trace it is later asked to judge cold. The detection arm is 929 of the same
        labelled-incorrect traces, so this is a live collision, not a hypothetical one.
        """
        got = natural_candidates(self.TRACES, limit=4, exclude={"gsm8k-1", "math-3"})

        assert [i.item_id for i in got] == ["gsm8k-2", "omnimath-4"]

    def test_the_limit_caps_the_pool(self):
        assert len(natural_candidates(self.TRACES, limit=2)) == 2

    def test_asking_for_more_than_exist_returns_all_of_them(self):
        assert len(natural_candidates(self.TRACES, limit=99)) == 4


class TestNeutralContinuationIsolatesTheRepair:
    """`AS - AV` varies the repair *and* whether a continuation exists at all.

    AV deletes the repair request and the repair, so AS - AV changes content and length at once,
    and AF cannot fix that because it removes the audit too. Under episode clustering AS - AV
    survives on only one of three models, so it does less work than it looked like it did.

    AN keeps the audit, keeps a continuation of matched length and turn structure, and makes that
    continuation *not a repair* -- a restatement of what the audit found. The identified repair
    contribution is then AS - AN, with AV - AN separating "a continuation exists" from "the
    continuation is a repair".
    """

    NEUTRAL = "Restated: the solution's steps were checked in order and the verdict above stands."

    def test_the_continuation_is_not_a_repair_request(self):
        messages = build_neutral_messages(TARGET, WARMUP, self.NEUTRAL, "F1")
        prior = "\n".join(m["content"] for m in messages[1:-1])

        assert REPAIR_REQUEST not in prior
        assert WARMUP.repair not in prior
        assert self.NEUTRAL in prior

    def test_it_has_the_same_turn_structure_as_the_AS_cell(self):
        """Same number of turns in the same roles, so only the continuation's content differs."""
        neutral = build_neutral_messages(TARGET, WARMUP, self.NEUTRAL, "F1")
        as_ = build_r4_messages(TARGET, WARMUP, "AS", "F1")

        assert [m["role"] for m in neutral] == [m["role"] for m in as_]

    def test_the_audit_is_preserved_verbatim(self):
        rendered = "\n".join(
            m["content"] for m in build_neutral_messages(TARGET, WARMUP, self.NEUTRAL, "F1"))

        assert WARMUP.audit in rendered

    def test_the_target_request_is_identical_to_the_2x2_cells(self):
        neutral_tail = _target_request(build_neutral_messages(TARGET, WARMUP, self.NEUTRAL, "F1"))
        cell_tail = _target_request(build_r4_messages(TARGET, WARMUP, "AS", "F1"))

        assert neutral_tail == cell_tail

    def test_it_is_attributed_to_self_like_the_AS_cell(self):
        rendered = "\n".join(
            m["content"] for m in build_neutral_messages(TARGET, WARMUP, self.NEUTRAL, "F1"))

        assert LABELS["self"] in rendered
        assert LABELS["peer"] not in rendered


# --- neutral continuation sets ------------------------------------------------------------------
#
# Two cells now consume one: AN pairs neutral continuations with the ordinary (mostly
# verdict-"correct") pool, and AXN pairs them with the incorrect-verdict pool so that AX - AXN
# varies only whether a *genuine* correction followed a *genuine* fault. Both need the same three
# guarantees, and a silent gap in either would substitute a different pairing rather than fail.

_INCORRECT = WarmupEpisode(
    item=AuditItem("math-77::early", "What is 2 + 2 * 3?",
                   ["2 * 3 = 5.", "2 + 5 = 7.", "The answer is 7."]),
    audit='{"verdict": "incorrect", "first_error_step": 1, "confidence": 0.9, '
          '"error_type": "arithmetic", "evidence": "2 * 3 = 6, not 5."}',
    repair="Step 1 should read 2 * 3 = 6, so 2 + 6 = 8. The answer is 8.",
)


def _blob(auditor: str, *pairs: tuple[str, str]) -> dict:
    return {"auditor": auditor, "neutrals": [{"item_id": i, "neutral": n} for i, n in pairs]}


def test_neutral_map_returns_the_continuation_for_each_episode():
    got = neutral_map(_blob("qwen", ("gsm8k-999", "I concluded the solution was correct.")),
                      "qwen", [WARMUP], "AN")

    assert got == {"gsm8k-999": "I concluded the solution was correct."}


def test_neutral_map_rejects_a_set_written_by_a_different_model():
    """The continuation has to be the auditor's own work, like every other part of the episode."""
    blob = _blob("ministral-14B", ("gsm8k-999", "..."))

    with pytest.raises(SystemExit, match="ministral-14B"):
        neutral_map(blob, "qwen", [WARMUP], "AN")


def test_neutral_map_rejects_a_set_that_does_not_cover_the_whole_pool():
    """A partial set is the dangerous case: the cell would still build, on a different pairing."""
    blob = _blob("qwen", ("gsm8k-999", "..."))

    with pytest.raises(SystemExit, match="AXN"):
        neutral_map(blob, "qwen", [WARMUP, _INCORRECT], "AXN")


# --- pool consistency --------------------------------------------------------------------------
#
# T=0 is not reproducible on this stack, so two cells of the same condition can be built from
# different episodes without any aggregate revealing it. Each cell records a content hash of its
# pool and contrasts must not cross two hashes. But there are now TWO legitimate pools -- the
# ordinary episodes and the incorrect-verdict ones -- so a single "all hashes must match" rule
# fires on a correct run, which is how it first behaved when AXN was added.

def _cells(**prints: str) -> dict[str, dict]:
    return {c: {"episode_fingerprint": p} for c, p in prints.items()}


def test_cells_from_one_pool_must_share_a_fingerprint():
    with pytest.raises(SystemExit, match="ordinary"):
        check_pool_fingerprints(_cells(AS="aaa", AO="aaa", AN="bbb"))


def test_the_incorrect_verdict_cells_are_allowed_their_own_fingerprint():
    """AX and AXN are built on faulty traces, so their hash differs by design, not by drift."""
    check_pool_fingerprints(_cells(AS="aaa", AN="aaa", AX="bbb", AXN="bbb"))


def test_but_the_incorrect_verdict_cells_must_agree_with_each_other():
    """AX - AXN is a contrast within that pool, so a mismatch there voids it just as badly."""
    with pytest.raises(SystemExit, match="incorrect-verdict"):
        check_pool_fingerprints(_cells(AS="aaa", AN="aaa", AX="bbb", AXN="ccc"))


def test_cells_predating_fingerprinting_are_not_treated_as_a_second_pool():
    """R0 has no episode at all, and the earliest cells were written before hashes existed."""
    check_pool_fingerprints({"R0": {}, "AS": {"episode_fingerprint": "aaa"}, "AF": {}})


# --- cell selection ----------------------------------------------------------------------------
#
# The reasoning-enabled arm (E2) runs R0/AS/AF only: with thinking on, each generation costs ~8x
# the tokens, so running the full 2x2 as well doubles a multi-hour job for cells that arm does not
# report. Subsetting has to fail loudly on a cell that is not available, because the quiet failure
# -- silently dropping AF because --filler was omitted -- produces an arm whose control is simply
# missing, and every downstream contrast then compares against nothing.

def test_selecting_a_subset_keeps_the_canonical_order():
    """Not the order the user typed: R0 must run first so a crash still leaves the baseline."""
    assert select_cells("AF,R0,AS", ("R0", "AS", "AO", "AF")) == ("R0", "AS", "AF")


def test_selecting_nothing_runs_every_available_cell():
    assert select_cells("", ("R0", "AS", "AF")) == ("R0", "AS", "AF")


def test_requesting_an_unavailable_cell_raises_rather_than_dropping_it():
    """AF without --filler is the real case: the arm would run, minus its control."""
    with pytest.raises(SystemExit, match="AF"):
        select_cells("R0,AS,AF", ("R0", "AS", "AO"))


def test_requesting_an_unknown_cell_name_raises():
    with pytest.raises(SystemExit, match="AZ"):
        select_cells("R0,AZ", ("R0", "AS"))


def test_whitespace_around_names_is_tolerated():
    assert select_cells(" R0 , AS ", ("R0", "AS", "AF")) == ("R0", "AS")


class TestEpisodeItemsPrefersWhatThePoolCarries:
    """A pool's inline problem/steps must win over the benchmark lookup.

    Injected and natural-fault pools carry corrupted steps that exist only in the pool: the trace
    ProcessBench holds under that id is the *clean* one. `run_detection` built its items from the
    lookup alone, so an AX or injected pool would have raised KeyError for ids present inline, and a
    pool that happened to share ids would silently have audited the wrong trace.
    """

    def _ep(self, item_id, **extra):
        return {"item_id": item_id, "audit": "{}", "repair": "{}", **extra}

    def test_inline_steps_are_used_and_the_lookup_is_not_consulted(self):
        from critxer.core.r4 import episode_items

        eps = [self._ep("x-1", problem="p-inline", steps=["s1", "s2"])]
        clean = {"x-1": AuditItem("x-1", "p-clean", ["clean1", "clean2"])}

        items = episode_items(eps, clean)

        assert items[0].problem == "p-inline"
        assert items[0].steps == ["s1", "s2"]

    def test_a_pool_without_inline_steps_falls_back_to_the_lookup(self):
        from critxer.core.r4 import episode_items

        got = episode_items([self._ep("x-2")],
                            {"x-2": AuditItem("x-2", "p-clean", ["a", "b"])})

        assert got[0].problem == "p-clean"

    def test_an_id_in_neither_place_names_the_pool_rather_than_raising_keyerror(self):
        from critxer.core.r4 import episode_items

        with pytest.raises(SystemExit, match="x-3"):
            episode_items([self._ep("x-3")], {})
