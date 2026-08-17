"""repairgym: controlled fault injection with a repair-cost knob.

Two fault families go into the *same* clean source trace, so the repair-cost contrast is
within-trace: **local** is a wrong numeric result for a correct operation, repaired by
fixing one value and recomputing downstream; **structural** is a wrong *method*, repaired by
re-deriving from that step.

Both are propagated consistently through the remaining steps, which is load-bearing: an
un-propagated fault leaves a *second* visible inconsistency downstream, changing the fault's
**detectability** rather than its **repair cost**, and the two families stop being comparable.

Generation is model-assisted; acceptance is not, and it is blocking -- a candidate that fails any
check is dropped, never patched.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# A purely numeric binary claim: `a op b = c`. Deliberately conservative -- a spurious match
# would reject a valid injection, so anything symbolic, exponentiated or subscripted is left
# alone. Character classes only (no nested quantifiers) so it cannot backtrack badly.
_NUM = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"
_OPS = {"+": "+", "-": "-", "*": "*", "/": "/",
        r"\times": "*", r"\cdot": "*", r"\div": "/", "×": "*", "÷": "/"}
_OP_ALT = "|".join(re.escape(k) for k in sorted(_OPS, key=len, reverse=True))
# Lookarounds reject a number that is part of a larger token (`b_1`, `2^3`) or the fractional
# tail of a decimal -- but must still allow a trailing sentence period, which an earlier
# `(?![\w.^_])` silently swallowed, so `"3 + 4 = 7."` matched nothing at all.
_EDGE_BEFORE = r"(?<![\w^_])(?<!\.)"
_EDGE_AFTER = r"(?![\w^_])(?!\.\d)"
# Chains, not just pairs: `10 + 7 + 6 = 23` is ubiquitous in these traces, and reading only
# its last two operands gives `7 + 6 = 23`. Measured against the 1,179 human-verified-correct
# clean seeds, the pairwise-only version flagged 17.56% of them as inconsistent.
_TERM = rf"{_EDGE_BEFORE}(?:{_NUM}){_EDGE_AFTER}"
_CLAIM = re.compile(rf"{_TERM}(?:\s*(?:{_OP_ALT})\s*{_TERM})+\s*=\s*{_TERM}")
_TOKEN = re.compile(rf"({_NUM})|({_OP_ALT})")

# Traces round aggressively (1/3 -> 0.333), so equality is tolerant. Relative tolerance
# handles large products; absolute handles rounded fractions near zero.
# Step k must retain at least this share of its original length. 0.7 targets the >30% shrink
# the manual audit measured, and is loose enough that dropping a redundant lead-in clause
# still passes -- the check is aimed at gutted steps, not at ordinary rewording.
MIN_STEP_LENGTH_RATIO = 0.7

_REL_TOL = 5e-3
_ABS_TOL = 5e-3


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


@dataclass(frozen=True)
class Claim:
    """One asserted numeric computation: ``operands`` joined by ``ops`` equals ``result``."""

    operands: tuple[float, ...]
    ops: tuple[str, ...]
    result: float

    @property
    def value(self) -> float:
        """Left-to-right evaluation with normal precedence (* and / before + and -)."""
        nums = [self.operands[0]]
        signs = []
        for op, rhs in zip(self.ops, self.operands[1:], strict=True):
            if op in ("*", "/"):
                if op == "/" and rhs == 0:
                    return math.nan
                nums[-1] = nums[-1] * rhs if op == "*" else nums[-1] / rhs
            else:
                signs.append(op)
                nums.append(rhs)
        total = nums[0]
        for sign, n in zip(signs, nums[1:], strict=True):
            total = total + n if sign == "+" else total - n
        return total

    @property
    def holds(self) -> bool:
        got = self.value
        if math.isnan(got):
            return True  # division by zero: skip rather than crash or judge
        return math.isclose(got, self.result, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def arithmetic_claims(text: str) -> list[Claim]:
    """Every purely numeric ``a op b [op c ...] = r`` claim in ``text``, left to right."""
    out = []
    for m in _CLAIM.finditer(text):
        nums, ops = [], []
        for tok in _TOKEN.finditer(m.group(0)):
            if tok.group(1) is not None:
                nums.append(_to_float(tok.group(1)))
            else:
                ops.append(_OPS[tok.group(2)])
        if len(nums) < 3 or len(ops) != len(nums) - 2:
            continue  # malformed for our purposes; stay conservative
        out.append(Claim(tuple(nums[:-1]), tuple(ops), nums[-1]))
    return out


def _numbers(text: str) -> set[float]:
    """Every standalone number in ``text``, for tracking whether a value was propagated."""
    return {_to_float(m.group(0)) for m in re.finditer(_TERM, text)}


def unpropagated_steps(
    original: list[str], injected: list[str], step_k: int
) -> list[tuple[int, float]]:
    """Downstream steps that still carry a value the fault replaced.

    Per-claim arithmetic is not enough: if step k changes ``10 + 5 = 15`` to ``= 16`` but step k+1
    still reads ``15 - 3 = 12``, each claim is self-consistent while the trace is not, and the fault
    shows up twice -- inflating detectability and breaking family comparability.
    """
    replaced = _numbers(original[step_k - 1]) - _numbers(injected[step_k - 1])
    stale = []
    for j in range(step_k, len(original)):
        # Checked whether or not the step was rewritten. Skipping rewritten steps made this
        # guard vacuous, because the generator rewrites every step from k onward -- prepending
        # "Now, " was enough to carry a stale pre-fault value through undetected.
        stale.extend((j + 1, v) for v in sorted(replaced & _numbers(injected[j])))
    return stale


def first_inconsistent_claim(steps: list[str]) -> tuple[int, Claim] | None:
    """The first ``(step_index, claim)`` whose arithmetic does not hold, or None."""
    for i, step in enumerate(steps):
        for c in arithmetic_claims(step):
            if not c.holds:
                return i, c
    return None


@dataclass(frozen=True)
class Validation:
    """Outcome of the blocking acceptance checks. ``ok`` only when ``reasons`` is empty."""

    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


# Defence in depth against the cue described in `inject.normalise_step`: if normalisation is
# ever bypassed, the blocking gate still catches it rather than letting a marked item through.
_STEP_PREFIX = re.compile(r"^\s*\**\s*step\s*\d+\s*\**\s*[:.]", re.IGNORECASE)


# `[\d,]+` alone matches a bare comma, which reached float("") and crashed on real traces.
# Anchor on a digit.
_ANSWER_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_BOXED = re.compile(rf"\\boxed\{{\s*({_ANSWER_NUM})\s*\}}")
_ANSWER_IS = re.compile(rf"(?:answer|total|result)\b[^0-9\-]{{0,20}}({_ANSWER_NUM})", re.I)


def final_answer(text: str) -> str | None:
    """The answer a final step asserts, or None if none can be read.

    Looks for an actual answer -- ``\\boxed{}``, then an "answer/total/result is N" phrase, then the
    last standalone number. Stripping non-digits from the whole step instead lets an injected trace
    whose answer is still ground truth validate clean.
    """
    for pattern in (_BOXED, _ANSWER_IS):
        found = [c for c in (_canonical(m) for m in pattern.findall(text)) if c is not None]
        if found:
            return found[-1]
    numbers = [c for c in (_canonical(m) for m in re.findall(_TERM, text)) if c is not None]
    return numbers[-1] if numbers else None


def _canonical(token: str) -> str | None:
    """Normalise a numeric token, or None if it is not actually a number."""
    try:
        value = float(token.replace(",", ""))
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else str(value)


def validate_injection(
    original: list[str],
    injected: list[str],
    step_k: int,
    original_answer: str,
) -> Validation:
    """Run every blocking check on one injected trace. ``step_k`` is 1-based.

    All must pass: step count unchanged (a different length is a rewrite); prefix byte-identical
    (else the fault is not localised where the label says); fault step actually changed; no
    downstream inconsistency; and the final answer moved (a trace still landing on ground truth
    makes a "correct" verdict arguably right).
    """
    if not 1 <= step_k <= len(original):
        raise ValueError(f"step_k={step_k} outside 1..{len(original)}")

    reasons: list[str] = []
    if len(injected) != len(original):
        reasons.append(f"step count {len(injected)} != {len(original)}")
        return Validation(reasons)  # later checks assume alignment

    if original[: step_k - 1] != injected[: step_k - 1]:
        reasons.append(f"prefix before step {step_k} was modified")
    if original[step_k - 1] == injected[step_k - 1]:
        reasons.append(f"step {step_k} unchanged, so no fault was injected")

    # A fault is a *substitution*, not a deletion. The manual audit found step k shrunk by
    # >30% in 10.2% of accepted traces, which is harmful twice over: a systematically shorter
    # step at k is a cue in the same class as the emphasis and step-prefix cues normalise_step
    # corrects, and deletion can change the problem instead of injecting a fault -- one audited
    # item dropped a whole case from a maximisation, so the answer moved because the candidate
    # set shrank, with every step still arithmetically sound.
    kept = len(" ".join(injected[step_k - 1].split()))
    was = len(" ".join(original[step_k - 1].split()))
    if was and kept < MIN_STEP_LENGTH_RATIO * was:
        reasons.append(
            f"step {step_k} is {kept / was:.0%} shorter than the original ({kept} vs {was} "
            f"chars): a fault must be substituted, not deleted"
        )

    for step_no, value in unpropagated_steps(original, injected, step_k):
        reasons.append(
            f"step {step_no} is unchanged but still uses pre-fault value {value:g}: "
            "inconsistent with the injected step, i.e. the fault was not propagated"
        )

    bad = first_inconsistent_claim(injected[step_k - 1 :])
    if bad is not None:
        offset, claim = bad
        if offset > 0:  # at step_k itself the wrong value IS the fault
            reasons.append(
                f"inconsistent arithmetic at step {step_k + offset}: {claim} -- "
                "fault was not propagated"
            )

    for j, step in enumerate(injected[step_k - 1 :], start=step_k):
        # A model writing `\times` inside a JSON string emits a legal `\t` escape, so the
        # parsed step contains TAB + "imes". Valid JSON, corrupted LaTeX -- and a cue present
        # only on injected items. Only flag characters the original did not already contain.
        # TAB and friends only. A bare newline is legitimate multi-line step text and is
        # handled by `inject.normalise_step`; treating it as corruption rejected 45% of valid
        # candidates. The signature this catches is a LaTeX command eaten by an escape layer.
        introduced = set(re.findall(r"[\t\r\f\v]", step)) - set(
            re.findall(r"[\t\r\f\v]", original[j - 1])
        )
        if introduced:
            reasons.append(
                f"step {j} introduced control character(s) {sorted(map(repr, introduced))}: "
                "almost certainly a LaTeX command mangled by JSON escaping"
            )
        if _STEP_PREFIX.match(step) and not _STEP_PREFIX.match(original[j - 1]):
            reasons.append(
                f"step {j} gained an echoed step prefix absent from the original: a formatting "
                "cue that would mark every injected item"
            )

    injected_answer, truth = final_answer(injected[-1]), final_answer(original_answer)
    if injected_answer is None or truth is None:
        reasons.append("final answer unreadable in the injected trace or the ground truth")
    elif injected_answer == truth:
        reasons.append(f"final answer still matches ground truth {truth!r}")

    return Validation(reasons)


# Ordered, first match wins, so specific keys must precede general ones: "echoed step prefix"
# contains "prefix", and both the unpropagated-value and inconsistent-arithmetic reasons end
# "fault was not propagated" while the first also contains "unchanged". Kept beside
# `validate_injection` because it classifies that function's strings -- two CLI copies drifted, and
# the older bucketed three acceptance checks as "other", under-reporting the cues they exist to
# catch.
REJECTION_CLASSES = ("echoed step prefix", "inconsistent arithmetic", "control character",
                     "not propagated", "final answer", "step count", "shorter",
                     "prefix", "unchanged")


def reason_class(reason: str) -> str:
    """Collapse a rejection reason from `validate_injection` to a countable class."""
    for key in REJECTION_CLASSES:
        if key in reason:
            return key
    return "other"


def injectable_steps(steps: list[str]) -> list[int]:
    """1-based indices of steps carrying a numeric claim we can perturb.

    Only **step 1** is excluded, and the ends are asymmetric on purpose: step 1 has no untouched
    prefix to anchor the fault's location, so perturbing it is a whole-trace rewrite. The last step
    is kept -- zero downstream steps is the cheapest repair, an informative end of the cost range.
    """
    return [i for i in range(2, len(steps) + 1) if arithmetic_claims(steps[i - 1])]


def is_eligible_source(steps: list[str], min_steps: int = 4) -> bool:
    """May this clean trace be used as an injection source?

    Requires enough steps, one injectable mid-trace claim, and -- critically -- that the extractor
    already reads the *unmodified* trace as consistent. It mis-reads multi-``=`` chains and symbolic
    equations on ~5% of clean seeds; excluding those makes any inconsistency in an injected trace
    attributable to the injection rather than to a known blind spot.
    """
    return (
        len(steps) >= min_steps
        and bool(injectable_steps(steps))
        and first_inconsistent_claim(steps) is None
    )
