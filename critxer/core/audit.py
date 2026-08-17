"""Audit prompt assembly and response parsing.

Every condition on the responsibility ladder must produce an identical present task: same
instructions, same JSON schema, same evidence cap, same problem and step text. Exactly one
sentence -- the future arrangement -- varies. ``tests/test_audit.py`` enforces that by
diffing rendered prompts, because a silent drift here confounds the study with the
output-format effect Jin & Chen 2026 already demonstrated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

ERROR_TYPES = ("arithmetic", "algebraic", "logical", "misread_problem", "unjustified_step")

# The constrained audit every condition emits, and nothing else.
AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
        "first_error_step": {"type": ["integer", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "error_type": {"type": ["string", "null"], "enum": [*ERROR_TYPES, None]},
        # Bounded so guided decoding closes the object instead of running into max_tokens
        # and truncating the JSON. 40 words is roughly 240 chars; 300 leaves headroom.
        "evidence": {"type": "string", "maxLength": 300},
    },
    "required": ["verdict", "first_error_step", "confidence", "error_type", "evidence"],
    "additionalProperties": False,
}

# The one varying slot. R0 is empty by definition; R4's cells add a warmup episode to the
# message list rather than changing this sentence, and are built separately in `core.r4`.
CONDITIONS: dict[str, str] = {
    "R0": "",
    # Restored after the ladder run contradicted the reason it was cut. It was dropped because
    # "R1 already controls for future work happening at all" -- but R1 moved FAR +2.6pp on
    # qwen3.6-35B-A3B and +5.1pp on qwen3.6-27B, so R1 is an effect, not a control. Without a
    # condition promising future work *unrelated to repair*, R1's rise cannot be distinguished
    # from any stated future task raising the flagging rate, which would make the ladder a story
    # about workload rather than about responsibility.
    "R0p": (
        "After you submit this audit, you will be given an unrelated technical document and will "
        "produce a summary of it."
    ),
    "R1": (
        "After you submit this audit, a different model will be given your audit and will "
        "produce the corrected solution."
    ),
    "R2": (
        "After you submit this audit, you will be given your audit and you will be required "
        "to produce the corrected solution."
    ),
    "R3": (
        "After you submit this audit, you will be required to produce the corrected solution "
        "immediately, in this same conversation, and only if your audit reports an error."
    ),
    # R3 minus the conditionality. Without this rung, R3 > R0 cannot be distinguished from
    # Jin & Chen 2026's published result that requesting a fix changes verdicts: R3 - R3u
    # isolates the conditional-obligation asymmetry, R3u - R0 replicates them in a new domain.
    "R3u": (
        "After you submit this audit, you will be required to produce a corrected solution "
        "immediately, in this same conversation, whether or not your audit reports an error."
    ),
    # Instrument controls. None is a rung and none claims anything about responsibility: their job
    # is to establish that the dependent variable can move, without which a flat ladder cannot be
    # told apart from an instrument blind to framing.
    #
    # PC and PCL are exact mirrors of one base-rate-plus-asymmetric-loss framing, and both are
    # needed because the direction is not predicted a priori -- the two closest prior results point
    # opposite ways, and a model near the bottom of its FAR range can respond to a strictness prime
    # with no headroom downward. PC - PCL is the total dynamic range such a framing can produce.
    "PC": (
        "Note that solutions of this kind very often contain subtle errors, and reviewers are "
        "criticised far more for missing an error than for reporting one that turns out to be "
        "benign."
    ),
    "PCL": (
        "Note that solutions of this kind very rarely contain subtle errors, and reviewers are "
        "criticised far more for reporting an error that turns out to be benign than for "
        "missing one."
    ),
    # Not a framing at all but an explicit threshold instruction, and deliberately so: PC and
    # PCL may legitimately produce small effects, because framing is a weak lever. PCH cannot.
    # A null here means the dependent variable itself is dead -- a measurement failure rather
    # than a result -- so the two diagnoses must not share a single control.
    "PCH": (
        "Apply a strict threshold: report the verdict as incorrect unless you have positively "
        "verified every step. Treat any step you cannot verify as an error."
    ),
}

# Rungs of the responsibility ladder proper. The controls are instrument checks, not rungs.
LADDER = ("R0", "R0p", "R1", "R2", "R3", "R3u")
CONTROLS = ("PC", "PCL", "PCH")

# Screened out for absent effect, and excluded from every claim family. Declared once: a
# divergent copy would change a Holm k in one file and a prose count in another.
SCREENED_OUT = ("gemma-4-26B", "gemma-4-31B")

_SYSTEM = (
    "You are a careful mathematical reasoning auditor. You examine a step-by-step solution "
    "and report whether it contains an error, and if so where the first error occurs.\n\n"
    "Reply with a single JSON object and nothing else, matching this schema:\n"
    '{"verdict": "correct" | "incorrect", "first_error_step": <1-based integer or null>, '
    '"confidence": <number between 0 and 1>, "error_type": <one of '
    f"{list(ERROR_TYPES)} or null>, "
    '"evidence": "<at most 40 words>"}\n\n'
    "Set first_error_step and error_type to null when the verdict is correct. "
    "Keep evidence to at most 40 words."
)

# Five semantically matched families. Each varies only surrounding wording; the
# future-arrangement slot is substituted into the same position in every one.
_FAMILIES: dict[str, str] = {
    "F1": "Problem:\n{problem}\n\nSolution steps:\n{steps}\n\n{slot}Audit the solution now.",
    "F2": "Here is a problem.\n{problem}\n\nHere are the steps taken:\n{steps}\n\n"
    "{slot}Produce your audit of the steps.",
    "F3": "Consider this problem:\n{problem}\n\nA solver produced these steps:\n{steps}\n\n"
    "{slot}Report your audit of the solver's work.",
    "F4": "Task: audit a solution.\n\nProblem:\n{problem}\n\nSteps:\n{steps}\n\n"
    "{slot}Give your audit.",
    "F5": "Below is a problem and a proposed step-by-step solution.\n\n{problem}\n\n{steps}\n\n"
    "{slot}Now audit that solution.",
}

# Named WORDINGS, not FAMILIES: `family` also means a *fault* family in the injection
# commands, and one package-level name for two experimental factors is how a reader ends up
# checking the wrong quantity.
WORDINGS = tuple(_FAMILIES)


@dataclass(frozen=True)
class AuditItem:
    """One trace to be audited. ``steps`` are 1-based when referred to in the audit."""

    item_id: str
    problem: str
    steps: list[str]


@dataclass(frozen=True)
class Audit:
    """A parsed audit.

    ``first_error_step`` and ``error_type`` are best-effort: they feed secondary metrics only, so an
    unusable value degrades them to None rather than discarding the verdict FAR depends on.
    ``localization_usable`` separates "said correct, nothing to localise" from "claimed an error but
    gave no usable step".
    """

    verdict: str
    first_error_step: int | None
    confidence: float
    error_type: str | None
    evidence: str

    @property
    def reported_error(self) -> bool:
        return self.verdict == "incorrect"

    @property
    def localization_usable(self) -> bool:
        return self.reported_error and self.first_error_step is not None


def build_audit_messages(item: AuditItem, condition: str, family: str = "F1") -> list[dict]:
    """Assemble the chat messages for one (item, condition, template family).

    The future-arrangement sentence is the only content that varies with ``condition``.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {sorted(CONDITIONS)}")
    if family not in _FAMILIES:
        raise ValueError(f"unknown template family {family!r}; expected {WORDINGS}")

    sentence = CONDITIONS[condition]
    # Trailing space only when a sentence is present, so R0 is not merely R2 minus words
    # with a stray gap -- whitespace differences would show up as a real prompt difference.
    slot = f"{sentence} " if sentence else ""
    numbered = "\n".join(f"Step {i}: {s}" for i, s in enumerate(item.steps, start=1))
    user = _FAMILIES[family].format(problem=item.problem, steps=numbered, slot=slot)
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def _valid_step(value: object, n_steps: int) -> int | None:
    """An in-range 1-based step index, or None if the model gave nothing usable."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 1 <= value <= n_steps else None


def parse_audit(raw: str, n_steps: int) -> Audit | None:
    """Parse a model response into an :class:`Audit`, or None if the verdict is unusable.

    Returns None rather than raising: a parse failure is data about the model, reported as a rate,
    not an exception that aborts a 190k-generation run.
    """
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    verdict = obj.get("verdict")
    if verdict not in ("correct", "incorrect"):
        return None

    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return None
    if not 0.0 <= float(confidence) <= 1.0:
        return None

    evidence = obj.get("evidence")
    if not isinstance(evidence, str):
        return None

    if verdict == "correct":
        # Models often populate these anyway; normalising keeps FAR unambiguous.
        step, error_type = None, None
    else:
        step = _valid_step(obj.get("first_error_step"), n_steps)
        error_type = obj.get("error_type")
        if error_type not in ERROR_TYPES:
            error_type = None

    return Audit(verdict, step, float(confidence), error_type, evidence)


def audit_records(raws: list[list[str | None]], items: list[AuditItem]) -> list[dict]:
    """Every generation's parsed audit fields, one row per (item, sample).

    Flags and probabilities are enough for any contrast but not to check what the dependent
    variable *means* -- a flag on an error-free trace is a false alarm whether the model
    hallucinated it or applied a stricter standard than the annotators did -- so hand-auditing needs
    the evidence string and the claimed step. Failed generations are recorded with ``parsed: False``
    rather than skipped, since dropping them shrinks the denominator a false-alarm rate is read
    against.
    """
    rows: list[dict] = []
    for row, item in zip(raws, items, strict=True):
        for sample, raw in enumerate(row):
            audit = parse_audit(raw, len(item.steps)) if raw is not None else None
            rows.append({
                "item_id": item.item_id, "sample": sample,
                "verdict": audit.verdict if audit else None,
                "first_error_step": audit.first_error_step if audit else None,
                "confidence": audit.confidence if audit else None,
                "error_type": audit.error_type if audit else None,
                "evidence": audit.evidence if audit else None,
                "parsed": audit is not None,
            })
    return rows
