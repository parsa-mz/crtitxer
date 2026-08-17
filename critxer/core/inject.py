"""Fault injection generation for repairgym.

Generation is model-assisted; **acceptance is not**. Everything passes
`repairgym.validate_injection` and anything failing is dropped rather than patched. Two structural
choices remove whole classes of avoidable rejection:

* Only steps ``k..n`` are requested and the untouched prefix is spliced here, so prefix integrity
  holds **by construction** rather than by asking politely and checking afterwards.
* The suffix comes back as sentinel-delimited plain text, not JSON, because LaTeX does not survive
  JSON string escaping (see `parse_suffix`). Step count is therefore checked, not schema-pinned.

That leaves the validator the part that matters: whether the fault is real, propagated consistently,
and consequential for the final answer.
"""

from __future__ import annotations

import re

from critxer.core import backend
from critxer.core.repairgym import arithmetic_claims, validate_injection

SENTINEL = "---STEP---"


FAMILY_BRIEFS: dict[str, str] = {
    "local": (
        "Change exactly one numeric VALUE in the target step so the arithmetic result is wrong, "
        "while keeping the operation and the reasoning identical. The step should still describe "
        "the same computation -- only the number it arrives at changes."
    ),
    # Revised after gate-2 calibration measured detection at 0.94-0.97 for this family --
    # ceilinged, with no headroom for a framing effect to move it. The first brief invited
    # operator swaps (`R = 4B` -> `R = 4 + B`), which are glaring once the problem statement is
    # in context. A wrong method has to be PLAUSIBLE to be a useful test item.
    "structural": (
        "Change the METHOD used in the target step so the approach is wrong but still looks "
        "reasonable. Prefer: applying a correct-looking formula that does not fit this problem; "
        "operating on the wrong pair of quantities; including or omitting a term that plausibly "
        "belongs; or using a related quantity in place of the right one (a rate for a total, a "
        "part for a whole, a boundary case counted the wrong way).\n"
        "Do NOT simply swap one arithmetic operator for another, and do not choose a method whose "
        "wrongness is obvious from the problem statement alone. A careful reader should have to "
        "follow the reasoning to see the mistake. Write it with the same confidence as the "
        "original -- no hedging, no hint that anything is off."
    ),
}

_SYSTEM = (
    "You construct evaluation data for a reasoning-auditing benchmark by introducing a single, "
    "deliberate flaw into an otherwise correct mathematical solution.\n\n"
    "Hard requirements:\n"
    "1. Introduce the flaw in the TARGET STEP only.\n"
    "2. PROPAGATE the consequences consistently through every later step, so that every later "
    "step follows correctly from the flawed value or method. After the target step there must be "
    "no second inconsistency and no leftover value from the original solution -- a reader should "
    "find exactly one thing wrong, at the target step.\n"
    "3. The final answer must change as a result.\n"
    "4. Preserve the writing style, formatting and level of detail of the original. Do not add "
    "hedging, do not flag the error, and do not add or remove steps.\n"
    "5. Return the rewritten steps only, from the target step to the last step.\n"
    f"6. Separate consecutive steps with a line containing exactly {SENTINEL} and nothing "
    "else. Output no JSON, no numbering and no commentary -- just the step text and the "
    "separators. Reproduce mathematical notation exactly as in the original, backslashes "
    "included."
)


def parse_suffix(raw: str, n_expected: int) -> list[str] | None:
    """Split a sentinel-delimited suffix into exactly ``n_expected`` steps, or None.

    Plain text rather than JSON because LaTeX does not survive JSON string escaping: a model
    writing ``\times`` emits a legal ``\t`` escape and the step arrives as TAB + "imes". That is
    valid JSON, so nothing objects -- it corrupted half the accepted injections in the first pilot.
    The cost is that step count is no longer pinned by a schema, so it is checked here and a
    mismatch is retried.
    """
    parts = [p.strip() for p in raw.split(SENTINEL)]
    if len(parts) != n_expected or any(not p for p in parts):
        return None
    return parts


def suffix_schema(n_suffix: int) -> dict:
    """Schema pinning the rewritten suffix to exactly ``n_suffix`` steps.

    Superseded and unused -- guided decoding against it corrupted the LaTeX (`parse_suffix`) -- but
    kept, because the first pilot's accepted injections were generated under it.
    """
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n_suffix,
                "maxItems": n_suffix,
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }


def build_injection_messages(steps: list[str], step_k: int, family: str) -> list[dict]:
    """Chat messages asking for steps ``step_k..n`` rewritten with a ``family`` fault."""
    if family not in FAMILY_BRIEFS:
        raise ValueError(f"unknown family {family!r}; expected {sorted(FAMILY_BRIEFS)}")
    # The last step is allowed: zero downstream steps is the cheapest repair and a real point
    # on the continuous repair-cost range. Step 1 is not -- no untouched prefix to anchor on.
    if not 2 <= step_k <= len(steps):
        raise ValueError(f"step_k={step_k} must be in 2..{len(steps)}")

    n_suffix = len(steps) - step_k + 1
    numbered = "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, start=1))
    claims = arithmetic_claims(steps[step_k - 1])
    hint = (
        f"\nThe target step asserts: {claims[0].operands} joined by {claims[0].ops} "
        f"equals {claims[0].result:g}.\n"
        if claims
        else "\n"
    )
    user = (
        f"Correct solution:\n{numbered}\n\n"
        f"TARGET STEP: {step_k}\n{hint}\n"
        f"{FAMILY_BRIEFS[family]}\n\n"
        f"Return exactly {n_suffix} steps: the rewritten Step {step_k} through Step {len(steps)}, "
        f"separated by {SENTINEL}, propagating the consequences so no later step is inconsistent."
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


# Anchored to the start of the step, so a genuine mid-sentence reference like
# "From Step 3 we know..." is left alone.
# Trailing `\**` because the punctuation can sit inside the emphasis (`**Step 7:**`).
_ECHOED_PREFIX = re.compile(r"^\s*\**\s*step\s*\d+\s*\**\s*[:.]\s*\**\s*", re.IGNORECASE)
# Markdown emphasis around a span of text. Non-greedy and single-line so it cannot swallow a
# paragraph, and `_` is required to sit on a word boundary -- LaTeX is full of `x_1`, `a_{ij}`.
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC_STAR = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_ITALIC_UNDER = re.compile(r"(?<![\w\\])_(?!_)([^_\n]+?)_(?![\w])")


def strip_emphasis(text: str) -> str:
    """Unwrap markdown emphasis, keeping the emphasised text."""
    for pattern in (_BOLD, _ITALIC_STAR, _ITALIC_UNDER):
        text = pattern.sub(r"\1", text)
    return text


def normalise_step(text: str, original: str) -> str:
    """Remove formatting that would mark an injected step, given its own original.

    Three observed cues: an echoed step number (the prompt numbers steps itself, so it yields
    "Step 7: Step 7: ..."); line structure, matched to the original rather than imposed; and
    **markdown emphasis**, in 16.3% of injected traces and none of their originals. Each is
    corrected *relative to the original*, never to a house style -- unconditional stripping would
    make the absence of emphasis the cue on source traces that use it.
    """
    out = _ECHOED_PREFIX.sub("", text).strip()
    if not any(m.search(original) for m in (_BOLD, _ITALIC_STAR, _ITALIC_UNDER)):
        out = strip_emphasis(out)
    if "\n" not in original:
        out = re.sub(r"\s+", " ", out)
    return out


def splice(steps: list[str], step_k: int, suffix: list[str]) -> list[str]:
    """Untouched prefix plus a normalised, model-written suffix.

    Prefix integrity holds by construction; ``normalise_step`` removes the formatting drift that
    would otherwise mark every injected item.
    """
    expected = len(steps) - step_k + 1
    if len(suffix) != expected:
        raise ValueError(f"suffix length {len(suffix)} != expected {expected}")
    return [
        *steps[: step_k - 1],
        *(normalise_step(t, o) for t, o in zip(suffix, steps[step_k - 1 :], strict=True)),
    ]


async def inject_at(http, ep: backend.Endpoint, steps: list[str], step_k: int, answer: str,
                    attempts: int) -> tuple[list[str] | None, list[str]]:
    """One accepted `local` fault at ``step_k``, plus every rejection reason seen trying.

    Reasons are returned rather than logged because the acceptance audit's rejection breakdown is
    built from them.
    Acceptance is `validate_injection`, never the model's own say-so, and a rejected attempt is
    retried rather than patched.
    """
    seen: list[str] = []
    for _ in range(attempts):
        raws = await backend.sample(
            http, ep, build_injection_messages(steps, step_k, "local"),
            n=1, temperature=0.8, max_tokens=2048)
        if not raws or raws[0] is None:
            seen.append("no generation")
            continue
        suffix = parse_suffix(raws[0], len(steps) - step_k + 1)
        if suffix is None:
            seen.append("step count mismatch in suffix")
            continue
        injected = splice(steps, step_k, suffix)
        v = validate_injection(steps, injected, step_k, answer)
        if v.ok:
            return injected, seen
        seen.extend(v.reasons)
    return None, seen
