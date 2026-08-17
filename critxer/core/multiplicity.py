"""Holm-Bonferroni step-down over an explicitly declared family.

A paper reporting forty bootstrap p-values and starring those below 0.05 will star two by chance.
The correction is not the interesting part; **declaring the family** is. A family chosen after
seeing which members are small is not a correction, so each caller names its family in code next to
the contrasts that belong to it.

`critxer.core.power.holm_rejects_any` answers a different question -- whether a simulated family
rejects *anything*, for power calculations. This says which members survive.
"""

from __future__ import annotations

from collections.abc import Sequence

ALPHA = 0.05


def holm[K](pvals: dict[K, float], alpha: float = ALPHA) -> dict[K, dict]:
    """Which hypotheses in a family reject, and the threshold each was tested against.

    Sorts ascending, tests rank *i* of *k* against ``alpha / (k - i)``, and **stops at the first
    failure**: once one fails its threshold, no later one rejects however small it is. That is the
    whole of "step-down" and the easiest part to get wrong by testing each rank independently.
    Returns every member with its threshold, so a table can show what a borderline result faced.
    """
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    k = len(ordered)
    out: dict[K, dict] = {}
    still_rejecting = True
    for rank, (key, p) in enumerate(ordered):
        threshold = alpha / (k - rank)
        still_rejecting = still_rejecting and p <= threshold
        out[key] = {"p": p, "threshold": threshold, "reject": still_rejecting}
    return out


def short_family_message(
    family: str,
    *,
    got: int,
    expected: int,
    present: Sequence[str],
    alpha: float = ALPHA,
) -> str | None:
    """``None`` if the family is exactly as declared, else the warning the caller must print.

    The error is not symmetric: too few members raises every threshold (``alpha/k`` with a smaller
    *k*), so survivors clear an easier bar than the paper describes. Both wording sweeps hit this,
    which is why it is a tested function rather than a comment. Over-population is reported too --
    it cannot flatter a survivor, but the declaration and the code disagreeing is not for this
    function to resolve silently.
    """
    if got == expected:
        return None
    direction = ("MORE lenient" if got < expected else "stricter")
    return (
        f"WARNING: {family} family has {got} members, not the declared {expected}. "
        f"Present: {sorted(set(present))}. Its survivors are being tested at "
        f"{alpha}/{got} rather than {alpha}/{expected}, which is {direction} than declared "
        f"-- reconcile the declaration with the artefacts before claiming a survivor."
    )
