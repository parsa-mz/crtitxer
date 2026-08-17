"""Canonical, deterministic item allocation across the pilot's arms.

The single source of truth: given a pool and the arm sizes, it returns the same disjoint,
source-stratified assignment every time.

Disjointness is not bookkeeping. The R4 2x2 pairs each target with a warmup episode the model
already audited *and repaired*; a warmup item that is also a clean-arm target has previewed that
target, and no amount of sampling fixes the contamination.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def _stratified_take(available: dict[str, list[str]], n: int) -> list[str]:
    """``n`` ids spread as evenly as the remaining per-source counts allow.

    Largest-remainder rather than a fixed quota: sources are unequal in size, so a fixed quota would
    either under-fill the arm or exhaust a small source. Deficits roll over to sources with items.
    """
    taken: list[str] = []
    while len(taken) < n:
        live = [s for s in sorted(available) if available[s]]
        if not live:
            break
        # Round-robin one at a time. O(n * sources) and n is in the hundreds; clarity wins.
        for source in live:
            if len(taken) == n:
                break
            taken.append(available[source].pop())
    return taken


def allocate(
    pool: Sequence[Mapping[str, str]],
    sizes: Mapping[str, int],
    seed: int,
    restrict: Mapping[str, Iterable[str]] | None = None,
    key: str = "split",
) -> dict[str, list[str]]:
    """Partition ``pool`` into disjoint, stratified arms of the requested ``sizes``.

    ``restrict`` limits an arm to a subset of ids, used for the injection-source arm, whose
    eligibility rule is stricter than the other arms'.
    Restricted arms fill **first**: otherwise an unrestricted arm consumes seeds only the restricted
    arm can use, which is how the injected arm once ended up short.

    Raises rather than returning a short arm -- the power figures are quoted against exact sizes,
    so a silent shortfall would make the analysis quietly wrong instead of loudly broken.
    """
    total = sum(sizes.values())
    if total > len(pool):
        raise ValueError(f"requested {total} items from a pool of {len(pool)}")

    restrict = {k: set(v) for k, v in (restrict or {}).items()}
    rng = np.random.default_rng(seed)

    # Sort by id so the permutation, not dict or dataset order, is the only source of randomness.
    by_source: dict[str, list[str]] = {}
    for row in sorted(pool, key=lambda r: r["id"]):
        by_source.setdefault(row[key], []).append(row["id"])
    for ids in by_source.values():
        rng.shuffle(ids)

    arms: dict[str, list[str]] = {}
    # Restricted arms first, most-constrained first among them.
    order = sorted(sizes, key=lambda a: (a not in restrict, len(restrict.get(a, ()))))
    for arm in order:
        if arm in restrict:
            eligible = {s: [i for i in ids if i in restrict[arm]] for s, ids in by_source.items()}
            n_eligible = sum(len(v) for v in eligible.values())
            if n_eligible < sizes[arm]:
                raise ValueError(
                    f"arm {arm!r} wants {sizes[arm]} items but only {n_eligible} eligible remain"
                )
            picked = _stratified_take(eligible, sizes[arm])
            chosen = set(picked)
            for ids in by_source.values():
                ids[:] = [i for i in ids if i not in chosen]
        else:
            picked = _stratified_take(by_source, sizes[arm])
        if len(picked) < sizes[arm]:
            raise ValueError(f"arm {arm!r} wants {sizes[arm]} items but only {len(picked)} remain")
        arms[arm] = sorted(picked)
    return arms
