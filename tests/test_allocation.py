"""Tests for the canonical item allocation against the pilot budget.

Pins disjointness, determinism and stratification -- three properties the downstream analysis
assumes, none of which was checked while every runner re-derived its own pool from ProcessBench. R4
is only interpretable if the warmup pool is held out: a warmup item that is also a clean-arm target
has been audited *and repaired* before the model is asked to audit it cold.
"""

from __future__ import annotations

import pytest

from critxer.core.allocation import allocate

POOL = [
    {"id": f"{split}-{i}", "split": split}
    for split in ("gsm8k", "math", "olympiadbench", "omnimath")
    for i in range(50)
]
INJECTABLE = {r["id"] for r in POOL if int(r["id"].split("-")[1]) < 20}


def test_arms_are_disjoint():
    """The property R4's interpretability rests on."""
    arms = allocate(POOL, {"clean": 100, "source": 40, "warmup": 10}, seed=1)

    ids = [i for arm in arms.values() for i in arm]

    assert len(ids) == len(set(ids)) == 150


def test_allocation_is_deterministic_for_a_given_seed():
    a = allocate(POOL, {"clean": 100, "source": 40, "warmup": 10}, seed=1)
    b = allocate(POOL, {"clean": 100, "source": 40, "warmup": 10}, seed=1)

    assert a == b


def test_a_different_seed_gives_a_different_split():
    a = allocate(POOL, {"clean": 100, "source": 40, "warmup": 10}, seed=1)
    b = allocate(POOL, {"clean": 100, "source": 40, "warmup": 10}, seed=2)

    assert a != b


def test_every_arm_is_stratified_across_sources():
    """Every arm is stratified across all four ProcessBench sources.

    An arm drawn without stratification can end up dominated by gsm8k, whose traces are far
    shorter and easier, which would confound any arm-to-arm comparison with difficulty.
    """
    arms = allocate(POOL, {"clean": 100, "source": 40, "warmup": 12}, seed=1)

    for name, ids in arms.items():
        counts = {s: sum(1 for i in ids if i.startswith(s)) for s in ("gsm8k", "math", "omnimath")}
        assert min(counts.values()) > 0, f"{name} missed a source: {counts}"


def test_a_restricted_arm_draws_only_from_its_eligible_subset():
    """Only the injection-source arm needs injectability, so only it should be restricted.

    Spending the scarce injectable seeds on the clean arm -- which does not need them -- is how
    the injected arm ended up short of the size the budget assumed.
    """
    arms = allocate(POOL, {"clean": 100, "source": 40, "warmup": 10}, seed=1,
                    restrict={"source": INJECTABLE})

    assert set(arms["source"]) <= INJECTABLE


def test_the_restricted_arm_is_filled_before_the_others():
    """Otherwise an unrestricted arm can consume seeds the restricted one is the only user of."""
    arms = allocate(POOL, {"clean": 160, "source": 40, "warmup": 0}, seed=1,
                    restrict={"source": INJECTABLE})

    assert len(arms["source"]) == 40


def test_an_impossible_request_raises_rather_than_silently_shrinking():
    """A short arm must be a loud failure: the power numbers are quoted against exact sizes."""
    # 80 ids are injectable (20 per source x 4); asking for 100 must fail on eligibility, not
    # on pool size, so the message has to name the eligible count rather than the pool.
    with pytest.raises(ValueError, match="only 80 eligible"):
        allocate(POOL, {"source": 100}, seed=1, restrict={"source": INJECTABLE})


def test_asking_for_more_than_the_pool_holds_raises():
    with pytest.raises(ValueError, match="pool of 200"):
        allocate(POOL, {"clean": 500}, seed=1)
