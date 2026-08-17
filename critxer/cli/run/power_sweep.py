#!/usr/bin/env python
"""Task 1: how many items and samples per item does the pilot need?

Sweeps a 3pp MDE against a cheaper 5pp alternative across plausible baseline false-alarm rates and
both effect mechanisms, reporting the fewest items reaching 80% power, and re-checks null
calibration at each samples-per-item. A sensitivity table rather than a single number, since
baseline FAR and the propensity spread are unknown until Task 0 measures them. Markdown to stdout.

**The concentration this sweeps at was later falsified** -- it implies an N1 that would fail this
project's own gate 1. Kept because the sizes actually used were chosen from this table, so a rerun
at a fitted concentration would not be the record of that decision. Use
`core.power.fit_concentration` before powering anything new; the fitted value depends on the
baseline FAR, so it is not one constant across the rates swept below.
"""

from __future__ import annotations

import numpy as np

from critxer.core.power import min_items_for_power, power_fast

DELTAS = [0.03, 0.05]
SAMPLES = [4, 8, 16]
BASELINES = [0.05, 0.15, 0.30]
MECHANISMS = ["logit", "additive"]
CONCENTRATION = 8.0
REPS = 1000
SEED = 20260804


def main() -> None:
    print("# Task 1 — power simulation\n")
    print(f"Beta concentration {CONCENTRATION}, {REPS} reps/cell, alpha=0.05, target power 0.80.")
    print("NOTE: concentration 8.0 implies N1 ~0.31-0.36; Task 0 measured 0.086 and 0.012, so\n"
          "this generative model was refuted after the fact. Retained as the record of the\n"
          "sizing decision -- use core.power.fit_concentration for anything new.\n")
    print("Cells are the fewest items reaching 80% power; `--` means >20,000 (infeasible).\n")

    print("## Null calibration (delta = 0, should sit near 0.05)\n")
    print("| samples/item | items | false-positive rate |")
    print("|---|---|---|")
    for n_samples in SAMPLES:
        for n_items in (400, 1200):
            fpr = power_fast(
                np.random.default_rng(SEED), n_items, n_samples, 0.15,
                CONCENTRATION, 0.0, "logit", reps=4000,
            )
            print(f"| {n_samples} | {n_items} | {fpr:.3f} |")

    for delta in DELTAS:
        print(f"\n## MDE = {delta * 100:.0f}pp — minimum items for 80% power\n")
        header = "| baseline FAR | mechanism | " + " | ".join(f"n={s}" for s in SAMPLES) + " |"
        print(header)
        print("|---" * (2 + len(SAMPLES)) + "|")
        for baseline in BASELINES:
            for mech in MECHANISMS:
                cells = []
                for n_samples in SAMPLES:
                    n = min_items_for_power(
                        np.random.default_rng(SEED), n_samples=n_samples,
                        baseline_far=baseline, concentration=CONCENTRATION,
                        delta=delta, mechanism=mech, reps=REPS,
                    )
                    cells.append("--" if n is None else str(n))
                print(f"| {baseline:.2f} | {mech} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
