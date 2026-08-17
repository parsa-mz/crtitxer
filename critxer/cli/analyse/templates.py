#!/usr/bin/env python
"""Template robustness of the main claim.

The paper's positive ladder result is R1 − R0p: promising a *repair* raises the false-alarm rate
relative to promising an unrelated future task of comparable weight. Measured on F1 it is +4.7pp and
+5.2pp. A condition effect has to exceed the template random-effect SD, and the SD
measured in `kill_tests.json` (1.40pp / 1.05pp) was the SD of **R0's level**, not of an *effect*.
Those are different quantities: a template can shift both conditions together, leaving the contrast
untouched.

So this recomputes R1 − R0p within each template family and reports the SD of the effect across
families. The claim survives if that SD is small next to the effect; it does not if the effect is
partly a property of one wording.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from critxer.core.paths import artefact, run_dir
from critxer.core.resample import interval_from_reps

ALPHA = 0.05
N_BOOT = 20000
SEED = 20260805


def boot(diff: np.ndarray, rng: np.random.Generator) -> dict:
    """Per-family interval, sharing the project's one interval-and-p implementation.

    One implementation across the project, so a figure drawn from here and a table drawn from
    elsewhere cannot disagree about the same contrast.
    """
    d = diff[~np.isnan(diff)]
    reps = d[rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)
    return {"n": int(d.size), **interval_from_reps(reps, float(d.mean()), ALPHA, N_BOOT)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(run_dir("ladder")))
    ap.add_argument("--high", default="R1", help="condition expected to be higher")
    ap.add_argument("--low", default="R0p", help="matched comparison condition")
    ap.add_argument("--out", default=str(artefact("template_robustness.json")))
    args = ap.parse_args()

    # (model, family) -> {condition: record}
    runs: dict[tuple[str, str], dict[str, dict]] = {}
    for path in sorted(Path(args.dir).glob("*.json")):
        rec = json.loads(path.read_text())
        runs.setdefault((rec["auditor"], rec["family"]), {})[rec["condition"]] = rec

    rng = np.random.default_rng(SEED)
    by_model: dict[str, dict[str, dict]] = {}
    for (model, family), by_cond in sorted(runs.items()):
        if args.high not in by_cond or args.low not in by_cond:
            continue
        hi, lo = by_cond[args.high], by_cond[args.low]
        if hi["item_ids"] != lo["item_ids"]:
            raise SystemExit(f"{model}/{family}: item order differs; the contrast would be void")
        d = boot(np.array(hi["per_item_probs"]) - np.array(lo["per_item_probs"]), rng)
        d["far_high"], d["far_low"] = hi["far"], lo["far"]
        by_model.setdefault(model, {})[family] = d

    out = {}
    label = f"{args.high} - {args.low}"
    print(f"| model | family | FAR {args.low} | FAR {args.high} | {label} | 95% CI |")
    print("|---|---|---|---|---|---|")
    for model, fams in by_model.items():
        ordered = sorted(fams.items())
        for family, d in ordered:
            print(f"| {model} | {family} | {d['far_low']:.4f} | {d['far_high']:.4f} "
                  f"| {d['effect']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] |")
        effects = np.array([d["effect"] for _, d in ordered])
        levels = np.array([d["far_low"] for _, d in ordered])
        row = {
            "families": sorted(fams),
            "per_family": fams,
            "effect_mean": float(effects.mean()),
            # ddof=1: these families are a sample of possible wordings, not the population.
            "effect_sd": float(effects.std(ddof=1)) if effects.size > 1 else float("nan"),
            "level_sd": float(levels.std(ddof=1)) if levels.size > 1 else float("nan"),
            "all_same_sign": bool(np.all(effects > 0) or np.all(effects < 0)),
        }
        out[model] = row

    print()
    for model, row in out.items():
        n = len(row["families"])
        ratio = abs(row["effect_mean"]) / row["effect_sd"] if row["effect_sd"] else float("inf")
        print(f"{model}: {n} families, mean effect {row['effect_mean']:+.4f}, "
              f"SD across families {row['effect_sd']:.4f} (ratio {ratio:.1f}x)")
        if n < 2:
            print(f"  SD of the {args.low} level: not defined on one family")
        elif row["level_sd"] > row["effect_sd"]:
            print(f"  SD of the {args.low} *level* across families: {row['level_sd']:.4f}"
                  " -- templates shift the level more than the contrast")
        else:
            print(f"  SD of the {args.low} level: {row['level_sd']:.4f}")
        # With one family there is no across-family SD, so no robustness claim is available --
        # saying "same sign" of a single number would read as evidence when it is none.
        verdict = ("only 1 family measured -- no robustness claim available" if n < 2 else
                   "ROBUST" if row["all_same_sign"] and ratio >= 3 else
                   "same sign but wording-sensitive" if row["all_same_sign"] else
                   "NOT ROBUST -- the effect changes sign by wording")
        print(f"  -> {verdict}")

    Path(args.out).write_text(json.dumps(
        {"contrast": label, "alpha": ALPHA, "n_boot": N_BOOT, "results": out}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
