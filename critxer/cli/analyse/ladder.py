#!/usr/bin/env python
"""Evaluate gate 3 on the ladder run.

Reads the checkpoints `run/ladder.py` writes and answers the one question the pilot exists to reach:
did any rung move the false-alarm rate away from R0? A rung must clear three bars:

1. **Its own model's N2 band** -- the split-half same-condition null, computed empirically
   from the persisted per-sample flags rather than modelled.
2. **Holm-Bonferroni across the declared family**. The naive "any one clears" rule fires at
   18.5% under a complete null, measured in Task 1.
3. **The gate-0 framing range for that model**, since a rung beating the strongest mirrored
   framing the model responds to at all is more likely an artefact than an effect.

Holm is reported as a *step-down list* rather than "did anything reject": which rungs survive is the
point, and `holm_rejects_any` answers only the family-level question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from critxer.core.audit import LADDER
from critxer.core.multiplicity import holm
from critxer.core.paths import artefact, run_dir
from critxer.core.power import n2_band
from critxer.core.resample import Streams, cluster_bootstrap, paired_bootstrap

# Contrasts between rungs, not against R0. Each isolates one step of the ladder, and vs-R0
# comparisons cannot substitute: R3 - R0 and R3u - R0 both being non-zero says nothing about
# whether the *conditionality* matters, which is the contrast the R3/R3u pair exists to identify.
PAIRWISE = (
    ("R1", "R0p", "repair-specific vs any stated future task"),
    ("R2", "R1", "self vs other as the future repairer (Khullar's null)"),
    ("R3", "R2", "enacted vs merely stated"),
    ("R3", "R3u", "conditional obligation vs unconditional repair"),
)
ALPHA = 0.05
N_BOOT = 20000
SEED = 20260805
# Every condition present except the baseline, rather than a hardcoded list: the same analysis
# serves the R4 directory, whose cells (AS/AO/US/UO/AF) are not ladder rungs. Ordered by LADDER
# first so ladder output stays in rung order, then anything else alphabetically.
def _contrasts(present: set[str]) -> list[str]:
    known = [r for r in LADDER if r != "R0" and r in present]
    return known + sorted(present - set(LADDER))


def rectangular(flags: list[list[int]]) -> np.ndarray:
    """Items whose every sample parsed, as an items x samples array.

    N2 needs disjoint halves of equal size, so ragged rows cannot contribute: padding would invent
    samples and reweighting would change the null.
    """
    if not flags:
        return np.empty((0, 0))
    width = max(len(f) for f in flags)
    return np.array([f for f in flags if len(f) == width], dtype=float)


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
           episodes: list | None = None) -> dict:
    """Effect, CI and p-value for mean(b - a), clustered on the episode where the record has one.

    Delegates to `critxer.core.resample` rather than bootstrapping here, so the same contrast
    cannot be significant in one table and null in another.

    The clustering rule is read off the record, not passed in by the caller: this analysis serves
    two directories, and ladder rungs have no episode while R4 cells cycle 465 targets over 50
    frozen episodes. `clustered` is persisted so a reader can tell which rule produced an interval.
    """
    keep = ~(np.isnan(a) | np.isnan(b))
    diff = b[keep] - a[keep]
    if episodes is not None:
        ep = np.asarray(episodes)[keep]
        out = cluster_bootstrap(diff, ep, rng, n_boot=N_BOOT, alpha=ALPHA)
        return {"n_paired": out.pop("n"), "clustered": True, **out}
    out = paired_bootstrap(diff, rng, n_boot=N_BOOT, alpha=ALPHA)
    return {"n_paired": out.pop("n"), "clustered": False, **out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(run_dir("ladder")))
    ap.add_argument("--family", default="F1")
    ap.add_argument("--gate0", action="append", default=[],
                    help="reanalysis json from analyse_gate0.py; repeatable")
    ap.add_argument("--out", default=str(artefact("ladder_verdict.json")))
    args = ap.parse_args()

    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(args.dir).glob(f"*__{args.family}.json")):
        rec = json.loads(path.read_text())
        runs.setdefault(rec["auditor"], {})[rec["condition"]] = rec
    if not runs:
        raise SystemExit(f"no checkpoints in {args.dir} for family {args.family}")

    framing_range = {}
    for g in args.gate0:
        for r in json.loads(Path(g).read_text())["results"]:
            framing_range[r["auditor"]] = abs(r["framing_range"])

    # One stream per (model, contrast): a shared generator makes every interval a function of
    # how many draws the models and rungs before it consumed.
    stream = Streams(SEED)
    results, pvals = {}, {}
    for model, by_cond in sorted(runs.items()):
        if "R0" not in by_cond:
            print(f"skip {model}: no R0 checkpoint to compare against")
            continue
        r0 = np.array(by_cond["R0"]["per_item_probs"])
        obs = rectangular(by_cond["R0"].get("per_item_flags", []))
        band = (n2_band(obs, stream(model, "n2_band"), n_boot=N_BOOT, alpha=ALPHA)
                if obs.size else float("nan"))
        row = {"far_r0": by_cond["R0"]["far"], "n2_band": band,
               "n2_items_used": int(obs.shape[0]) if obs.size else 0,
               "framing_range": framing_range.get(model), "rungs": {}}
        for rung in _contrasts(set(by_cond) - {"R0"}):
            eff = paired(r0, np.array(by_cond[rung]["per_item_probs"]),
                         stream(model, f"{rung}_vs_R0"), by_cond[rung].get("episode_ids"))
            eff["far"] = by_cond[rung]["far"]
            eff["beats_n2"] = bool(abs(eff["effect"]) > band) if band == band else None
            fr = framing_range.get(model)
            eff["within_framing_range"] = bool(abs(eff["effect"]) <= fr) if fr else None
            row["rungs"][rung] = eff
            pvals[(model, rung)] = eff["p"]
        results[model] = row

    # Pairwise rung contrasts, reported alongside but NOT folded into the gate-3 Holm family:
    # gate 3 asks whether any rung moved off R0, and adding contrasts that do not test that
    # question would inflate the family and cost power on the hypothesis being gated.
    for model, by_cond in sorted(runs.items()):
        if model not in results:
            continue
        pw = {}
        for a, b, why in PAIRWISE:
            if a in by_cond and b in by_cond:
                d = paired(np.array(by_cond[b]["per_item_probs"]),
                           np.array(by_cond[a]["per_item_probs"]),
                           stream(model, f"{a}_vs_{b}"), by_cond[a].get("episode_ids"))
                d["meaning"] = why
                pw[f"{a}-{b}"] = d
        results[model]["pairwise"] = pw

    decided = holm(pvals) if pvals else {}
    for (model, rung), d in decided.items():
        results[model]["rungs"][rung]["holm"] = d

    print("\n| model | rung | FAR | effect vs R0 | 95% CI | p | Holm α | beats N2 | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model, row in results.items():
        print(f"| {model} | R0 | {row['far_r0']:.4f} | — | — | — | — | "
              f"N2={row['n2_band']:.4f} | (baseline) |")
        for rung, e in row["rungs"].items():
            h = e.get("holm", {})
            ok = h.get("reject") and e.get("beats_n2")
            verdict = "MOVED" if ok else "no"
            if ok and e.get("within_framing_range") is False:
                verdict = "MOVED (exceeds framing range — suspect)"
            print(f"| {model} | {rung} | {e['far']:.4f} | {e['effect']:+.4f} | "
                  f"[{e['lo']:+.4f}, {e['hi']:+.4f}] | {e['p']:.4f} | "
                  f"{h.get('threshold', float('nan')):.4f} | "
                  f"{'yes' if e.get('beats_n2') else 'no'} | {verdict} |")

    if any(row.get("pairwise") for row in results.values()):
        print("\n| model | contrast | estimate | 95% CI | p | isolates |")
        print("|---|---|---|---|---|---|")
        for model, row in results.items():
            for name, d in row.get("pairwise", {}).items():
                print(f"| {model} | {name} | {d['effect']:+.4f} | "
                      f"[{d['lo']:+.4f}, {d['hi']:+.4f}] | {d['p']:.4f} | {d['meaning']} |")

    moved = [(m, r) for m, row in results.items() for r, e in row["rungs"].items()
             if e.get("holm", {}).get("reject") and e.get("beats_n2")]
    print(f"\nGATE 3: {'PASS' if moved else 'FAIL — THE KILL'}")
    if moved:
        print("  rungs that moved: " + ", ".join(f"{m}/{r}" for m, r in moved))
    else:
        print("  No rung in {R1, R2, R3, R3u} moved FAR beyond its model's N2 band under Holm")
        print("  correction, in either model. The kill gate says stop: record this verdict.")

    Path(args.out).write_text(json.dumps(
        {"alpha": ALPHA, "n_boot": N_BOOT, "family_size": len(pvals),
         "gate3_pass": bool(moved), "moved": moved, "results": results}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
