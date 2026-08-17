#!/usr/bin/env python
"""Score the gate-0 controls from persisted per-item probabilities. No generation.

Two things it does that a single-pass report gets wrong:

* **A nuisance band per model.** One sham CI bound across all models judges the quietest model
  against the noisiest one's noise, and the shams here differ by ~30x.
* **A pooled R0 reference.** The sham offset is systematic, not an error bar around zero: with the
  seed fixed, sampling is deterministic, so seed 1 vs seed 2 is one draw from the space of seed
  pairs and the offset is a property of that pair. Measuring controls against a single R0 run
  inherits the whole offset; pooling both halves the reference's variance and centres it.

This matters because a downward effect can be smaller than its own model's sham offset in the same
direction -- and downward is the direction the ladder predicts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from critxer.core.audit import CONTROLS
from critxer.core.paths import artefact

N_BOOT = 20000
ALPHA = 0.05
SEED = 20260805


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    keep = ~(np.isnan(a) | np.isnan(b))
    diff = b[keep] - a[keep]
    boot = diff[rng.integers(0, diff.size, size=(N_BOOT, diff.size))].mean(axis=1)
    return {
        "effect": float(diff.mean()),
        "lo": float(np.quantile(boot, ALPHA / 2)),
        "hi": float(np.quantile(boot, 1 - ALPHA / 2)),
    }


def fmt(d: dict) -> str:
    return f"{d['effect']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(artefact("pc_diagnostic.json")))
    ap.add_argument("--out", default=str(artefact("pc_diagnostic_reanalysis.json")))
    args = ap.parse_args()
    blob = json.loads(Path(args.src).read_text())
    rng = np.random.default_rng(SEED)
    out = []

    for rec in blob["results"]:
        p = {k: np.array(v) for k, v in rec["per_item_probs"].items()}
        # 16-sample reference: the mean of two 8-sample runs is the 16-sample estimate, since
        # every item contributes the same count to each.
        ref = (p["R0#1"] + p["R0#2"]) / 2
        sham = paired(p["R0#1"], p["R0#2"], rng)
        # Own nuisance band: how far a control must sit from zero to be distinguishable from
        # this harness's own run-to-run offset on this model.
        band = max(abs(sham["lo"]), abs(sham["hi"]))
        eff = {c: paired(ref, p[f"{c}#1"], rng) for c in CONTROLS}
        row = {
            "auditor": rec["auditor"],
            "far_r0_pooled": float(np.nanmean(ref)),
            "instability_n1": rec["instability_n1"],
            "sham": sham,
            "own_band": band,
            "effects": eff,
            # Does the CI clear the model's own nuisance band, in either direction?
            "clears": {
                c: bool(eff[c]["lo"] > band or eff[c]["hi"] < -band) for c in CONTROLS
            },
            "framing_range": eff["PC"]["effect"] - eff["PCL"]["effect"],
            "downward_headroom": float(np.nanmean(ref)),
        }
        out.append(row)

    print("| model | R0 FAR (n=16) | N1 | own sham band | PC | PCL | PCH |")
    print("|---|---|---|---|---|---|---|")
    for r in out:
        marks = {c: ("" if r["clears"][c] else " ns") for c in CONTROLS}
        print(f"| {r['auditor']} | {r['far_r0_pooled']:.4f} | {r['instability_n1']:.4f} "
              f"| {r['own_band']:.4f} | " + " | ".join(
                  fmt(r["effects"][c]) + marks[c] for c in CONTROLS) + " |")
    print("\n'ns' = CI does not clear that model's own sham band. Reference is R0 pooled to n=16.")

    print("\n--- gate 0 per model ---")
    for r in out:
        dead = not r["clears"]["PCH"]
        unmeasurable = not (r["clears"]["PC"] or r["clears"]["PCL"])
        # Only the downward direction matters for the ladder's predicted sign.
        down = r["clears"]["PCL"]
        verdict = ("FAIL: dead DV" if dead else
                   "FAIL: framing unmeasurable" if unmeasurable else
                   "PASS" if down else "PASS (upward only) -- ladder direction unverified")
        print(f"{r['auditor']}: {verdict}")
        print(f"  framing range PC-PCL = {r['framing_range']:+.4f}  "
              f"downward headroom = {r['downward_headroom']:.4f}  "
              f"operative MDE = max(0.03, {abs(r['framing_range']):.4f}) = "
              f"{max(0.03, abs(r['framing_range'])):.4f}")

    Path(args.out).write_text(json.dumps(
        {"n_boot": N_BOOT, "reference": "R0 pooled over seeds 1 and 2 (n=16)", "results": out},
        indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
