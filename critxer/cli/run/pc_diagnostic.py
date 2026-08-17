#!/usr/bin/env python
"""Kill test 3, rebuilt: can the dependent variable move, and by how much?

Three conditions, because the original single positive control conflated two things:

* **PCL**, a leniency prime -- the direction-matched control. The ladder predicts FAR *falls*, and
  an auditor near the bottom of its own FAR range can respond upward while having no downward
  headroom, so upward sensitivity does not establish the sensitivity the ladder needs.
* **PC**, a strictness prime, which moves FAR up.
* **PCH**, an explicit threshold instruction rather than a framing, which separates "framing is a
  weak lever" from "the dependent variable cannot move at all".

Effects are judged against a *paired sham null* -- two independent runs of R0 itself, resampled the
same way -- so the question is whether an effect exceeds this harness's own nuisance rather than
whether it clears a bar picked before any dispersion was measured. Per-item report probabilities are
persisted, without which no interval can be put on any of these differences.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from critxer.core import backend
from critxer.core.audit import AUDIT_SCHEMA, CONTROLS, AuditItem, build_audit_messages, parse_audit
from critxer.core.backend import parse_endpoint
from critxer.core.metrics import far, instability_index
from critxer.core.paths import artefact

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
MAX_TOKENS = 512
SEED = 20260805
N_BOOT = 5000
ALPHA = 0.05

AUDITORS = [
    backend.Endpoint("qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B", "http://127.0.0.1:9021"),
    backend.Endpoint("gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it", "http://127.0.0.1:9022"),
]



# R0 twice: the first is the reference every control is paired against, the second turns the
# sham replicate into a *paired* null in the same units as the effects, which the aggregate
# FAR difference in run_kill_tests.py could not provide.
PASSES = (("R0", 1), ("R0", 2), *((c, 1) for c in CONTROLS))


def load_clean(n: int) -> list[AuditItem]:
    """The same clean seeds run_kill_tests.py used -- same SEED, same stratification."""
    rng = np.random.default_rng(SEED)
    out: list[AuditItem] = []
    for split in SPLITS:
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        pool = [r for r in ds if r["label"] == -1 and len(r["steps"]) >= 4]
        for r in (pool[i] for i in rng.permutation(len(pool))[: n // len(SPLITS)]):
            out.append(AuditItem(r["id"], r["problem"], list(r["steps"])))
    return out


def probs(raws, items) -> np.ndarray:
    out = []
    for row, item in zip(raws, items, strict=True):
        flags = [
            a.reported_error
            for raw in row
            if raw is not None and (a := parse_audit(raw, len(item.steps))) is not None
        ]
        out.append(float(np.mean(flags)) if flags else np.nan)
    return np.array(out)


def paired_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """Bootstrap CI for mean(b - a), resampling items.

    Items are the resampling unit: on the clean arm each is one independent source trace, so there
    is no cluster structure to respect. Pairing is what buys the precision.
    """
    keep = ~(np.isnan(a) | np.isnan(b))
    diff = b[keep] - a[keep]
    idx = rng.integers(0, diff.size, size=(N_BOOT, diff.size))
    boot = diff[idx].mean(axis=1)
    return {
        "n_paired": int(diff.size),
        "effect": float(diff.mean()),
        "ci_lo": float(np.quantile(boot, ALPHA / 2)),
        "ci_hi": float(np.quantile(boot, 1 - ALPHA / 2)),
        "boot_sd": float(boot.std(ddof=1)),
    }


async def measure(http, ep, items, condition, samples, concurrency, seed) -> np.ndarray:
    prompts = [build_audit_messages(i, condition, "F1") for i in items]
    raws = await backend.map_prompts(
        http, ep, prompts, concurrency=concurrency, n=samples, temperature=0.7,
        max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA, seed=seed)
    return probs(raws, items)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=str(artefact("pc_diagnostic.json")))
    ap.add_argument("--endpoint", action="append", metavar="LABEL=MODEL@URL",
                    help="override the auditor list; repeatable")
    args = ap.parse_args()
    auditors = [parse_endpoint(s) for s in args.endpoint] if args.endpoint else AUDITORS

    items = load_clean(args.items)
    print(f"clean items: {len(items)}", flush=True)
    rng = np.random.default_rng(SEED)
    results = []

    async with backend.client() as http:
        for ep in auditors:
            per_item: dict[str, np.ndarray] = {}
            for condition, seed in PASSES:
                key = f"{condition}#{seed}"
                per_item[key] = await measure(http, ep, items, condition, args.samples,
                                             args.concurrency, seed)
                print(f"  {ep.label} {key}: FAR={far(per_item[key]):.4f}", flush=True)

            ref = per_item["R0#1"]
            rec: dict = {
                "auditor": ep.label,
                "n_items": len(items),
                "far": {k: far(v) for k, v in per_item.items()},
                "instability_n1": instability_index(ref),
                # The null: R0 against itself, paired and resampled exactly like every effect.
                "sham": paired_ci(ref, per_item["R0#2"], rng),
                "effects": {
                    c: paired_ci(ref, per_item[f"{c}#1"], rng) for c in CONTROLS
                },
                # How far FAR could fall at all. The ladder's predicted direction is downward,
                # so an effect larger than this is not merely undetected but impossible.
                "downward_headroom": far(ref),
                "per_item_probs": {k: v.tolist() for k, v in per_item.items()},
            }
            results.append(rec)

    nuisance = max(abs(r["sham"]["ci_lo"]) for r in results), max(
        abs(r["sham"]["ci_hi"]) for r in results)
    band = max(abs(v) for v in nuisance)

    print("\n| model | control | effect | 95% CI | beats sham? |")
    print("|---|---|---|---|---|")
    for r in results:
        s = r["sham"]
        print(f"| {r['auditor']} | sham R0/R0 | {s['effect']:+.4f} | "
              f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | (null) |")
        for c in CONTROLS:
            e = r["effects"][c]
            beats = "yes" if e["ci_lo"] > band or e["ci_hi"] < -band else "no"
            print(f"| {r['auditor']} | {c} | {e['effect']:+.4f} | "
                  f"[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] | {beats} |")

    print(f"\nnuisance band (widest sham CI bound across models) = {band:.4f}")
    for r in results:
        rng_span = r["effects"]["PC"]["effect"] - r["effects"]["PCL"]["effect"]
        print(f"{r['auditor']}: framing dynamic range PC-PCL = {rng_span:+.4f}; "
              f"PCH = {r['effects']['PCH']['effect']:+.4f}; "
              f"downward headroom = {r['downward_headroom']:.4f}")

    print(
        "\nVerdicts. (a) DEAD DV if PCH's CI contains 0 -- an explicit threshold instruction "
        "cannot move it, so nothing will. (b) FRAMING UNMEASURABLE if neither PC nor PCL clears "
        "the nuisance band. (c) Otherwise the instrument works, and PC-PCL is the honest MDE to "
        "pre-register: no framing manipulation should be expected to beat it."
    )

    Path(args.out).write_text(json.dumps(
        {"n_boot": N_BOOT, "alpha": ALPHA, "nuisance_band": band, "results": results}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
