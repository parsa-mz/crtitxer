#!/usr/bin/env python
"""Detectability of early vs late faults on the position injection set. Secondary.

R0 only. The injected arm is a *descriptive* finding with no framing manipulation: how much
easier is a fault to spot when many downstream steps depend on it than when few do?

Both arms come from the same source trace, so the comparison is within-trace and resampling
clusters on **source trace**, never on item -- an early/late pair is one observation, not two.

Runs on `injection_set_position.json`, built under the post-audit validator. The older sets
contain the markdown-emphasis cue found in 16.3% of the audited sample, which inflates detection
for reasons unrelated to the fault, so their detection rates are not comparable to these.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from critxer.core import backend
from critxer.core.audit import AUDIT_SCHEMA, AuditItem, build_audit_messages, parse_audit
from critxer.core.backend import parse_endpoint
from critxer.core.paths import artefact

MAX_TOKENS = 512
ALPHA = 0.05
N_BOOT = 20000
SEED = 20260805




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


def paired_by_trace(early: np.ndarray, late: np.ndarray, rng: np.random.Generator) -> dict:
    """Paired bootstrap on the early-minus-late difference, resampling source traces.

    Not `cluster_bootstrap`: the unit here IS the source trace, which contributes one early and one
    late measurement, so the pairing already absorbs it. Episode-clustered semantics do not apply.
    """
    keep = ~(np.isnan(early) | np.isnan(late))
    d = (early - late)[keep]
    reps = d[rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)
    return {"n_traces": int(d.size), "gap": float(d.mean()),
            "lo": float(np.quantile(reps, ALPHA / 2)),
            "hi": float(np.quantile(reps, 1 - ALPHA / 2))}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", action="append", required=True, metavar="LABEL=MODEL@URL")
    ap.add_argument("--set", default=str(artefact("injection_set_position.json")))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--out", default=str(artefact("position_detectability.json")))
    args = ap.parse_args()

    pairs = json.loads(Path(args.set).read_text())["pairs"]
    print(f"source traces: {len(pairs)} (each contributing one early and one late fault)")
    rng = np.random.default_rng(SEED)
    out = []

    async with backend.client() as http:
        for spec in args.endpoint:
            ep = parse_endpoint(spec)
            rec: dict = {"auditor": ep.label, "n_traces": len(pairs)}
            per_arm = {}
            for arm in ("early", "late"):
                items = [AuditItem(f"{p['id']}::{arm}", p["problem"], p[arm]["steps"])
                         for p in pairs]
                raws = await backend.map_prompts(
                    http, ep, [build_audit_messages(i, "R0", "F1") for i in items],
                    concurrency=args.concurrency, n=args.samples, temperature=0.7,
                    max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA, seed=1)
                per_arm[arm] = probs(raws, items)
                rec[arm] = {
                    "detection": float(np.nanmean(per_arm[arm])),
                    "mean_downstream": float(np.mean([p[arm]["downstream"] for p in pairs])),
                    "per_trace": per_arm[arm].tolist(),
                }
                print(f"  {ep.label} {arm}: detection={rec[arm]['detection']:.4f} "
                      f"(mean downstream {rec[arm]['mean_downstream']:.1f})", flush=True)
            rec["gap"] = paired_by_trace(per_arm["early"], per_arm["late"], rng)
            out.append(rec)

    print("\n| auditor | early | late | gap (early − late) | 95% CI | in gate-2 band? |")
    print("|---|---|---|---|---|---|")
    for r in out:
        e, latest = r["early"]["detection"], r["late"]["detection"]
        band = "yes" if all(0.35 <= v <= 0.90 for v in (e, latest)) else "no"
        g = r["gap"]
        print(f"| {r['auditor']} | {e:.4f} | {latest:.4f} | {g['gap']:+.4f} | "
              f"[{g['lo']:+.4f}, {g['hi']:+.4f}] | {band} |")

    Path(args.out).write_text(json.dumps(
        {"condition": "R0", "family": "F1", "n_boot": N_BOOT, "results": out}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
