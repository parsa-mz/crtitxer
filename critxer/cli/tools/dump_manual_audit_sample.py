#!/usr/bin/env python
"""Render a 50-item stratified sample for the manual injection audit.

The audit is blocking and it had never been run. The automated checks in
`repairgym.validate_injection` can only test arithmetic consistency and answer change; they
cannot test whether a human reading the trace would call the fault *single-step-attributable* and
coherent, which is the claim the injected arm rests on.

Re-validates under the *current* validator first, because `injection_set_full.json` was accepted
under the version whose propagation guard was near-vacuous -- 22% of that file is invalid, and
auditing rejected items would measure nothing.

Prints, per item, only what the attributability question needs: original step k, injected step k,
the next injected step (where propagation shows up), and both final answers. Full traces are
available in the JSON if an item looks wrong.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

from critxer.core.paths import artefact
from critxer.core.repairgym import final_answer, validate_injection

SEED = 20260805


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(artefact("injection_set_full.json")))
    ap.add_argument("--family", default="local")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=str(artefact("manual_audit_sample.json")))
    args = ap.parse_args()

    results = json.loads(Path(args.src).read_text())["results"]
    srcs = {s["id"]: s for s in json.loads((artefact("injection_sources.json")).read_text())}
    rows: dict[str, dict] = {}
    for split in sorted({s["split"] for s in srcs.values()}):
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        rows.update({r["id"]: r for r in ds})

    kept, dropped = [], Counter()
    for r in results:
        fam = r["families"].get(args.family) or {}
        if not fam.get("accepted"):
            dropped["not accepted originally"] += 1
            continue
        row = rows.get(r["id"])
        if row is None:
            dropped["source row missing"] += 1
            continue
        original = list(row["steps"])
        answer = re.sub(r"\s+", " ", original[-1]).strip()
        v = validate_injection(original, fam["steps"], r["step_k"], answer)
        if not v.ok:
            dropped[f"revalidate: {"; ".join(v.reasons)[:40]}"] += 1
            continue
        kept.append({"id": r["id"], "split": srcs[r["id"]]["split"], "step_k": r["step_k"],
                     "problem": row["problem"], "original": original, "injected": fam["steps"]})

    print(f"re-validated under the current validator: {len(kept)} of {len(results)} kept")
    for reason, n in dropped.most_common():
        print(f"  dropped {n}: {reason}")

    # Stratified by split, proportional, deterministic.
    rng = np.random.default_rng(SEED)
    by_split: dict[str, list] = {}
    for k in kept:
        by_split.setdefault(k["split"], []).append(k)
    sample = []
    for split in sorted(by_split):
        pool = by_split[split]
        take = max(1, round(args.n * len(pool) / len(kept)))
        sample += [pool[i] for i in rng.permutation(len(pool))[:take]]
    sample = sample[: args.n]
    print(f"\nsample: {len(sample)}  by split: {dict(Counter(s['split'] for s in sample))}\n")

    for i, s in enumerate(sample, 1):
        k = s["step_k"]
        nxt_o = s["original"][k] if k < len(s["original"]) else "(none -- k is last step)"
        nxt_i = s["injected"][k] if k < len(s["injected"]) else "(none -- k is last step)"
        print(f"--- [{i}] {s['id']} ({s['split']}) k={k} of {len(s['original'])} steps")
        print(f"  ORIG k : {' '.join(s['original'][k - 1].split())[:300]}")
        print(f"  INJ  k : {' '.join(s['injected'][k - 1].split())[:300]}")
        print(f"  ORIG k+1: {' '.join(nxt_o.split())[:220]}")
        print(f"  INJ  k+1: {' '.join(nxt_i.split())[:220]}")
        print(f"  answer: {final_answer(s['original'][-1])} -> {final_answer(s['injected'][-1])}")

    Path(args.out).write_text(json.dumps(
        {"n_revalidated": len(kept), "n_source_records": len(results),
         "dropped": dict(dropped), "sample": sample}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
