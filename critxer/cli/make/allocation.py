#!/usr/bin/env python
"""Materialise the canonical item allocation as a file everything reads.

Until now the split existed only as prose. Every script re-derived its own pool from
ProcessBench, so arm disjointness was unverified and arm sizes drifted between runs.

The sizes here are *derived*, not copied from the design, because the original 829 / 300 / 50 no
longer fits the instrument: the position knob needs a source with at least two injectable
steps spanning at least two, and only ~122 sources qualify. Asking for 300 would make
`allocate` raise, which is the intended behaviour -- a short arm must be loud. So the
injection-source arm takes every eligible source, warmup takes its fixed 50, and **the clean arm
takes the remainder**, which is larger than 829 and therefore strictly better for the primary
claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from critxer.core.allocation import allocate
from critxer.core.paths import artefact

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
SEED = 20260805
MIN_STEPS = 4
MIN_SPAN = 2
N_WARMUP = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(artefact("injection_sources_v2.json")))
    ap.add_argument("--out", default=str(artefact("allocation.json")))
    args = ap.parse_args()

    pool = []
    for split in SPLITS:
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        pool += [{"id": r["id"], "split": split}
                 for r in ds if r["label"] == -1 and len(r["steps"]) >= MIN_STEPS]
    print(f"clean pool: {len(pool)}  by split: {dict(Counter(p['split'] for p in pool))}")

    srcs = json.loads(Path(args.sources).read_text())
    in_pool = {p["id"] for p in pool}
    eligible = {
        s["id"] for s in srcs
        if s["id"] in in_pool
        and len(s["injectable_steps"]) >= 2
        and max(s["injectable_steps"]) - min(s["injectable_steps"]) >= MIN_SPAN
    }
    sizes = {"source": len(eligible), "warmup": N_WARMUP,
             "clean": len(pool) - len(eligible) - N_WARMUP}
    print(f"position-eligible sources: {len(eligible)}")
    print(f"arm sizes: {sizes}")

    arms = allocate(pool, sizes, seed=SEED, restrict={"source": eligible})
    split_of = {p["id"]: p["split"] for p in pool}

    # Check the properties the analysis assumes, here rather than only in unit tests: this is
    # the artefact the ladder run consumes, and a silent overlap would contaminate R4. Raising
    # rather than asserting, so `python -O` cannot strip the guard off the artefact's one writer.
    flat = [i for ids in arms.values() for i in ids]
    if not len(flat) == len(set(flat)) == sum(sizes.values()):
        raise SystemExit(f"arms overlap or are short: {len(flat)} ids, "
                         f"{len(set(flat))} distinct, {sum(sizes.values())} allocated")
    if not set(arms["source"]) <= eligible:
        raise SystemExit(f"{len(set(arms['source']) - eligible)} ineligible ids got into the "
                         "source arm")

    for name, ids in arms.items():
        print(f"  {name:7s} {len(ids):5d}  {dict(Counter(split_of[i] for i in ids))}")

    Path(args.out).write_text(json.dumps(
        {"seed": SEED, "min_steps": MIN_STEPS, "min_span": MIN_SPAN,
         "n_pool": len(pool), "sizes": sizes, "arms": arms}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
