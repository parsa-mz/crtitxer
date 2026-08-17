#!/usr/bin/env python
"""Task 3: generate injected faults and measure the acceptance yield.

Yield is the open question: it needs >=83% for the 363 eligible sources to cover the
300 the design requires. Prints a yield table and a breakdown of rejection reasons.

Both fault families are injected into the SAME source trace at the SAME step, so the
repair-cost contrast is within-trace. A source is only kept if BOTH families validate --
otherwise the pair is incomplete and the within-trace comparison is lost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from critxer.core import backend
from critxer.core.inject import (
    FAMILY_BRIEFS,
    build_injection_messages,
    parse_suffix,
    splice,
)
from critxer.core.paths import artefact
from critxer.core.repairgym import injectable_steps, reason_class, validate_injection

FAULT_FAMILIES = tuple(FAMILY_BRIEFS)




async def generate_pair(http, ep, source: dict, steps: list[str], step_k: int, answer: str,
                        attempts: int) -> dict:
    """Try both families at one step, retrying each independently."""
    out = {"id": source["id"], "step_k": step_k, "families": {}}
    for family in FAULT_FAMILIES:
        record = {"accepted": False, "attempts": 0, "reasons": []}
        for _ in range(attempts):
            record["attempts"] += 1
            raws = await backend.sample(
                http, ep, build_injection_messages(steps, step_k, family),
                n=1, temperature=0.8, max_tokens=2048,
            )
            if not raws or raws[0] is None:
                record["reasons"].append("empty generation")
                continue
            suffix = parse_suffix(raws[0], len(steps) - step_k + 1)
            if suffix is None:
                record["reasons"].append("wrong step count or empty step")
                continue
            injected = splice(steps, step_k, suffix)
            result = validate_injection(steps, injected, step_k, answer)
            if result.ok:
                record.update(accepted=True, steps=injected)
                break
            record["reasons"].extend(result.reasons)
        out["families"][family] = record
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=40)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--port", type=int, default=9022)
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--out", default=str(artefact("injection_pilot.json")))
    args = ap.parse_args()

    with (artefact("injection_sources.json")).open() as fh:
        sources = json.load(fh)[: args.sources]
    by_split: dict[str, dict] = {}
    for split in sorted({s["split"] for s in sources}):
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        by_split[split] = {r["id"]: r for r in ds}

    ep = backend.Endpoint("injector", args.model, f"http://127.0.0.1:{args.port}")
    gate = asyncio.Semaphore(args.concurrency)

    async def one(src: dict) -> dict:
        row = by_split[src["split"]][src["id"]]
        steps = list(row["steps"])
        cands = injectable_steps(steps)
        step_k = cands[len(cands) // 2]  # middle candidate: away from both ends
        answer = re.sub(r"\s+", " ", steps[-1]).strip()
        async with gate:
            return await generate_pair(http, ep, src, steps, step_k, answer, args.attempts)

    async with backend.client() as http:
        results = await asyncio.gather(*(one(s) for s in sources))

    per_family = {f: sum(r["families"][f]["accepted"] for r in results) for f in FAULT_FAMILIES}
    both = sum(all(r["families"][f]["accepted"] for f in FAULT_FAMILIES) for r in results)
    n = len(results)
    print(f"\nsources attempted: {n}   attempts/family: {args.attempts}   injector: {args.model}")
    print("| outcome | count | rate |")
    print("|---|---|---|")
    for f in FAULT_FAMILIES:
        print(f"| {f} accepted | {per_family[f]} | {per_family[f] / n:.1%} |")
    print(f"| **both accepted (usable pair)** | **{both}** | **{both / n:.1%}** |")
    print(f"\n>=83% pair yield needed for 363 sources to cover 300: "
          f"{'MET' if both / n >= 0.83 else 'NOT MET'}")

    print("\nrejection reasons (all attempts, both families):")
    counts = Counter(
        reason_class(r) for res in results for f in FAULT_FAMILIES
        for r in res["families"][f]["reasons"]
    )
    for k, v in counts.most_common():
        print(f"  {v:5d}  {k}")

    Path(args.out).write_text(json.dumps(
        {"injector": args.model, "attempts": args.attempts, "results": results}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
