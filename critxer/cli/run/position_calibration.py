#!/usr/bin/env python
"""Does fault position affect detectability? The gate on the reworked repair-cost knob.

Method-vs-value failed as a repair-cost knob because the two families are caught by different
routes -- recompute vs read -- giving a 27-point detectability gap. Position is the proposed
replacement: an early fault forces recomputation of many downstream steps, a late one of few,
while the detection route stays constant because both are `local` faults.

That only works if detectability does not itself track position. This measures it: the same
source trace gets a local fault at its earliest and latest injectable step, and both are audited
under R0. Flat detection means a clean knob; a slope means a confound to stratify on.

Within-trace by construction, so resampling clusters on source trace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

from critxer.core import backend
from critxer.core.audit import AUDIT_SCHEMA, AuditItem, build_audit_messages, parse_audit
from critxer.core.inject import inject_at
from critxer.core.metrics import far, instability_index
from critxer.core.paths import artefact

MAX_TOKENS = 512
INJECTOR = backend.Endpoint("ministral", "mistralai/Ministral-3-14B-Instruct-2512",
                            "http://127.0.0.1:9023")
AUDITORS = [
    backend.Endpoint("qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B", "http://127.0.0.1:9021"),
    backend.Endpoint("gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it", "http://127.0.0.1:9022"),
]




def detection(raws, items) -> tuple[float, float]:
    probs = []
    for row, item in zip(raws, items, strict=True):
        flags = [
            a.reported_error
            for raw in row
            if raw is not None and (a := parse_audit(raw, len(item.steps))) is not None
        ]
        probs.append(float(np.mean(flags)) if flags else np.nan)
    arr = np.array(probs)
    return far(arr), instability_index(arr)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=100)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--out", default=str(artefact("position_calibration.json")))
    args = ap.parse_args()

    srcs = json.loads((artefact("injection_sources.json")).read_text())
    usable = [
        s for s in srcs
        if len(s["injectable_steps"]) >= 2
        and max(s["injectable_steps"]) - min(s["injectable_steps"]) >= 2
    ][: args.sources]
    rows: dict[str, dict] = {}
    for split in sorted({s["split"] for s in usable}):
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        rows.update({r["id"]: r for r in ds})
    print(f"within-trace position pairs to build: {len(usable)}", flush=True)

    gate = asyncio.Semaphore(args.concurrency)
    async with backend.client() as http:
        async def build(s: dict):
            row = rows[s["id"]]
            steps = list(row["steps"])
            answer = re.sub(r"\s+", " ", steps[-1]).strip()
            k_early, k_late = min(s["injectable_steps"]), max(s["injectable_steps"])
            async with gate:
                # Reasons discarded here: this command reports positions, not rejection classes.
                early, _ = await inject_at(http, INJECTOR, steps, k_early, answer,
                                           args.attempts)
                late, _ = await inject_at(http, INJECTOR, steps, k_late, answer, args.attempts)
            if early is None or late is None:
                return None
            return {"id": s["id"], "problem": row["problem"], "n_steps": len(steps),
                    "early": {"k": k_early, "steps": early, "downstream": len(steps) - k_early},
                    "late": {"k": k_late, "steps": late, "downstream": len(steps) - k_late}}

        built = [b for b in await asyncio.gather(*(build(s) for s in usable)) if b]
        print(f"complete position pairs: {len(built)}", flush=True)

        out = []
        for ep in AUDITORS:
            rec: dict = {"auditor": ep.label, "n_pairs": len(built)}
            for arm in ("early", "late"):
                items = [
                    AuditItem(f"{b['id']}::{arm}", b["problem"], b[arm]["steps"]) for b in built
                ]
                raws = await backend.map_prompts(
                    http, ep, [build_audit_messages(i, "R0", "F1") for i in items],
                    concurrency=args.concurrency, n=args.samples, temperature=0.7,
                    max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA)
                d, n1 = detection(raws, items)
                downstream = float(np.mean([b[arm]["downstream"] for b in built]))
                rec[arm] = {"detection": d, "instability_n1": n1,
                            "mean_downstream": downstream}
                print(f"  {ep.label} / {arm}: detection={d:.4f} "
                      f"(mean downstream steps {rec[arm]['mean_downstream']:.1f})", flush=True)
            rec["detection_gap"] = rec["early"]["detection"] - rec["late"]["detection"]
            out.append(rec)

    print("\n| auditor | early detection | late detection | gap | early/late downstream |")
    print("|---|---|---|---|---|")
    for r in out:
        print(f"| {r['auditor']} | {r['early']['detection']:.4f} | {r['late']['detection']:.4f} "
              f"| {r['detection_gap']:+.4f} | {r['early']['mean_downstream']:.1f} / "
              f"{r['late']['mean_downstream']:.1f} |")
    worst = max(abs(r["detection_gap"]) for r in out)
    print(f"\nlargest |gap| = {worst:.4f}. Compare with the 0.27 method-vs-value gap this "
          f"replaces: {'MATCHED -- clean knob' if worst < 0.10 else 'confounded -- stratify'}")

    Path(args.out).write_text(json.dumps({"condition": "R0", "family": "local",
                                         "pairs": len(built), "results": out}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
