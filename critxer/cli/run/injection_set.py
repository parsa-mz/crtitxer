#!/usr/bin/env python
"""Build and persist the position-based injection set.

Nothing built this. `run_injection_pilot.py` builds the local/structural pairs the position knob
*replaced*, and `run_position_calibration.py` builds early/late pairs but discards them after
measuring the detectability gap. The injected arm has therefore never had a persisted set matching
the design.

One `local` fault per position, at each source's earliest and latest injectable step, so the
repair-cost contrast is within-trace and resampling clusters on source trace. Both arms must
validate or the source is dropped: an incomplete pair loses the within-trace design.

Runs under the post-audit validator and normaliser, which reject two cue classes the earlier
sets contain -- markdown emphasis present only on injected steps (16.3% of the audited sample) and
a gutted step k (10.2%). Yield will be lower than previously reported for that reason, and the
older files should not be reused.
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
from critxer.core.inject import inject_at
from critxer.core.paths import artefact
from critxer.core.repairgym import reason_class

INJECTOR = backend.Endpoint("ministral", "mistralai/Ministral-3-14B-Instruct-2512",
                            "http://127.0.0.1:9023")
# Early and late must be far enough apart that "many downstream steps" and "few" differ at all.
MIN_SPAN = 2






async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(artefact("injection_sources_v2.json")))
    ap.add_argument("--limit", type=int, default=0, help="0 = every eligible source")
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--out", default=str(artefact("injection_set_position.json")))
    args = ap.parse_args()

    srcs = json.loads(Path(args.sources).read_text())
    usable = [
        s for s in srcs
        if len(s["injectable_steps"]) >= 2
        and max(s["injectable_steps"]) - min(s["injectable_steps"]) >= MIN_SPAN
    ]
    if args.limit:
        usable = usable[: args.limit]
    rows: dict[str, dict] = {}
    for split in sorted({s["split"] for s in usable}):
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        rows.update({r["id"]: r for r in ds})
    print(f"eligible sources (>=2 injectable steps, span >= {MIN_SPAN}): {len(usable)}", flush=True)

    gate = asyncio.Semaphore(args.concurrency)
    rejected: Counter = Counter()

    async def build(s: dict) -> dict | None:
        row = rows[s["id"]]
        steps = list(row["steps"])
        answer = re.sub(r"\s+", " ", steps[-1]).strip()
        k_early, k_late = min(s["injectable_steps"]), max(s["injectable_steps"])
        async with gate:
            early, r_e = await inject_at(http, INJECTOR, steps, k_early, answer,
                                         args.attempts)
            late, r_l = await inject_at(http, INJECTOR, steps, k_late, answer, args.attempts)
        for r in r_e + r_l:
            rejected[reason_class(r)] += 1
        if early is None or late is None:
            return None
        return {
            "id": s["id"], "split": s["split"], "generator": s.get("generator"),
            "problem": row["problem"], "n_steps": len(steps), "original": steps,
            "early": {"k": k_early, "steps": early, "downstream": len(steps) - k_early},
            "late": {"k": k_late, "steps": late, "downstream": len(steps) - k_late},
        }

    async with backend.client() as http:
        built = [b for b in await asyncio.gather(*(build(s) for s in usable)) if b]

    print(f"complete position pairs: {len(built)} / {len(usable)} "
          f"({len(built) / max(1, len(usable)):.1%})")
    print("rejection reasons across all attempts:")
    for reason, n in rejected.most_common():
        print(f"  {n:5d}  {reason}")
    print(f"by split: {dict(Counter(b['split'] for b in built))}")
    if built:
        e = sum(b["early"]["downstream"] for b in built) / len(built)
        latest = sum(b["late"]["downstream"] for b in built) / len(built)
        print(f"mean downstream steps: early {e:.1f}, late {latest:.1f}")

    Path(args.out).write_text(json.dumps(
        {"family": "local", "knob": "fault position", "min_span": MIN_SPAN,
         "attempts": args.attempts, "n_eligible": len(usable), "n_pairs": len(built),
         "rejected": dict(rejected), "pairs": built}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
