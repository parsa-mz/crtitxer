#!/usr/bin/env python
"""Task 3 final step: evaluate gate 2 on *injected* faults.

Gate 2 requires R0 detection on injected items to sit in [0.35, 0.90] -- real faults with room
to move. It could not be evaluated in Task 0, which had to substitute ProcessBench's natural
faults because repairgym did not exist yet; those came in at 0.89-0.91 on an easier population.

This runs both auditors over the generated injected items under R0 only and reports detection
per fault family, which is also the calibration signal: if detection sits above 0.90 the
injected faults are too obvious and the family briefs need to produce subtler ones.

Both families come from the same source trace, so family comparisons are within-trace and
resampling clusters on source trace, never on item.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from critxer.core import backend
from critxer.core.audit import AUDIT_SCHEMA, AuditItem, build_audit_messages, parse_audit
from critxer.core.inject import FAMILY_BRIEFS
from critxer.core.metrics import far, instability_index
from critxer.core.paths import artefact

FAULT_FAMILIES = tuple(FAMILY_BRIEFS)
MAX_TOKENS = 512
AUDITORS = [
    backend.Endpoint("qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B", "http://127.0.0.1:9021"),
    backend.Endpoint("gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it", "http://127.0.0.1:9022"),
]


def load_problems(ids: list[str]) -> dict[str, str]:
    """Problem statements by id.

    Required, not optional: the injection file stores only steps, and an auditor cannot judge
    whether a *method* is wrong without the problem it was meant to solve.
    """
    sources = json.loads((artefact("injection_sources.json")).read_text())
    split_of = {s["id"]: s["split"] for s in sources}
    wanted = set(ids)
    out: dict[str, str] = {}
    for split in sorted({split_of[i] for i in ids}):
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        out.update({r["id"]: r["problem"] for r in ds if r["id"] in wanted})
    missing = wanted - out.keys()
    if missing:
        raise RuntimeError(
            f"no problem statement for {len(missing)} ids, e.g. {sorted(missing)[:3]}"
        )
    return out


def load_pairs(path: Path, limit: int | None) -> list[dict]:
    """Sources where BOTH families validated -- an incomplete pair loses the within-trace design."""
    with path.open() as fh:
        results = json.load(fh)["results"]
    pairs = [r for r in results if all(r["families"][f].get("accepted")
                                      for f in FAULT_FAMILIES)]
    return pairs[:limit] if limit else pairs


def per_item(raws, items, true_steps) -> tuple[np.ndarray, dict]:
    """Report probability per item, plus localization accuracy against the injected step."""
    probs, exact, near, usable = [], 0, 0, 0
    for row, item, k in zip(raws, items, true_steps, strict=True):
        flags = []
        for raw in row:
            audit = parse_audit(raw, len(item.steps)) if raw is not None else None
            if audit is None:
                continue
            flags.append(audit.reported_error)
            if audit.localization_usable:
                usable += 1
                exact += audit.first_error_step == k
                near += abs(audit.first_error_step - k) <= 1
        probs.append(float(np.mean(flags)) if flags else np.nan)
    return np.array(probs), {
        "localizable_samples": usable,
        "exact_step_accuracy": exact / usable if usable else None,
        "within_one_step_accuracy": near / usable if usable else None,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injections", default=str(artefact("injection_set_full.json")))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=str(artefact("gate2_calibration.json")))
    args = ap.parse_args()

    pairs = load_pairs(Path(args.injections), args.limit)
    problems = load_problems([p["id"] for p in pairs])
    print(f"complete pairs available: {len(pairs)}", flush=True)

    out = []
    async with backend.client() as http:
        for ep in AUDITORS:
            row: dict = {"auditor": ep.label, "n_sources": len(pairs)}
            for family in FAULT_FAMILIES:
                items = [
                    AuditItem(f"{p['id']}::{family}", problems[p["id"]],
                              p["families"][family]["steps"])
                    for p in pairs
                ]
                prompts = [build_audit_messages(i, "R0", "F1") for i in items]
                raws = await backend.map_prompts(
                    http, ep, prompts, concurrency=args.concurrency, n=args.samples,
                    temperature=0.7, max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA)
                probs, loc = per_item(raws, items, [p["step_k"] for p in pairs])
                row[family] = {
                    "detection_rate": far(probs),
                    "instability_n1": instability_index(probs),
                    **loc,
                }
                print(f"  {ep.label} / {family}: detection={row[family]['detection_rate']:.4f}",
                      flush=True)
            out.append(row)

    print("\n| auditor | family | detection | gate 2 [0.35,0.90] | N1 | exact loc |")
    print("|---|---|---|---|---|---|")
    for row in out:
        for family in FAULT_FAMILIES:
            r = row[family]
            d = r["detection_rate"]
            verdict = "PASS" if 0.35 <= d <= 0.90 else ("too obvious" if d > 0.90 else "too hard")
            loc = r["exact_step_accuracy"]
            loc_cell = f"{loc:.4f}" if loc is not None else "-"
            print(f"| {row['auditor']} | {family} | {d:.4f} | {verdict} | "
                  f"{r['instability_n1']:.4f} | {loc_cell} |")

    Path(args.out).write_text(json.dumps({"condition": "R0", "results": out}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
