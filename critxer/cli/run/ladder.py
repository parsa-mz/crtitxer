#!/usr/bin/env python
"""Task 7: the ladder run -- the primary claim, and gate 3 of the kill gate.

R0 / R1 / R2 / R3 / R3u over the clean arm, per model. Per-item report probabilities are persisted
so every downstream CI resamples items rather than trusting an aggregate.

**Checkpointed per (model, condition)**, so a crash five conditions in does not discard the first
four; a pass whose checkpoint exists is skipped, making the script resumable without flags.

**The promise is kept, but not for every sample.** The design requires the stated future repair to
actually happen so the instruction is not a lie, but doing it for every sample would roughly double
the run for tokens that cannot enter the DV -- the audit JSON is parsed *before* any repair token
exists. So it happens once per item per condition at T=0, and is logged for the analysis to state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from critxer.core import backend
from critxer.core.audit import (
    AUDIT_SCHEMA,
    CONDITIONS,
    LADDER,
    AuditItem,
    build_audit_messages,
    parse_audit,
)
from critxer.core.backend import parse_endpoint
from critxer.core.metrics import far, instability_index
from critxer.core.paths import artefact, run_dir
from critxer.core.r4 import REPAIR_REQUEST

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
MAX_TOKENS = 512
REPAIR_MAX_TOKENS = 1024
# Conditions whose prompt states that a repair will follow. R0 promises nothing.
PROMISES_REPAIR = ("R1", "R2", "R3", "R3u")




def load_arm(allocation: Path, arm: str) -> list[AuditItem]:
    wanted = set(json.loads(allocation.read_text())["arms"][arm])
    rows: dict[str, dict] = {}
    for split in SPLITS:
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        rows.update({r["id"]: r for r in ds if r["id"] in wanted})
    missing = wanted - rows.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} {arm} ids not in ProcessBench, e.g. {sorted(missing)[:3]}")
    return [AuditItem(i, rows[i]["problem"], list(rows[i]["steps"])) for i in sorted(wanted)]


def summarise(raws, items) -> dict:
    """Per-item report probability, the per-sample flags behind it, and parse rates.

    The individual flags are persisted, not just their mean: gate 3 is stated against the N2
    split-half band, and a split-half cannot be reconstructed from an 8-sample average. The mean
    alone would force a *modelled* N2 in place of the empirical band the spec asks for.
    """
    probs, flags_out, parsed, total, loc_usable = [], [], 0, 0, 0
    for row, item in zip(raws, items, strict=True):
        flags = []
        for raw in row:
            total += 1
            audit = parse_audit(raw, len(item.steps)) if raw is not None else None
            if audit is None:
                continue
            parsed += 1
            flags.append(audit.reported_error)
            if audit.localization_usable:
                loc_usable += 1
        probs.append(float(np.mean(flags)) if flags else np.nan)
        flags_out.append([int(f) for f in flags])
    arr = np.array(probs)
    return {
        "far": far(arr),
        "instability_n1": instability_index(arr),
        "parse_ok_rate": parsed / total if total else 0.0,
        "n_items_all_unparsed": int(np.isnan(arr).sum()),
        "localizable_samples": loc_usable,
        "per_item_probs": arr.tolist(),
        # Ragged wherever a sample failed to parse, so the analysis must not assume a rectangle.
        "per_item_flags": flags_out,
    }


async def keep_the_promise(http, ep, items, condition, family, concurrency) -> dict:
    """Perform the repair the prompt promised, once per item, at temperature 0.

    For R3/R3u the repair must occur in the *same* context, which is what appending to the audit
    turn does. For R1 it is nominally another model's job, so performing it here would make the R1
    prompt false; R1 is recorded as delegated rather than performed.
    """
    if condition == "R1":
        return {"mode": "delegated", "n_performed": 0}
    gate = asyncio.Semaphore(concurrency)

    async def one(item):
        msgs = build_audit_messages(item, condition, family)
        async with gate:
            raws = await backend.sample(http, ep, msgs, n=1, temperature=0.0,
                                        max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA)
            if not raws or raws[0] is None:
                return 0
            follow = [*msgs, {"role": "assistant", "content": raws[0].strip()},
                      {"role": "user", "content": REPAIR_REQUEST}]
            out = await backend.sample(http, ep, follow, n=1, temperature=0.0,
                                       max_tokens=REPAIR_MAX_TOKENS)
        return int(bool(out and out[0]))

    done = await asyncio.gather(*(one(i) for i in items))
    return {"mode": "same-context" if condition.startswith("R3") else "separate-request",
            "n_performed": int(sum(done))}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", action="append", required=True, metavar="LABEL=MODEL@URL")
    ap.add_argument("--conditions", default=",".join(LADDER))
    ap.add_argument("--family", default="F1")
    ap.add_argument("--samples", type=int, default=8)
    # store_true with default=True can never be switched off; the determinism check is the
    # default and --no-greedy is the escape hatch.
    ap.add_argument("--no-greedy", action="store_false", dest="greedy",
                    help="skip the T=0 determinism sample")
    ap.add_argument("--fulfil-promise", action="store_true", default=False,
                    help="perform the promised repair once per item; costs a second pass")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--allocation", default=str(artefact("allocation.json")))
    ap.add_argument("--outdir", default=str(run_dir("ladder")))
    args = ap.parse_args()

    conditions = args.conditions.split(",")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; known: {sorted(CONDITIONS)}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    items = load_arm(Path(args.allocation), "clean")
    print(f"clean arm: {len(items)} items; conditions {conditions}; family {args.family}",
          flush=True)

    async with backend.client() as http:
        for spec in args.endpoint:
            ep = parse_endpoint(spec)
            for condition in conditions:
                out = outdir / f"{ep.label}__{condition}__{args.family}.json"
                if out.exists():
                    print(f"  skip {out.name} (checkpoint exists)", flush=True)
                    continue
                prompts = [build_audit_messages(i, condition, args.family) for i in items]
                raws = await backend.map_prompts(
                    http, ep, prompts, concurrency=args.concurrency, n=args.samples,
                    temperature=0.7, max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA, seed=1)
                rec = {"auditor": ep.label, "model": ep.model, "condition": condition,
                       "family": args.family, "samples": args.samples, "temperature": 0.7,
                       "n_items": len(items), "item_ids": [i.item_id for i in items],
                       **summarise(raws, items)}

                if args.greedy:
                    g = await backend.map_prompts(
                        http, ep, prompts, concurrency=args.concurrency, n=1, temperature=0.0,
                        max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA)
                    rec["greedy"] = summarise(g, items)

                if args.fulfil_promise and condition in PROMISES_REPAIR:
                    rec["promise"] = await keep_the_promise(
                        http, ep, items, condition, args.family, args.concurrency)

                out.write_text(json.dumps(rec, indent=1))
                print(f"  {ep.label} {condition}: FAR={rec['far']:.4f} "
                      f"N1={rec['instability_n1']:.4f} parse={rec['parse_ok_rate']:.3f} "
                      f"-> {out.name}", flush=True)

    print("\ndone. analyse with `critxer analyse-ladder`")


if __name__ == "__main__":
    asyncio.run(main())
