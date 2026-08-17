#!/usr/bin/env python
"""Three cheap kill tests in one run, before any ladder generation.

Each can end the project more cheaply than the ladder can, and all three read the same generations:

1. **Template SD** -- a condition effect has to exceed the template random-effect SD. If SD
   exceeds the MDE, the effect sits below prompt noise and the ladder buys an uninterpretable
   number.
2. **R0-vs-R0 sham replicate** -- two independent runs of the *same* condition, so whatever appears
   is pure nuisance (token layout, batch composition, prefix cache, order). The MDE must exceed it.
3. **Positive control** -- a heavy-handed strictness prime. If it does not move FAR, the instrument
   cannot see a framing effect at all and a null across the ladder would be uninterpretable.

Writes a verdict per model. Nothing downstream should run until all three pass.
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
    WORDINGS,
    AuditItem,
    build_audit_messages,
    parse_audit,
)
from critxer.core.metrics import far, instability_index
from critxer.core.paths import artefact

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
MAX_TOKENS = 512
MDE = 0.03
AUDITORS = [
    backend.Endpoint("qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B", "http://127.0.0.1:9021"),
    backend.Endpoint("gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it", "http://127.0.0.1:9022"),
]
SEED = 20260805


def load_clean(n: int) -> list[AuditItem]:
    """Clean seeds (label == -1, >=4 steps), stratified across the four sources."""
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


async def measure(http, ep, items, condition, family, samples, concurrency, seed) -> np.ndarray:
    prompts = [build_audit_messages(i, condition, family) for i in items]
    raws = await backend.map_prompts(
        http, ep, prompts, concurrency=concurrency, n=samples, temperature=0.7,
        max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA, seed=seed)
    return probs(raws, items)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=str(artefact("kill_tests.json")))
    args = ap.parse_args()

    items = load_clean(args.items)
    print(f"clean items: {len(items)}", flush=True)
    results = []

    async with backend.client() as http:
        for ep in AUDITORS:
            rec: dict = {"auditor": ep.label, "n_items": len(items)}

            # 1. R0 under every template family; also gives replicate A on F1.
            by_family = {}
            for family in WORDINGS:
                p = await measure(http, ep, items, "R0", family, args.samples,
                                  args.concurrency, seed=1)
                by_family[family] = far(p)
                print(f"  {ep.label} R0/{family}: FAR={by_family[family]:.4f}", flush=True)
            rec["far_by_template"] = by_family
            rec["template_sd"] = float(np.std(list(by_family.values()), ddof=1))

            # 2. Sham replicate: same condition, same template, different sampling seed.
            p_a = await measure(http, ep, items, "R0", "F1", args.samples,
                                args.concurrency, seed=1)
            p_b = await measure(http, ep, items, "R0", "F1", args.samples,
                                args.concurrency, seed=2)
            rec["sham_far_a"], rec["sham_far_b"] = far(p_a), far(p_b)
            rec["sham_abs_diff"] = abs(rec["sham_far_a"] - rec["sham_far_b"])
            rec["instability_n1"] = instability_index(p_a)
            print(f"  {ep.label} sham R0 vs R0: {rec['sham_abs_diff']:.4f}", flush=True)

            # 3. Positive control.
            p_pc = await measure(http, ep, items, "PC", "F1", args.samples,
                                 args.concurrency, seed=1)
            rec["far_pc"] = far(p_pc)
            rec["pc_effect"] = rec["far_pc"] - rec["sham_far_a"]
            print(f"  {ep.label} PC: FAR={rec['far_pc']:.4f} "
                  f"(effect {rec['pc_effect']:+.4f})", flush=True)
            results.append(rec)

    print("\n| model | template SD | sham |R0-R0| | PC effect | verdict |")
    print("|---|---|---|---|---|")
    for r in results:
        fails = []
        if r["template_sd"] > MDE:
            fails.append("template SD > MDE")
        if r["sham_abs_diff"] > MDE:
            fails.append("nuisance > MDE")
        if abs(r["pc_effect"]) < 2 * MDE:
            fails.append("PC too weak")
        verdict = "PASS" if not fails else "; ".join(fails)
        print(f"| {r['auditor']} | {r['template_sd']:.4f} | {r['sham_abs_diff']:.4f} | "
              f"{r['pc_effect']:+.4f} | {verdict} |")
    print(f"\nMDE = {MDE}. Template SD and nuisance must be BELOW it; PC must clearly exceed it.")

    Path(args.out).write_text(json.dumps({"mde": MDE, "results": results}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
