#!/usr/bin/env python
"""Task 0: model bake-off / instrument check -- gates 1 and 2 of the kill gate.

Runs condition R0 only against both candidate models and reports the numbers gates 1 and 2
need, plus the two quantities that set later design choices:

  * N1 instability index on clean seeds        -> gate 1 (< 0.25)
  * R0 detection rate on natural faulty traces -> gate 2 (in [0.35, 0.90])
  * baseline FAR on clean seeds                -> sets n=8 vs n=12 (threshold 0.25)
  * T=0 disagreement rate across two batch compositions -> evidence for the sampled DV
  * localization accuracy against human first-error labels, and parse-failure rate

Detection is measured on ProcessBench's *natural* faulty traces (`label >= 0`, human-annotated
first-error step) rather than on repairgym injections, which do not exist yet -- Task 3 builds
them. Natural faults are also the ecologically valid capability check and cost nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

from critxer.core import backend
from critxer.core.audit import AUDIT_SCHEMA, AuditItem, build_audit_messages, parse_audit
from critxer.core.metrics import far, instability_index
from critxer.core.paths import artefact

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
ENDPOINTS = [
    backend.Endpoint("qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B",
                     "http://127.0.0.1:9021", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    backend.Endpoint("gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it",
                     "http://127.0.0.1:9022", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
]
SEED = 20260805
# Headroom above the schema's 300-char evidence cap so a well-formed audit always fits.
MAX_TOKENS = 512


def load_items(n_clean: int, n_faulty: int) -> tuple[list[AuditItem], list[tuple[AuditItem, int]]]:
    """Stratified clean seeds and natural faulty traces, balanced across the four sources."""
    rng = np.random.default_rng(SEED)
    clean: list[AuditItem] = []
    faulty: list[tuple[AuditItem, int]] = []
    per_clean, per_faulty = n_clean // len(SPLITS), n_faulty // len(SPLITS)

    for split in SPLITS:
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        # >=4 steps only: shorter traces have nowhere to hold a mid-trace fault, and the
        # pilot's injected arm excludes them, so the bake-off must match that pool.
        ok = [r for r in ds if len(r["steps"]) >= 4]
        c = [r for r in ok if r["label"] == -1]
        f = [r for r in ok if r["label"] >= 0]
        for r in (c[i] for i in rng.permutation(len(c))[:per_clean]):
            clean.append(AuditItem(r["id"], r["problem"], list(r["steps"])))
        for r in (f[i] for i in rng.permutation(len(f))[:per_faulty]):
            faulty.append((AuditItem(r["id"], r["problem"], list(r["steps"])), r["label"]))
    return clean, faulty


def report_probs(
    raws: list[list[str | None]], items: list[AuditItem]
) -> tuple[np.ndarray, int, int]:
    """Per-item report probability, plus (parsed, total) counts. NaN where nothing parsed."""
    probs, parsed, total = [], 0, 0
    for row, item in zip(raws, items, strict=True):
        flags = []
        for raw in row:
            total += 1
            audit = parse_audit(raw, len(item.steps)) if raw is not None else None
            if audit is not None:
                parsed += 1
                flags.append(audit.reported_error)
        probs.append(float(np.mean(flags)) if flags else np.nan)
    return np.array(probs), parsed, total


def localization(raws: list[list[str | None]], faulty: list[tuple[AuditItem, int]]) -> dict:
    """Exact and +/-1 step agreement with the human first-error label, over reported errors."""
    exact = near = usable = 0
    for row, (item, true_step) in zip(raws, faulty, strict=True):
        for raw in row:
            audit = parse_audit(raw, len(item.steps)) if raw is not None else None
            if audit is None or not audit.localization_usable:
                continue
            usable += 1
            # ProcessBench `label` is a 0-based step index; our prompt numbers from 1.
            delta = abs(audit.first_error_step - (true_step + 1))
            exact += delta == 0
            near += delta <= 1
    return {
        "localizable_samples": usable,
        "exact_step_accuracy": exact / usable if usable else None,
        "within_one_step_accuracy": near / usable if usable else None,
    }


def verdicts_at_zero(raws: list[list[str | None]], items: list[AuditItem]) -> list[bool | None]:
    """One greedy verdict per item, or None if it did not parse."""
    out: list[bool | None] = []
    for row, item in zip(raws, items, strict=True):
        audit = parse_audit(row[0], len(item.steps)) if row and row[0] is not None else None
        out.append(None if audit is None else audit.reported_error)
    return out


async def run_endpoint(ep: backend.Endpoint, clean, faulty, n_samples, temperature, concurrency):
    """All R0 generations for one model: sampled arm plus two greedy passes."""
    prompts_clean = [build_audit_messages(i, "R0", "F1") for i in clean]
    prompts_faulty = [build_audit_messages(i, "R0", "F1") for i, _ in faulty]

    async with backend.client() as http:
        common = dict(max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA, concurrency=concurrency)
        t0 = time.time()
        sampled_clean = await backend.map_prompts(
            http, ep, prompts_clean, n=n_samples, temperature=temperature, **common)
        sampled_faulty = await backend.map_prompts(
            http, ep, prompts_faulty, n=n_samples, temperature=temperature, **common)
        # Two greedy passes over the same prompts. Pass B runs at a different concurrency so
        # the scheduler forms different batches -- that is the whole point of the probe.
        greedy_a = await backend.map_prompts(
            http, ep, prompts_clean, n=1, temperature=0.0,
            max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA, concurrency=concurrency)
        greedy_b = await backend.map_prompts(
            http, ep, prompts_clean, n=1, temperature=0.0,
            max_tokens=MAX_TOKENS, schema=AUDIT_SCHEMA, concurrency=max(1, concurrency // 8))
        elapsed = time.time() - t0

    p_clean, parsed_c, total_c = report_probs(sampled_clean, clean)
    p_faulty, parsed_f, total_f = report_probs(sampled_faulty, [i for i, _ in faulty])
    va, vb = verdicts_at_zero(greedy_a, clean), verdicts_at_zero(greedy_b, clean)
    both = [(a, b) for a, b in zip(va, vb, strict=True) if a is not None and b is not None]

    return {
        "endpoint": ep.label,
        "model": ep.model,
        "revision": ep.revision,
        "n_clean": len(clean),
        "n_faulty": len(faulty),
        "n_samples": n_samples,
        "temperature": temperature,
        "elapsed_s": round(elapsed, 1),
        "baseline_far": far(p_clean),
        "instability_index_n1": instability_index(p_clean),
        "detection_rate_faulty": far(p_faulty),
        "parse_ok_rate_clean": parsed_c / total_c if total_c else None,
        "parse_ok_rate_faulty": parsed_f / total_f if total_f else None,
        "greedy_compared": len(both),
        "greedy_disagreement_rate": (
            sum(a != b for a, b in both) / len(both) if both else None
        ),
        **localization(sampled_faulty, faulty),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=int, default=300)
    ap.add_argument("--faulty", type=int, default=300)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=str(artefact("bakeoff_results.json")))
    args = ap.parse_args()

    clean, faulty = load_items(args.clean, args.faulty)
    print(f"loaded {len(clean)} clean seeds, {len(faulty)} natural faulty traces", flush=True)

    results = []
    for ep in ENDPOINTS:
        print(f"--- {ep.label} ---", flush=True)
        results.append(await run_endpoint(
            ep, clean, faulty, args.samples, args.temperature, args.concurrency))
        print(json.dumps(results[-1], indent=2), flush=True)

    Path(args.out).write_text(json.dumps(
        {"seed": SEED, "condition": "R0", "family": "F1", "results": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
