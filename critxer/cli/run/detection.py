#!/usr/bin/env python
"""Run conditions on the labelled-**incorrect** arm and score detection.

A false-alarm rate on verified-correct traces alone cannot separate an auditor that became uniformly
more lenient from one that became better calibrated. This draws a stratified arm from ProcessBench's
expert-annotated faulty traces, runs the same conditions with the same prompts and schema, and
records what the clean arm cannot: detection rate, first-error localisation, and enough to compute
d' and criterion against the matching clean-arm condition.

It also persists **the reported step per item** (the mode over samples that flagged an error), so
localisation can be scored, and **the mean confidence per item**, so calibration can be.

The arm is disjoint from every other arm for free: `allocation.json` draws only from traces whose
label marks every step correct, and this draws only from traces where it does not.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

from critxer.core import backend
from critxer.core.audit import AUDIT_SCHEMA, AuditItem, build_audit_messages, parse_audit
from critxer.core.backend import parse_endpoint
from critxer.core.paths import artefact, run_dir
from critxer.core.r4 import (
    CELLS,
    FillerEpisode,
    WarmupEpisode,
    build_audit_only_messages,
    build_filler_messages,
    episode_fingerprint,
    episode_items,
)
from critxer.core.r4 import build_r4_messages as build_cell
from critxer.core.resample import episode_ids_for

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
MAX_TOKENS = 512
SEED = 20260805




def build_arm(n: int, allocation: Path) -> list[dict]:
    """A stratified sample of labelled-incorrect traces, matching the clean arm's source mix.

    Proportional rather than equalised, for the same reason `run.r4.proportional_subset` is: gsm8k
    is by far the easiest source, so equalising would put detection rates on a different footing
    from the FAR they are compared against.
    """
    clean = set(json.loads(allocation.read_text())["arms"]["clean"])
    rows: dict[str, list[dict]] = {}
    clean_mix: Counter[str] = Counter()
    for split in SPLITS:
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        rows[split] = [r for r in ds if r["label"] != -1]
        clean_mix[split] = sum(1 for r in ds if r["id"] in clean)

    total_clean = sum(clean_mix.values())
    rng = np.random.default_rng(SEED)
    out: list[dict] = []
    for split in SPLITS:
        want = round(n * clean_mix[split] / total_clean)
        pool = rows[split]
        if want > len(pool):
            raise SystemExit(f"{split}: want {want} incorrect traces, only {len(pool)} exist")
        for j in rng.permutation(len(pool))[:want]:
            r = pool[j]
            out.append({"id": r["id"], "split": split, "problem": r["problem"],
                        "steps": list(r["steps"]), "gold_label": int(r["label"])})
    out.sort(key=lambda r: r["id"])
    print(f"incorrect arm: {len(out)} traces "
          f"{dict(Counter(r['split'] for r in out))}", flush=True)
    return out


def summarise(raws, items: list[AuditItem], gold: list[int]) -> dict:
    """Per-item detection probability, modal reported step, and mean confidence."""
    probs, steps, confs, flags_out, parsed, total = [], [], [], [], 0, 0
    for row, item in zip(raws, items, strict=True):
        flags, reported, cs = [], [], []
        for raw in row:
            total += 1
            audit = parse_audit(raw, len(item.steps)) if raw is not None else None
            if audit is None:
                continue
            parsed += 1
            flags.append(audit.reported_error)
            cs.append(audit.confidence)
            if audit.first_error_step is not None:
                reported.append(audit.first_error_step)
        probs.append(float(np.mean(flags)) if flags else np.nan)
        confs.append(float(np.mean(cs)) if cs else np.nan)
        # Mode over the samples that named a step. One number per item keeps localisation on the
        # same item-level footing as every other metric here.
        steps.append(int(Counter(reported).most_common(1)[0][0]) if reported else None)
        flags_out.append([int(f) for f in flags])
    arr = np.array(probs)
    return {"detection": float(np.nanmean(arr)), "parse_ok_rate": parsed / total if total else 0.0,
            "per_item_probs": arr.tolist(), "per_item_flags": flags_out,
            "per_item_steps": steps, "per_item_confidence": confs, "gold_labels": gold}


def build_parser() -> argparse.ArgumentParser:
    """Module-level so the reasoning arm's two flags are testable without a served model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, metavar="LABEL=MODEL@URL")
    ap.add_argument("--conditions", default="R0,R2,R3,R3u",
                    help="ladder conditions to run on the incorrect arm")
    ap.add_argument("--r4-cells", default="",
                    help="R4 cells to also run; needs --warmup, plus --filler for AF")
    ap.add_argument("--warmup", default="")
    ap.add_argument("--filler", default="")
    ap.add_argument("--family", default="F1")
    ap.add_argument("--n", type=int, default=929, help="arm size; default matches the clean arm")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help=f"output budget per generation (default {MAX_TOKENS}). With --thinking "
                         "this must be several thousand: a trace that hits the cap returns "
                         "content=null for every choice, which reads downstream as a parse "
                         "failure and not as truncation")
    ap.add_argument("--thinking", action="store_true", default=False,
                    help="re-enable reasoning per request, overriding the server default. "
                         "Robustness arm only, and never comparable to the main detection arm: "
                         "thinking and budget both differ, so only within-arm contrasts hold")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--allocation", default=str(artefact("allocation.json")))
    ap.add_argument("--arm-out", default=str(artefact("incorrect_arm.json")))
    ap.add_argument("--outdir", default=str(run_dir("detection")))
    return ap


async def main() -> None:
    args = build_parser().parse_args()

    ep = parse_endpoint(args.endpoint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    arm_path = Path(args.arm_out)
    if arm_path.exists():
        arm = json.loads(arm_path.read_text())["traces"]
        print(f"reusing arm from {arm_path} ({len(arm)} traces)", flush=True)
    else:
        arm = build_arm(args.n, Path(args.allocation))
        arm_path.write_text(json.dumps({"n": len(arm), "seed": SEED, "traces": arm}, indent=1))
        print(f"wrote {arm_path}", flush=True)

    items = [AuditItem(r["id"], r["problem"], r["steps"]) for r in arm]
    gold = [r["gold_label"] for r in arm]

    # R4 cells need the same frozen episodes the clean-arm run used, so the two are comparable.
    episodes, fillers = [], {}
    cells = [c for c in args.r4_cells.split(",") if c]
    if cells:
        blob = json.loads(Path(args.warmup).read_text())
        if blob["auditor"] != ep.label:
            raise SystemExit(f"warmup episodes are {blob['auditor']!r}, endpoint is {ep.label!r}")
        warm = blob["episodes"]
        ids = {e["item_id"] for e in warm}
        lookup: dict[str, AuditItem] = {}
        for split in SPLITS:
            ds = load_dataset("Qwen/ProcessBench", split=split,
                              cache_dir=str(artefact("processbench")))
            lookup.update({r["id"]: AuditItem(r["id"], r["problem"], list(r["steps"]))
                           for r in ds if r["id"] in ids})
        episodes = [WarmupEpisode(item, e["audit"], e["repair"])
                    for item, e in zip(episode_items(warm, lookup), warm, strict=True)]
        if args.filler:
            fb = json.loads(Path(args.filler).read_text())
            fillers = {f["item_id"]: FillerEpisode(f["request"], f["response"])
                       for f in fb["fillers"]}
        fingerprint = episode_fingerprint(episodes)
        print(f"episodes: {len(episodes)}; fillers: {len(fillers)}; "
              f"fingerprint {fingerprint}", flush=True)

    conditions = [c for c in args.conditions.split(",") if c]
    async with backend.client() as http:
        for cond in [*conditions, *cells]:
            out = outdir / f"{ep.label}__{cond}__{args.family}.json"
            if out.exists():
                print(f"  skip {out.name} (checkpoint exists)", flush=True)
                continue
            ep_ids = None
            if cond in conditions:
                prompts = [build_audit_messages(i, cond, args.family) for i in items]
            else:
                if not episodes:
                    raise SystemExit(f"cell {cond} needs --warmup")
                paired = [episodes[k]
                          for k in episode_ids_for(len(items), len(episodes))]
                ep_ids = [w.item.item_id for w in paired]
                if cond == "AF":
                    prompts = [build_filler_messages(i, fillers[w.item.item_id], args.family)
                               for i, w in zip(items, paired, strict=True)]
                elif cond == "AV":
                    prompts = [build_audit_only_messages(i, w, args.family)
                               for i, w in zip(items, paired, strict=True)]
                elif cond in CELLS:
                    prompts = [build_cell(i, w, cond, args.family)
                               for i, w in zip(items, paired, strict=True)]
                else:
                    raise SystemExit(f"unknown cell {cond!r}")
            usage: list[dict] = []
            raws = await backend.map_prompts(
                http, ep, prompts, concurrency=args.concurrency, n=args.samples,
                temperature=0.7, max_tokens=args.max_tokens, schema=AUDIT_SCHEMA, seed=1,
                thinking=args.thinking, usage_sink=usage)
            # Per-CHOICE tokens, and truncation counted from finish_reason rather than from token
            # arithmetic. Without this a reasoning-enabled run cannot be told apart from a
            # budget-starved one after the fact: both show only a depressed parse rate.
            per_choice = [u["completion_tokens"] / u["n"] for u in usage if u.get("n")]
            truncated = sum(r == "length" for u in usage for r in u.get("finish_reasons", []))
            tokens = {
                "mean_per_choice": float(np.mean(per_choice)),
                "p50_per_choice": float(np.percentile(per_choice, 50)),
                "p95_per_choice": float(np.percentile(per_choice, 95)),
                "max_per_choice": float(max(per_choice)),
                "truncated_choices": truncated,
                "n_choices": sum(len(u.get("finish_reasons", [])) for u in usage),
                "n_requests": len(usage),
            } if per_choice else {}
            rec = {"auditor": ep.label, "model": ep.model, "condition": cond, "arm": "incorrect",
                   "family": args.family, "samples": args.samples, "n_items": len(items),
                   # Stamped on every record, not just the reasoning arm: a thinking-enabled cell
                   # silently joined to the main arm is a two-way confound that nothing downstream
                   # can detect after the fact.
                   "max_tokens": args.max_tokens, "thinking": args.thinking,
                   **({"completion_tokens": tokens} if tokens else {}),
                   "item_ids": [i.item_id for i in items],
                   **({"episode_ids": ep_ids, "episode_fingerprint": fingerprint}
                      if ep_ids else {}),
                   **summarise(raws, items, gold)}
            out.write_text(json.dumps(rec, indent=1))
            print(f"  {ep.label} {cond}: detection={rec['detection']:.4f} "
                  f"parse={rec['parse_ok_rate']:.3f} -> {out.name}", flush=True)

    # Reported so a reader can check the R4 cells' clustering without re-deriving the pairing.
    if cells and episodes:
        print(f"\nR4 cells on this arm reuse {min(len(items), len(episodes))}"
              f" episodes over {len(items)} targets", flush=True)
    print("\ndone. score with `critxer analyse-detection`")


if __name__ == "__main__":
    asyncio.run(main())
