#!/usr/bin/env python
"""The R4 2x2 run. Secondary to the ladder.

Four cells crossing placement (assistant turn / user turn) with attribution (self / peer). This is
labels this a *prospective replication* of arXiv:2603.04582 plus its missing assistant-turn /
peer-label cell: three of the four cells are theirs, only R4-AO is new.

Each target item is paired with one **frozen** warmup episode, and the same episode is used for all
four of that item's cells -- matching is within-item, so a cell-to-cell difference cannot come from
the episode. Pairing is by position in the sorted target list modulo the episode pool, which is
deterministic and spreads the 50 episodes evenly rather than letting a few dominate.

R0 is re-measured here on the *same* target subset rather than reused from the ladder run: R4 uses
half the clean arm, and comparing a subset's cells against the full arm's R0 would confound
the 2x2 with which items happen to be in the subset.
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
    AuditItem,
    audit_records,
    build_audit_messages,
    parse_audit,
)
from critxer.core.backend import parse_endpoint
from critxer.core.metrics import far, instability_index
from critxer.core.paths import artefact, run_dir
from critxer.core.r4 import (
    CELLS,
    FillerEpisode,
    WarmupEpisode,
    build_audit_only_messages,
    build_filler_messages,
    build_neutral_messages,
    build_r4_messages,
    episode_fingerprint,
    episode_items,
    neutral_map,
    select_cells,
)
from critxer.core.resample import episode_ids_for

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
MAX_TOKENS = 512
SEED = 20260805




def load_items(ids: set[str]) -> tuple[dict[str, AuditItem], dict[str, str]]:
    """Items by id, plus each id's ProcessBench source -- the source is needed to stratify."""
    rows: dict[str, dict] = {}
    source: dict[str, str] = {}
    for split in SPLITS:
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=str(artefact("processbench")))
        found = {r["id"]: r for r in ds if r["id"] in ids}
        rows.update(found)
        source.update({i: split for i in found})
    missing = ids - rows.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} ids not in ProcessBench, e.g. {sorted(missing)[:3]}")
    return ({i: AuditItem(i, rows[i]["problem"], list(rows[i]["steps"])) for i in ids}, source)


def proportional_subset(ids: list[str], source: dict[str, str], fraction: float) -> list[str]:
    """A seeded subset that preserves the arm's per-source mix.

    Neither obvious alternative works. `ids[:half]` is alphabetical and ProcessBench ids sort by
    source, so it drops a source entirely; `core.allocation.allocate` equalises sources instead,
    over-weighting the easiest one and making R4 effect sizes incomparable to the ladder's.
    Proportional keeps both properties: every source present, in the arm's own proportions.
    """
    rng = np.random.default_rng(SEED)
    by_source: dict[str, list[str]] = {}
    for i in ids:
        by_source.setdefault(source[i], []).append(i)
    out: list[str] = []
    for s in sorted(by_source):
        pool = by_source[s]
        take = round(len(pool) * fraction)
        out += [pool[j] for j in rng.permutation(len(pool))[:take]]
    return sorted(out)


def summarise(raws, items) -> dict:
    probs, flags_out, parsed, total = [], [], 0, 0
    for row, item in zip(raws, items, strict=True):
        flags = []
        for raw in row:
            total += 1
            audit = parse_audit(raw, len(item.steps)) if raw is not None else None
            if audit is None:
                continue
            parsed += 1
            flags.append(audit.reported_error)
        probs.append(float(np.mean(flags)) if flags else np.nan)
        flags_out.append([int(f) for f in flags])
    arr = np.array(probs)
    return {"far": far(arr), "instability_n1": instability_index(arr),
            "parse_ok_rate": parsed / total if total else 0.0,
            "per_item_probs": arr.tolist(), "per_item_flags": flags_out}


def episode_pool(blob: dict, label: str) -> list[WarmupEpisode]:
    """Frozen episodes as `WarmupEpisode`s, in the file's own order.

    Injected-trace episodes cannot be looked up in ProcessBench, because the corrupted steps exist
    only in the injection set. Records carry ``problem`` and ``steps`` inline where available, so
    prefer those and fall back to the lookup.
    """
    if blob["auditor"] != label:
        raise SystemExit(
            f"episodes are {blob['auditor']!r} but the endpoint is {label!r}; the 2x2 requires "
            "the episode be the auditor's own work"
        )
    eps = blob["episodes"]
    need = {e["item_id"] for e in eps if "steps" not in e}
    fetched, _ = load_items(need) if need else ({}, {})
    return [WarmupEpisode(item, e["audit"], e["repair"])
            for item, e in zip(episode_items(eps, fetched), eps, strict=True)]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, metavar="LABEL=MODEL@URL")
    ap.add_argument("--warmup", required=True, help="warmup_episodes_*.json for this model")
    ap.add_argument("--filler", default="",
                    help="filler_*.json; adds the AF cell controlling for context per se")
    ap.add_argument("--audit-only", action="store_true", default=False,
                    help="add the AV cell: the same episode with the repair turns removed, so "
                         "AS - AV isolates the repair from the verdict that precedes it")
    ap.add_argument("--neutral", default="",
                    help="neutral_*.json; adds the AN cell -- the audit followed by a "
                         "length-matched NON-repair continuation, which is what makes the repair "
                         "contrast identified (AS - AN rather than AS - AV)")
    ap.add_argument("--warmup-incorrect", default="",
                    help="episodes built on faulty traces; adds the AX cell, a full audit->repair "
                         "whose verdict is 'incorrect' rather than the usual 'correct'")
    ap.add_argument("--neutral-incorrect", default="",
                    help="neutral continuations for the --warmup-incorrect pool; adds the AXN "
                         "cell, which makes AX - AXN the contribution of a GENUINE correction to "
                         "a GENUINE fault -- the only contrast here in which the repair repairs "
                         "something. Requires --warmup-incorrect")
    ap.add_argument("--family", default="F1")
    ap.add_argument("--capture-audits", action="store_true", default=False,
                    help="also write <cell>__audits.json with every generation's error_type, "
                         "evidence and claimed step. The summaries keep only flags, so without "
                         "this a false alarm cannot be told from a stricter-but-defensible "
                         "standard by anyone, us included")
    ap.add_argument("--cells", default="",
                    help="comma-separated subset to run, e.g. 'R0,AS,AF' for the reasoning arm; "
                         "default runs every cell this run's inputs support. Raises on a cell "
                         "whose input file was not supplied rather than dropping it")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help=f"output budget per generation (default {MAX_TOKENS}). With --thinking "
                         "this must be several thousand: a trace that hits the cap returns "
                         "content=null for every choice, which reads downstream as a parse "
                         "failure and not as truncation")
    ap.add_argument("--thinking", action="store_true", default=False,
                    help="re-enable reasoning per request, overriding the server default that "
                         "serve.sh sets for the whole study. Robustness arm only -- traces of "
                         "variable length break the identical-budget invariant the study rests on")
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="share of the clean arm used as R4 targets (secondary gets half)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--allocation", default=str(artefact("allocation.json")))
    ap.add_argument("--outdir", default=str(run_dir("r4")))
    args = ap.parse_args()

    ep = parse_endpoint(args.endpoint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clean = sorted(json.loads(Path(args.allocation).read_text())["arms"]["clean"])
    clean_items, source = load_items(set(clean))
    # Stratified, NOT the first half of the sorted ids. ProcessBench ids sort by source, so
    # `clean[:464]` drew 315 math, 126 gsm8k, 23 olympiadbench and *zero* omnimath -- the arm's
    # own R0 FAR was 0.132 on that subset against 0.219 on the full arm. Re-measuring R0 inside
    # the subset keeps the 2x2 internally valid, but the subset still has to represent the arm.
    targets = proportional_subset(clean, source, args.fraction)
    pool = episode_pool(json.loads(Path(args.warmup).read_text()), ep.label)
    target_items = {i: clean_items[i] for i in targets}

    def paired(episodes: list[WarmupEpisode]) -> list[tuple[AuditItem, WarmupEpisode]]:
        """Deterministic, even pairing: target i gets episode i mod len(episodes)."""
        overlap = set(targets) & {e.item.item_id for e in episodes}
        if overlap:
            raise SystemExit(
                f"{len(overlap)} targets are also episode items, e.g. {sorted(overlap)[:3]}")
        idx = episode_ids_for(len(targets), len(episodes))
        return [(target_items[t], episodes[i]) for t, i in zip(targets, idx, strict=True)]

    pairs = paired(pool)
    fingerprint = episode_fingerprint(pool)
    print(f"R4 targets: {len(pairs)}; episodes: {len(pool)}; "
          f"targets per episode: {len(pairs) / len(pool):.1f}", flush=True)

    # AX's episodes are a different pool, so its pairing is built separately -- it is compared to
    # AS across the same targets, not episode-for-episode.
    incorrect_pairs: list[tuple[AuditItem, WarmupEpisode]] = []
    pool_x: list[WarmupEpisode] = []
    if args.warmup_incorrect:
        blob_x = json.loads(Path(args.warmup_incorrect).read_text())
        pool_x = episode_pool(blob_x, ep.label)
        still_correct = [e for e in pool_x if '"incorrect"' not in e.audit]
        if still_correct:
            raise SystemExit(
                f"{len(still_correct)} of {len(pool_x)} --warmup-incorrect episodes do not report "
                "an error; regenerate that set with --require-error or the AX contrast is not a "
                "verdict contrast at all"
            )
        incorrect_pairs = paired(pool_x)
        fingerprint_x = episode_fingerprint(pool_x)
        print(f"incorrect-verdict episodes: {len(pool_x)} -> AX cell enabled", flush=True)
    elif args.neutral_incorrect:
        raise SystemExit("--neutral-incorrect needs --warmup-incorrect: the AXN cell is that "
                         "pool's episodes with the correction replaced, so without the pool there "
                         "is nothing for it to be the control for")

    fillers = {}
    if args.filler:
        blob_f = json.loads(Path(args.filler).read_text())
        if blob_f["auditor"] != ep.label:
            raise SystemExit(f"filler set is {blob_f['auditor']!r}, endpoint is {ep.label!r}")
        fillers = {f["item_id"]: FillerEpisode(f["request"], f["response"])
                   for f in blob_f["fillers"]}
        print(f"filler episodes: {len(fillers)} -> AF cell enabled", flush=True)

    neutrals = {}
    if args.neutral:
        neutrals = neutral_map(
            json.loads(Path(args.neutral).read_text()), ep.label, [w for _, w in pairs], "AN")
        print(f"neutral continuations: {len(neutrals)} -> AN cell enabled", flush=True)

    neutrals_x = {}
    if args.neutral_incorrect:
        neutrals_x = neutral_map(
            json.loads(Path(args.neutral_incorrect).read_text()), ep.label, pool_x, "AXN")
        print(f"neutral continuations (incorrect pool): {len(neutrals_x)} -> AXN cell enabled",
              flush=True)

    # AF/AV/AX/AN/AXN are not cells of the 2x2 -- they are the controls that make it readable, so
    # they run last and stay out of CELLS. AF removes the episode; AV removes only its repair; AN
    # replaces the repair with an inert continuation of matched length; AX keeps the whole structure
    # and flips the verdict; AXN is AN over AX's pool, which is the one place where the thing being
    # removed is a real correction to a real fault.
    cells = select_cells(args.cells, ("R0", *CELLS, *(("AF",) if fillers else ()),
                                     *(("AV",) if args.audit_only else ()),
                                     *(("AN",) if neutrals else ()),
                                     *(("AX",) if incorrect_pairs else ()),
                                     *(("AXN",) if neutrals_x else ())))
    if args.thinking:
        print(f"THINKING ENABLED, max_tokens={args.max_tokens}: robustness arm, not comparable to "
              "the main study's R0 (thinking and budget both differ)", flush=True)

    async with backend.client() as http:
        for cell in cells:
            out = outdir / f"{ep.label}__{cell}__{args.family}.json"
            if out.exists():
                print(f"  skip {out.name} (checkpoint exists)", flush=True)
                continue
            on_incorrect = cell in ("AX", "AXN")
            used = incorrect_pairs if on_incorrect else pairs
            if cell == "R0":
                prompts = [build_audit_messages(t, "R0", args.family) for t, _ in used]
            elif cell == "AF":
                # Same episode-to-target pairing, so AF and the cells share their prior-exchange
                # assignment and differ only in what that exchange was.
                prompts = [build_filler_messages(t, fillers[w.item.item_id], args.family)
                           for t, w in used]
            elif cell == "AV":
                prompts = [build_audit_only_messages(t, w, args.family) for t, w in used]
            elif cell in ("AN", "AXN"):
                src = neutrals_x if cell == "AXN" else neutrals
                prompts = [build_neutral_messages(t, w, src[w.item.item_id], args.family)
                           for t, w in used]
            else:
                # AX is an AS cell over the incorrect-verdict pool: same placement, same
                # attribution, same structure, opposite verdict.
                structure = "AS" if cell == "AX" else cell
                prompts = [build_r4_messages(t, w, structure, args.family) for t, w in used]
            # One list per cell, appended to as requests complete, so the order is arrival order
            # rather than target order. Only the distribution is reported, never a per-item join.
            usage: list[dict] = []
            raws = await backend.map_prompts(
                http, ep, prompts, concurrency=args.concurrency, n=args.samples,
                temperature=0.7, max_tokens=args.max_tokens, schema=AUDIT_SCHEMA, seed=1,
                thinking=args.thinking, usage_sink=usage)
            # Per-CHOICE tokens: the server reports `completion_tokens` summed over a request's n
            # choices, so a raw p95 read against max_tokens looked like a cap being exceeded. And
            # truncation is counted from finish_reason, never from token arithmetic.
            per_choice = [u["completion_tokens"] / u["n"] for u in usage if u.get("n")]
            truncated = sum(r == "length" for u in usage for r in u.get("finish_reasons", []))
            n_choices = sum(len(u.get("finish_reasons", [])) for u in usage)
            tokens = {
                "mean_per_choice": float(np.mean(per_choice)),
                "p50_per_choice": float(np.percentile(per_choice, 50)),
                "p95_per_choice": float(np.percentile(per_choice, 95)),
                "max_per_choice": float(max(per_choice)),
                "truncated_choices": truncated,
                "n_choices": n_choices,
                "n_requests": len(usage),
            } if per_choice else {}
            summary = summarise(raws, [t for t, _ in used])
            rec = {"auditor": ep.label, "model": ep.model, "condition": cell,
                   "family": args.family, "samples": args.samples,
                   "max_tokens": args.max_tokens, "thinking": args.thinking,
                   # Recorded for every run, not just the reasoning arm, so the two are comparable
                   # on budget as well as on FAR.
                   **({"completion_tokens": tokens} if tokens else {}),
                   "n_items": len(used), "item_ids": [t.item_id for t, _ in used],
                   # Which frozen episode each target was paired to. Contrasts must resample
                   # episodes as whole units : the pool is cycled, so each episode
                   # is reused for ~9 targets and item-only bootstrapping fakes precision.
                   # Absent for R0, which has no episode.
                   **({} if cell == "R0" else
                      {"episode_ids": [w.item.item_id for _, w in used],
                       "episode_fingerprint": fingerprint_x if on_incorrect else fingerprint}),
                   **summary}
            out.write_text(json.dumps(rec, indent=1))
            if args.capture_audits:
                eva = out.with_name(f"{out.stem}__audits.json")
                eva.write_text(json.dumps(
                    {"auditor": ep.label, "condition": cell, "family": args.family,
                     "records": audit_records(raws, [t for t, _ in used])}, indent=1))

            trunc = int(tokens.get("truncated_choices", 0))
            parse_ok = float(summary["parse_ok_rate"])
            budget = ""
            if tokens:
                budget = f" tok/choice_p95={tokens['p95_per_choice']:.0f}"
                if trunc:
                    budget += f" TRUNCATED={trunc}/{tokens['n_choices']}"
            print(f"  {ep.label} {cell}: FAR={rec['far']:.4f} N1={rec['instability_n1']:.4f} "
                  f"parse={parse_ok:.3f}{budget} -> {out.name}", flush=True)
            # The cell is on disk before this fires, so a raise costs the diagnosis and not the run.
            # A truncated cell is the dangerous shape: every choice comes back content=null, which
            # `summarise` counts as a parse failure and averages into a FAR of nan-or-plausible
            # rather than an error. Nothing legitimate here has run below 0.998.
            if parse_ok < 0.5:
                raise SystemExit(
                    f"{ep.label}/{cell}: only {parse_ok:.1%} of generations parsed"
                    + (f", and {trunc} of {tokens['n_choices']} choices hit max_tokens="
                       f"{args.max_tokens} -- raise the budget" if trunc else
                       " -- inspect the raw output before trusting any later cell")
                )

    print("\ndone. analyse with `critxer analyse-r4` for the factorial and its controls, or\n"
          "      `critxer analyse-ladder --dir <outdir>` for the cell-vs-R0 view")


if __name__ == "__main__":
    asyncio.run(main())
