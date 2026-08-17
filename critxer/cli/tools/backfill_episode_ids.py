#!/usr/bin/env python
"""Backfill `episode_ids` into R4 cells generated before they were persisted.

`run_r4.py` now records which frozen episode each target was paired to, because every R4 contrast
has to resample episodes as whole units (see `critxer.core.resample`) and the cells already on
disk do not say
which episode is which. Regenerating 60+ cells to recover an assignment that was deterministic all
along would be days of GPU for no new measurement, so this reconstructs it instead.

The reconstruction is exact, not approximate: `paired()` assigns target *i* the episode at
``i % len(pool)`` where ``pool`` is the episode file's own order. This script re-reads that file, so
the only assumption is that the file has not been regenerated since the cell was run --- which it
verifies as far as it can by checking the pool size against the cell's item count.

Cells are patched in place with the key added and nothing else touched. `R0` is skipped: it has no
episode. `AX` draws from the incorrect-verdict pool, so it gets its own file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from critxer.core.paths import data_root, run_dir
from critxer.core.resample import episode_ids_for

# Cells drawn from the ordinary warmup pool. AF is included: its filler is keyed by the warmup
# episode's item id, so it inherits exactly the same target-to-episode assignment.
MAIN_POOL_CELLS = ("AS", "AO", "US", "UO", "AF", "AV")
INCORRECT_POOL_CELLS = ("AX",)


def pool_ids(path: Path) -> list[str]:
    return [e["item_id"] for e in json.loads(path.read_text())["episodes"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(run_dir("r4")))
    ap.add_argument("--episodes-dir", default=str(data_root()))
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    pools: dict[tuple[str, str], list[str]] = {}
    patched = skipped = already = 0
    for path in sorted(Path(args.dir).glob("*.json")):
        rec = json.loads(path.read_text())
        cell, model = rec["condition"], rec["auditor"]
        if cell == "R0":
            skipped += 1
            continue
        if "episode_ids" in rec:
            already += 1
            continue
        if cell in MAIN_POOL_CELLS:
            stem = f"warmup_episodes_{model}_F1.json"
        elif cell in INCORRECT_POOL_CELLS:
            stem = f"warmup_incorrect_{model}_F1.json"
        else:
            raise SystemExit(f"{path.name}: unknown cell {cell!r}; refusing to guess its pool")

        key = (model, stem)
        if key not in pools:
            src = Path(args.episodes_dir) / stem
            if not src.exists():
                raise SystemExit(f"{path.name} needs {src}, which is missing")
            pools[key] = pool_ids(src)
        pool = pools[key]

        n = rec["n_items"]
        if len(pool) > n:
            raise SystemExit(
                f"{path.name}: pool has {len(pool)} episodes for {n} targets, so the cycling "
                "assumption does not hold and the assignment cannot be reconstructed"
            )
        rec["episode_ids"] = [pool[i] for i in episode_ids_for(n, len(pool))]
        rec["episode_ids_reconstructed"] = True
        if not args.dry_run:
            path.write_text(json.dumps(rec, indent=1))
        patched += 1
        print(f"  {path.name}: {len(pool)} episodes over {n} targets "
              f"({n / len(pool):.1f} targets per episode)")

    verb = "would patch" if args.dry_run else "patched"
    print(f"\n{verb} {patched}; {already} already had it; {skipped} R0 cells have no episode")


if __name__ == "__main__":
    main()
