#!/usr/bin/env python
"""Score the incorrect arm against the clean arm: threshold shift or better discrimination?

Every FAR reduction in this study has three possible readings -- better calibration, uniform
leniency, or genuinely improved discrimination -- and the clean arm alone cannot tell them apart.
This joins each condition's clean-arm FAR to its incorrect-arm detection rate and reports the pair
that separates them: **d'**, which responds to discrimination and is invariant to a pure threshold
shift, and **criterion c**, which is the reverse.

The reading is mechanical, and stated here so it cannot be chosen after the fact:

* FAR down, detection flat, d' up, c up  -> the auditor got better. The effect is good news.
* FAR down, detection down, d' flat, c up -> a pure threshold shift. The effect is leniency.
* FAR down, detection up                 -> discrimination improved outright.

Localisation accuracy is reported alongside because a condition can keep its detection rate while
getting worse at saying *where* the error is, which would be invisible in d'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from critxer.core.audit import LADDER
from critxer.core.detection import (
    balanced_accuracy,
    criterion,
    expected_calibration_error,
    localisation_accuracy,
    sensitivity_index,
)
from critxer.core.metrics import far as far_of
from critxer.core.multiplicity import holm, short_family_message
from critxer.core.paths import artefact, run_dir
from critxer.core.resample import (
    Streams,
    cluster_bootstrap,
    cluster_indices,
    interval_from_reps,
    paired_bootstrap,
)

SEED = 20260805
N_BOOT = 20000
# The d'/criterion replicates are a Python loop over two independently resampled arms rather than
# one vectorised paired mean, so they were run at 2,000 while everything else ran at 20,000. That
# is enough resolution to place an effect but not to place a p-value near a Holm threshold: the
# 27B's AS contrast moved between 0.036 and 0.028 across runs. 20,000 costs about a minute.
N_BOOT_SDT = 20000

# The multiplicity families, declared here rather than inside main() so they read as part of the
# design. See the comment at their use in main() for why all three quantities are corrected.
# The ladder minus its baseline, derived rather than retyped: the copy here was also reordered,
# which is inert for a membership test and exactly why a drift would go unnoticed.
LADDER_RUNGS = tuple(c for c in LADDER if c != "R0")
PRIOR_CONTEXT_CELLS = ("AS", "AV", "AF")
QUANTITIES = ("delta_d_prime", "delta_criterion", "delta_balanced_accuracy")


def missing_baseline_message(model: str, family: str, conds: list[str], clean_dir: str) -> str:
    """Why a condition cannot be scored, and the command that would fix it.

    d' needs a false-alarm rate and a detection rate on the SAME item set, so every condition is
    baselined against the R0 in its own clean directory -- and the ladder was swept across wordings
    without re-measuring R0, so at F2-F5 that baseline does not exist. Without this it surfaced as a
    bare KeyError naming a dict key.
    """
    return (
        f"{model} at wording {family}: no R0 in {clean_dir}, so "
        f"{', '.join(conds)} cannot be scored -- d' needs a false-alarm rate on the same item set "
        f"as the detection rate. Generate it with:\n"
        f"    critxer run-ladder --endpoint '{model}=<model>@<url>' "
        f"--family {family} --conditions R0"
    )


def load(dirname: Path, family: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(dirname.glob(f"*__{family}.json")):
        rec = json.loads(path.read_text())
        out.setdefault(rec["auditor"], {})[rec["condition"]] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incorrect-dir", default=str(run_dir("detection")))
    ap.add_argument("--clean-dirs", default=f"{run_dir("ladder")},{run_dir("r4")}",
                    help="comma-separated dirs holding the matching clean-arm records")
    ap.add_argument("--family", default="F1")
    ap.add_argument("--out", default=str(artefact("detection_scored.json")))
    args = ap.parse_args()

    inc = load(Path(args.incorrect_dir), args.family)
    # The clean arm exists at two sizes -- the ladder's 929 items and R4's 465-target subset -- and
    # R0 exists in both. A condition must be compared against the R0 measured on *its own* item set,
    # or the noise-side rates come from different populations and the d' difference is meaningless.
    # So each condition keeps the directory it came from, and its baseline is that directory's R0.
    clean: dict[str, dict[str, dict]] = {}
    origin: dict[tuple[str, str], str] = {}
    # Per-directory R0, so a cell from r4 is baselined on r4's R0 and a rung on the ladder's.
    baselines: dict[tuple[str, str], dict] = {}
    for d in args.clean_dirs.split(","):
        for model, by_cond in load(Path(d), args.family).items():
            for c, r in by_cond.items():
                if c not in clean.get(model, {}):
                    clean.setdefault(model, {})[c] = r
                    origin[(model, c)] = d
            if "R0" in by_cond:
                baselines[(model, d)] = by_cond["R0"]

    stream = Streams(SEED)
    out: dict[str, dict] = {}
    for model, by_cond in sorted(inc.items()):
        rows: dict[str, dict] = {}
        for cond, rec in sorted(by_cond.items()):
            if model not in clean or cond not in clean[model]:
                print(f"skip {model}/{cond}: no clean-arm record to pair with")
                continue
            det_probs = np.array(rec["per_item_probs"], dtype=float)
            detection = float(np.nanmean(det_probs))
            clean_probs = np.array(clean[model][cond]["per_item_probs"], dtype=float)
            far = far_of(clean_probs)
            # Both counts exclude items whose every sample failed to parse. They feed the same
            # log-linear correction, and counting one arm's unparsed items and not the other's
            # applied a different correction to the two rates that d' differences.
            n_sig = int(np.sum(~np.isnan(det_probs)))
            n_noi = int(np.sum(~np.isnan(clean_probs)))
            conf = np.array(rec["per_item_confidence"], dtype=float)
            rows[cond] = {
                "detection": detection,
                "far": far,
                "n_signal": n_sig,
                "n_noise": n_noi,
                "d_prime": sensitivity_index(detection, far, n_sig, n_noi),
                "criterion_c": criterion(detection, far, n_sig, n_noi),
                "balanced_accuracy": balanced_accuracy(detection, far),
                "localisation": localisation_accuracy(rec["per_item_steps"], rec["gold_labels"]),
                # "Correct" for calibration on this arm means the model flagged the error, since
                # every trace here really is flawed.
                "ece": expected_calibration_error(conf, det_probs),
            }
        # d' and c are functions of two aggregate rates measured on two disjoint item sets, so a
        # difference in them needs its own interval: the point values alone cannot say whether a
        # 0.05 change in d' is a gain or noise. Items are resampled independently in each arm
        # (they are different traces) and paired within arm across conditions, so the item's own
        # propensity cancels.
        unscorable: dict[str, list[str]] = {}
        # This model's R0 on the incorrect arm, built once for both loops below.
        det0 = (np.array(by_cond["R0"]["per_item_probs"], dtype=float)
                if "R0" in rows else None)
        if det0 is not None:
            for cond, row in rows.items():
                if cond == "R0":
                    continue
                base_clean = baselines.get((model, origin[(model, cond)]))
                if base_clean is None:
                    unscorable.setdefault(origin[(model, cond)], []).append(cond)
                    continue
                far0 = np.array(base_clean["per_item_probs"], dtype=float)
                det1 = np.array(by_cond[cond]["per_item_probs"], dtype=float)
                far1 = np.array(clean[model][cond]["per_item_probs"], dtype=float)
                if far0.size != far1.size:
                    raise SystemExit(
                        f"{model}/{cond}: clean baseline has {far0.size} items and the condition "
                        f"{far1.size}; they are different item sets and cannot be compared"
                    )
                n_s, n_n = row["n_signal"], row["n_noise"]
                # A dedicated, deterministically derived stream per contrast: a shared RNG makes
                # each p-value depend on how many contrasts ran before it, which here straddled a
                # Holm threshold.
                sub = stream(model, cond)
                # Clustered on the frozen episode in whichever arm has one, which for an R4 cell
                # is BOTH: the incorrect arm cycles 50 episodes over 929 targets (18.6x reuse) and
                # the clean arm 50 over 465 (9.3x). Ladder rungs have no episode: item-only.
                sig_ep = by_cond[cond].get("episode_ids")
                noi_ep = clean[model][cond].get("episode_ids")
                sig_c = np.asarray(sig_ep) if sig_ep else None
                noi_c = np.asarray(noi_ep) if noi_ep else None
                dd, dc = np.empty(N_BOOT_SDT), np.empty(N_BOOT_SDT)
                db = np.empty(N_BOOT_SDT)
                for b in range(N_BOOT_SDT):
                    si = (cluster_indices(sig_c, sub) if sig_c is not None
                          else sub.integers(0, det0.size, det0.size))
                    ni = (cluster_indices(noi_c, sub) if noi_c is not None
                          else sub.integers(0, far0.size, far0.size))
                    d0, d1 = np.nanmean(det0[si]), np.nanmean(det1[si])
                    f0, f1 = np.nanmean(far0[ni]), np.nanmean(far1[ni])
                    dd[b] = (sensitivity_index(d1, f1, n_s, n_n)
                             - sensitivity_index(d0, f0, n_s, n_n))
                    dc[b] = criterion(d1, f1, n_s, n_n) - criterion(d0, f0, n_s, n_n)
                    db[b] = balanced_accuracy(d1, f1) - balanced_accuracy(d0, f0)
                # The baseline d' and c, recomputed from `base_clean` rather than read off
                # `rows["R0"]`. They are not the same number: `rows["R0"]` takes its false-alarm
                # rate from whichever clean directory supplied R0 first (the ladder's, 929 items)
                # while this interval is built on the condition's OWN directory (465 for an R4
                # cell). Using both differenced every R4 point estimate against one baseline and
                # its interval against another.
                far0_rate = far_of(far0)
                base_det = float(np.nanmean(det0))
                base_dp = sensitivity_index(base_det, far0_rate, n_s, n_n)
                base_c = criterion(base_det, far0_rate, n_s, n_n)
                # Balanced accuracy is differenced against the SAME baseline as d' and c. It is
                # the operating-point quantity a reader wants after a false-alarm reduction, and
                # without an interval the smallest real effect here cannot be told from zero.
                base_ba = balanced_accuracy(base_det, far0_rate)
                pairs = (("delta_d_prime", dd, row["d_prime"] - base_dp),
                         ("delta_criterion", dc, row["criterion_c"] - base_c),
                         ("delta_balanced_accuracy", db,
                          row["balanced_accuracy"] - base_ba))
                for name, reps, point in pairs:
                    # Same interval-and-p helper as every other contrast in the project. This site
                    # had its own, which centred on the replicate mean rather than the plug-in
                    # estimate and could disagree with the interval printed beside it.
                    row[name] = {
                        **interval_from_reps(reps, float(point), n_boot=N_BOOT_SDT),
                        "clustered_signal": sig_c is not None,
                        "clustered_noise": noi_c is not None,
                    }

        # Detection contrasts against this model's own R0, clustered on episode where one applies.
        if det0 is not None:
            for cond, row in rows.items():
                if cond == "R0":
                    continue
                d = np.array(by_cond[cond]["per_item_probs"], dtype=float) - det0
                eps = by_cond[cond].get("episode_ids")
                # Its own stream, like every other contrast: a shared generator makes this
                # interval a function of how many draws the models sorted before it consumed.
                sub = stream(model, f"detection_vs_r0/{cond}")
                row["detection_vs_r0"] = (
                    cluster_bootstrap(d, np.asarray(eps), sub, n_boot=N_BOOT) if eps
                    else paired_bootstrap(d, sub, n_boot=N_BOOT)
                )
        for clean_dir, conds in sorted(unscorable.items()):
            print(missing_baseline_message(model, args.family, conds, clean_dir))
        out[model] = rows

    # Multiplicity, over families declared here rather than chosen once the p-values are visible.
    #
    # * **prior-context** -- the R4 cells (AS, AV, AF) across all models.
    # * **ladder/<model>** -- the stated and enacted rungs, per model rather than pooled, because
    #   the ladder's direction splits by model family. That split is not what carries the result:
    #   pooling all nine rung contrasts into one family of nine leaves the six Qwen degradations
    #   surviving (largest p = 0.0105 against a threshold of 0.0125).
    #
    # Declared for all THREE signal-detection quantities, one family each, not for delta-d' alone.
    # Correcting only d' would hold the hypothesis being rejected to a stricter standard than the
    # one being kept. It rescues nothing: criterion still survives 13 of 15 model x wording
    # combinations against d-prime's 0 of 15, and balanced accuracy 6 of 15.
    #
    # Separate families rather than one of 27: the paper advances three distinct sets of nine
    # hypotheses, and pooling would charge a survivor of one for the others' tests.
    fams: dict[str, dict[tuple[str, str, str], float]] = {}
    for model, rows in out.items():
        for cond, r in rows.items():
            for q in QUANTITIES:
                d = r.get(q)
                if d is None:
                    continue
                scope = f"ladder/{model}" if cond in LADDER_RUNGS else "prior-context"
                fams.setdefault(f"{scope}/{q}", {})[(model, cond, q)] = d["p"]
    # A Holm threshold is alpha/k, so a family missing members tests its survivors more leniently
    # than the declared family would. That happened silently once: the wording sweep ran only AS at
    # F2-F5, so the same cell was tested at 0.0056 at one wording and 0.0167 at another. The
    # declaration is what gets reviewed, so an under-populated family has to say so.
    expected = len(PRIOR_CONTEXT_CELLS) * len(out)
    for q in QUANTITIES:
        members = fams.get(f"prior-context/{q}", {})
        if msg := short_family_message(
            f"prior-context/{q}",
            got=len(members),
            expected=expected,
            present=[c for _, c, _ in members],
        ):
            print(f"\n{msg}")
    for fam, pv in sorted(fams.items()):
        for (model, cond, q), decided in holm(pv).items():
            out[model][cond][q]["holm"] = {"family": fam, "k": len(pv), **decided}

    for model, rows in out.items():
        print(f"\n### {model}")
        print("| condition | FAR (clean) | detection | d' | c | bal.acc | localisation | ECE |")
        print("|---|---|---|---|---|---|---|---|")
        for cond, r in rows.items():
            print(f"| {cond} | {r['far']:.4f} | {r['detection']:.4f} | {r['d_prime']:.3f} "
                  f"| {r['criterion_c']:+.3f} | {r['balanced_accuracy']:.4f} "
                  f"| {r['localisation']:.4f} | {r['ece']:.4f} |")
        base = rows.get("R0")
        if not base:
            continue
        print()
        for cond, r in rows.items():
            if cond == "R0" or "detection_vs_r0" not in r:
                continue
            # The clean-arm R0 record, which is a different object from `base` above: that one is
            # this model's R0 on the INCORRECT arm and supplies the detection rate, this one is its
            # R0 on the condition's own clean directory and supplies the false-alarm rate. Naming
            # both `base` silently read "detection" off the clean record.
            base_clean = baselines.get((model, origin[(model, cond)]))
            if base_clean is None:
                continue
            base_far = far_of(np.array(base_clean["per_item_probs"], dtype=float))
            dfar = (r["far"] - base_far) * 100
            ddet = (r["detection"] - base["detection"]) * 100
            b = r["detection_vs_r0"]
            dp, cp = r.get("delta_d_prime"), r.get("delta_criterion")
            sig = "" if b["p"] > 0.05 else " *"
            # Verdict logic fixed in advance (see the module docstring) so it cannot be chosen to
            # suit the numbers.
            if dfar >= -0.5:
                reading = "FAR did not fall; nothing to explain"
            elif dp and dp["holm"]["reject"] and dp["effect"] > 0:
                reading = "IMPROVEMENT: FAR fell and d' rose, surviving its family's correction"
            elif dp and dp["p"] <= 0.05 and dp["effect"] > 0:
                reading = (f"NOT CORRECTED: d' rose at p={dp['p']:.3f} but its family "
                           f"({dp['holm']['family']}) tests it at "
                           f"{dp['holm']['threshold']:.4f} -- do not claim an improvement")
            elif dp and dp["p"] > 0.05:
                reading = ("THRESHOLD SHIFT: FAR fell, d' interval includes zero -- the operating "
                           "point moved, discrimination did not")
            elif ddet > 0:
                reading = "DISCRIMINATION: FAR fell and detection rose"
            else:
                reading = "mixed: see d' and c"
            print(f"  {cond}: dFAR={dfar:+.2f}pp  ddetection={ddet:+.2f}pp{sig} "
                  f"(p={b['p']:.4f}, {b['n_clusters']} clusters)")
            if dp and cp:
                h = dp["holm"]
                print(f"     dd'={dp['effect']:+.3f} [{dp['lo']:+.3f},{dp['hi']:+.3f}] "
                      f"p={dp['p']:.3f} (Holm {h['family']}: alpha={h['threshold']:.4f}, "
                      f"{'REJECT' if h['reject'] else 'retain null'})"
                      f"   dc={cp['effect']:+.3f} "
                      f"[{cp['lo']:+.3f},{cp['hi']:+.3f}] p={cp['p']:.3f}")
            if ba := r.get("delta_balanced_accuracy"):
                print(f"     dbal.acc={100 * ba['effect']:+.2f}pp "
                      f"[{100 * ba['lo']:+.2f},{100 * ba['hi']:+.2f}] p={ba['p']:.3f}")
            print(f"     -> {reading}")

    Path(args.out).write_text(json.dumps(
        {"family": args.family, "n_boot": N_BOOT, "results": out}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
