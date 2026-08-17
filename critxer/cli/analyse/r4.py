#!/usr/bin/env python
"""Factorial analysis of the R4 2x2 plus its filler control.

`analyse/ladder.py` compares each cell to R0 one at a time; this measures the contrasts *between*
cells, which is what the 2x2 was built for. All are paired bootstraps on the same items, so each
item's own baseline propensity cancels.

* **Placement**: (AS + AO)/2 - (US + UO)/2. **Attribution**: (AS + US)/2 - (AO + UO)/2.
  **Interaction**: (AS - AO) - (US - UO).
* **Episode presence**: mean(all four cells) - R0, which is what the cells share and R0 lacks.
* **AF - R0**: the same context position holding a length-matched non-audit exchange. If episode
  presence moves FAR and AF does not, the effect is the episode's content, not context per se.
* **Episode - AV**: the repair's own contribution, AV being the episode with the repair deleted.
* **AX - AS**: AX is the same structure on a *faulty* trace, so its verdict is "incorrect". Verdict
  priming predicts FAR up; experienced repair predicts it behaves like AS.

The verdicts are lopsided, which is why AV and AN exist: 88%/86% of the Qwen models' episodes report
"correct" (34% on Ministral), so those cells put a "correct" assertion in front of a model about to
emit that field, and AF cannot separate that from repair content because it removes both at once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from critxer.core.audit import SCREENED_OUT
from critxer.core.multiplicity import holm, short_family_message
from critxer.core.paths import artefact, run_dir
from critxer.core.r4 import CELLS, check_pool_fingerprints
from critxer.core.resample import Streams, cluster_bootstrap, paired_bootstrap

ALPHA = 0.05
N_BOOT = 20000
SEED = 20260805

# The multiplicity family, declared here rather than chosen once the p-values are visible.
#
# Membership is the contrasts the episode-effect section draws a CONCLUSION from. The vs-R0 rows
# (episode - R0, AF - R0, AX - R0) are excluded deliberately, and the exclusion is the part to argue
# with: they are descriptive anchors, and no claim rests on their significance. Including them would
# raise k and take AX - AS below its threshold, which is why the boundary is drawn on what the
# section claims rather than on what the analysis happens to emit.
#
# Placement, attribution and interaction are their own question (a 2x2 replication of prior work,
# not the repair-context claim) and are corrected as a separate family below.
CLAIM_CONTRASTS = (
    "episode_vs_filler",
    "audit_only_vs_filler",
    "audit_plus_inert_vs_filler",
    "repair_contribution_as_minus_av",
    "repair_contribution_as_minus_an",
    "verdict_composition_ax_minus_as",
    "genuine_repair_ax_minus_axn",
    "polarity_axn_minus_an",
)
FACTORIAL_CONTRASTS = (
    "placement_assistant_minus_user",
    "attribution_self_minus_peer",
    "interaction",
)
# Models the instrument-sensitivity screen disqualified before any hypothesis was
# tested. Their cells are still run and reported -- the selection effect is worth showing -- but no
# claim rests on them, so counting them would raise k and test the claims we do make against
# hypotheses we never advanced. Imported rather than re-declared: a divergent second copy would
# change k here and a prose count elsewhere.


def boot(diff: np.ndarray, rng: np.random.Generator, clusters=None) -> dict:
    """A contrast's interval, clustered on the frozen episode when one applies.

    The pool of 50 episodes is cycled over 465 targets, so resampling items alone treats each
    episode's ~9 targets as independent. The item-only interval is kept alongside under
    ``item_only`` because the difference between the two is itself worth showing.

    ``clusters=None`` means no episode structure applies and the item bootstrap is correct alone.
    """
    item = paired_bootstrap(diff, rng, n_boot=N_BOOT, alpha=ALPHA)
    if clusters is None:
        return {**item, "clustered": False}
    out = cluster_bootstrap(diff, np.asarray(clusters), rng, n_boot=N_BOOT, alpha=ALPHA)
    return {**out, "clustered": True,
            "item_only": {k: item[k] for k in ("effect", "lo", "hi", "p")}}


def widest(*candidates: dict) -> dict:
    """The most conservative of several clusterings of the same contrast.

    Needed only for AX - AS, whose two sides come from *different* episode pools with disjoint ids,
    so no single grouping covers both sources of dependence. Rather than assume the structure away,
    cluster on each pool in turn and report whichever gives the wider interval.
    """
    return max(candidates, key=lambda d: d["hi"] - d["lo"])


def survives(d: dict) -> bool:
    """Whether a contrast clears its declared family, which is what a claim may rest on.

    Contrasts with no declared family (the vs-R0 anchors) fall back to the raw interval, which is
    all that is claimed of them.
    """
    h = d.get("holm")
    return bool(h["reject"]) if h else d["p"] <= ALPHA


def fmt(d: dict) -> str:
    """A contrast's line. Corrected significance is marked separately from the raw interval.

    ``*`` is the interval excluding zero; ``+`` is survival of the declared family. Both are shown
    because they answer different questions, and they do disagree.
    """
    star = "" if d["p"] > ALPHA else " *"
    h = d.get("holm")
    if h:
        star += " +" if h["reject"] else f" (Holm {h['family']} k={h['k']}: retained)"
    return f"{d['effect']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] p={d['p']:.4f}{star}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(run_dir("r4")))
    ap.add_argument("--family", default="F1")
    ap.add_argument("--out", default=str(artefact("r4_factorial.json")))
    args = ap.parse_args()

    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(args.dir).glob(f"*__{args.family}.json")):
        rec = json.loads(path.read_text())
        runs.setdefault(rec["auditor"], {})[rec["condition"]] = rec

    # One independent stream per (model, contrast) rather than one generator for the run. With a
    # shared one, models are iterated in sorted order, so draws consumed by whichever model sorts
    # first shift the stream for the rest -- enough to move a p-value across a Holm threshold with
    # the model's own data untouched.
    stream = Streams(SEED)

    out = {}
    for model, by_cell in sorted(runs.items()):
        missing = [c for c in ("R0", *CELLS) if c not in by_cell]
        if missing:
            print(f"skip {model}: missing {missing}")
            continue
        # Item order is identical across cells by construction; assert it rather than trust it.
        ids = by_cell["R0"]["item_ids"]
        for c in CELLS:
            if by_cell[c]["item_ids"] != ids:
                raise SystemExit(f"{model}/{c} item order differs from R0; contrasts would be void")
        p = {c: np.array(by_cell[c]["per_item_probs"]) for c in ("R0", *CELLS)}
        # Cluster labels for every contrast drawn from the ordinary episode pool. All four cells and
        # AF/AV share one assignment, so any of them gives it; AS is used by convention. Cells run
        # before `episode_ids` was persisted were backfilled by `critxer backfill-episode-ids`.
        # Cells of the same pool must have been built from the *same* frozen episodes -- T=0 is not
        # reproducible under continuous batching, so a regenerated pool can differ, and comparing
        # across two pools would silently form a contrast that varies the episode as well as the
        # condition. `check_pool_fingerprints` knows which cells belong to which pool.
        check_pool_fingerprints(by_cell)
        main_ep = by_cell["AS"].get("episode_ids")
        if main_ep is None:
            raise SystemExit(
                f"{model}/AS has no episode_ids; run `critxer backfill-episode-ids` first -- "
                "without it every R4 interval is clustered wrongly"
            )

        row = {
            "far": {c: by_cell[c]["far"] for c in by_cell},
            "episodes_in_pool": len(set(main_ep)),
            "targets_per_episode": len(ids) / len(set(main_ep)),
            "placement_assistant_minus_user": boot(
                (p["AS"] + p["AO"]) / 2 - (p["US"] + p["UO"]) / 2,
                stream(model, "placement_assistant_minus_user"), main_ep),
            "attribution_self_minus_peer": boot(
                (p["AS"] + p["US"]) / 2 - (p["AO"] + p["UO"]) / 2,
                stream(model, "attribution_self_minus_peer"), main_ep),
            "interaction": boot((p["AS"] - p["AO"]) - (p["US"] - p["UO"]),
                                stream(model, "interaction"), main_ep),
            "episode_presence_vs_r0": boot(
                sum(p[c] for c in CELLS) / len(CELLS) - p["R0"],
                stream(model, "episode_presence_vs_r0"), main_ep),
        }
        if "AF" in by_cell:
            if by_cell["AF"]["item_ids"] != ids:
                raise SystemExit(f"{model}/AF item order differs from R0")
            af = np.array(by_cell["AF"]["per_item_probs"])
            row["filler_vs_r0"] = boot(af - p["R0"], stream(model, "filler_vs_r0"), main_ep)
            # The contrast that actually isolates content: both sides carry a prior exchange of
            # comparable length, so only what the exchange was about differs. Comparing each to R0
            # separately leaves "has any prior turn at all" in both.
            row["episode_vs_filler"] = boot(
                sum(p[c] for c in CELLS) / len(CELLS) - af,
                stream(model, "episode_vs_filler"), main_ep)
        if "AV" in by_cell:
            if by_cell["AV"]["item_ids"] != ids:
                raise SystemExit(f"{model}/AV item order differs from R0")
            av = np.array(by_cell["AV"]["per_item_probs"])
            row["audit_only_vs_r0"] = boot(av - p["R0"], stream(model, "audit_only_vs_r0"), main_ep)
            # AS rather than the four-cell mean: AV is built at assistant placement and self
            # attribution, so pairing it with AS keeps the contrast to the repair alone.
            row["repair_contribution_as_minus_av"] = boot(
                p["AS"] - av, stream(model, "repair_contribution_as_minus_av"), main_ep)
            if "AF" in by_cell:
                row["audit_only_vs_filler"] = boot(
                    av - af, stream(model, "audit_only_vs_filler"), main_ep)
        if "AN" in by_cell:
            if by_cell["AN"]["item_ids"] != ids:
                raise SystemExit(f"{model}/AN item order differs from R0")
            an = np.array(by_cell["AN"]["per_item_probs"])
            # The identified decomposition. AS - AV varies the repair AND whether a continuation
            # exists at all; AN holds a continuation present on both sides and varies only
            # whether it is a repair. The AN continuations run 22-35% LONGER than the repairs they
            # replace, so AS - AN is conservative with respect to a length explanation.
            row["repair_contribution_as_minus_an"] = boot(
                p["AS"] - an, stream(model, "repair_contribution_as_minus_an"), main_ep)
            if "AV" in by_cell:
                row["continuation_presence_av_minus_an"] = boot(
                    av - an, stream(model, "continuation_presence_av_minus_an"), main_ep)
            if "AF" in by_cell:
                row["audit_plus_inert_vs_filler"] = boot(
                    an - af, stream(model, "audit_plus_inert_vs_filler"), main_ep)
        if "AX" in by_cell:
            if by_cell["AX"]["item_ids"] != ids:
                raise SystemExit(f"{model}/AX item order differs from R0")
            ax = np.array(by_cell["AX"]["per_item_probs"])
            ax_ep = by_cell["AX"].get("episode_ids")
            if ax_ep is None:
                raise SystemExit(f"{model}/AX has no episode_ids; run the backfill script")
            row["ax_episodes_in_pool"] = len(set(ax_ep))
            row["incorrect_verdict_vs_r0"] = boot(
                ax - p["R0"], stream(model, "incorrect_verdict_vs_r0"), ax_ep)
            # Two disjoint pools contribute to this one contrast, so neither grouping alone covers
            # the dependence. Report the more conservative of the two.
            row["verdict_composition_ax_minus_as"] = widest(
                boot(ax - p["AS"], stream(model, "verdict_composition_ax_minus_as#main"), main_ep),
                boot(ax - p["AS"], stream(model, "verdict_composition_ax_minus_as#ax"), ax_ep))
            if "AXN" in by_cell:
                if by_cell["AXN"]["item_ids"] != ids:
                    raise SystemExit(f"{model}/AXN item order differs from R0")
                axn = np.array(by_cell["AXN"]["per_item_probs"])
                # The only contrast here in which the removed continuation is a real correction to
                # a real fault. AS - AN cannot be: 88%/86% of the Qwen models' ordinary episodes
                # report "correct", so its "repair" is mostly a re-assertion of a clean verdict and
                # the contrast measures what a repair *request* elicits. Both sides of AX - AXN
                # carry a genuine "incorrect" audit; only one continues by fixing the error.
                row["genuine_repair_ax_minus_axn"] = boot(
                    ax - axn, stream(model, "genuine_repair_ax_minus_axn"), ax_ep)
                if "AN" in by_cell:
                    # An error-verdict episode with the repair removed from both sides. AXN and AN
                    # both carry an audit followed by an inert continuation, so this isolates what
                    # the *verdict* does from what the *repair* does -- but NOT the verdict token
                    # from the trace it describes: AN draws on the clean pool and AXN on the faulty
                    # one, so the episode's content varies with its verdict. Separating the token
                    # needs a fourth pool (the same faulty trace audited correct), which does not
                    # exist yet. Two disjoint pools contribute, so like AX - AS it reports the more
                    # conservative of the two clusterings.
                    row["polarity_axn_minus_an"] = widest(
                        boot(axn - an, stream(model, "polarity_axn_minus_an#main"), main_ep),
                        boot(axn - an, stream(model, "polarity_axn_minus_an#ax"), ax_ep))
        out[model] = row

    # Holm-Bonferroni over the families declared at the top of this module.
    #
    # Each wording is corrected as its own set of families over the SAME declared membership, which
    # is the convention `cli/make/tables.py` states to the reader. That only means anything if the
    # membership really is the same, and it was not once: AN, AX and AXN existed only at F1, so the
    # family ran at eighteen members there and twelve at F2-F5, and twelve is a threshold survivors
    # clear more easily. The guard now fails loudly rather than correcting against what is on disk.
    eligible = [m for m in out if m not in SCREENED_OUT]
    for fam_name, members in (("r4-claims", CLAIM_CONTRASTS),
                              ("r4-factorial", FACTORIAL_CONTRASTS)):
        pvals = {(m, c): out[m][c]["p"]
                 for m in eligible
                 for c in members if c in out[m] and out[m][c] is not None}
        if msg := short_family_message(
            fam_name,
            got=len(pvals),
            expected=len(members) * len(eligible),
            present=[c for _, c in pvals],
            alpha=ALPHA,
        ):
            print(f"\n{msg}")
        for (model, cond), decided in holm(pvals, alpha=ALPHA).items():
            out[model][cond]["holm"] = {"family": fam_name, "k": len(pvals), **decided}

    for model, row in out.items():
        print(f"\n### {model}")
        print("| contrast | estimate | 95% CI | p |")
        print("|---|---|---|---|")
        for name in ("episode_presence_vs_r0", "filler_vs_r0", "audit_only_vs_r0",
                     "incorrect_verdict_vs_r0", "placement_assistant_minus_user",
                     "attribution_self_minus_peer", "interaction"):
            if name in row:
                d = row[name]
                print(f"| {name} | {d['effect']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] "
                      f"| {d['p']:.4f} |")
        ep, fl = row["episode_presence_vs_r0"], row.get("filler_vs_r0")
        if fl:
            print(f"\n  episode moves FAR: {fmt(ep)}")
            print(f"  length-matched filler: {fmt(fl)}")
            print(f"  episode vs filler (content, length held): {fmt(row['episode_vs_filler'])}")
            # The confound is a filler moving the SAME way as the episode -- that would mean
            # prior context of any kind does it. A filler moving the *opposite* way, or not at
            # all, leaves content as the explanation and makes the vs-R0 estimate conservative.
            same_way = survives(fl) and (fl["effect"] > 0) == (ep["effect"] > 0)
            share = abs(fl["effect"]) / abs(ep["effect"]) if ep["effect"] else float("inf")
            if same_way and share > 0.5:
                print(f"  -> CONFOUNDED: filler moves the same way at {share:.0%} of the "
                      "episode effect; this may be context length rather than the episode")
            elif same_way:
                print(f"  -> partly confounded: filler moves the same way but only at "
                      f"{share:.0%} of the episode effect")
            else:
                print("  -> the episode's CONTENT carries it: the filler does not move the same "
                      "way, so prior context per se does not explain the effect")
        if "repair_contribution_as_minus_av" in row:
            rep, av0 = row["repair_contribution_as_minus_av"], row["audit_only_vs_r0"]
            print(f"\n  audit-only episode (AV) vs R0: {fmt(av0)}")
            print(f"  the repair's own contribution (AS - AV): {fmt(rep)}")
            if "audit_only_vs_filler" in row:
                print(f"  AV vs length-matched filler: {fmt(row['audit_only_vs_filler'])}")
            # Which half of the episode carries the effect. "Enacted responsibility" requires the
            # repair to add something; if AV already reproduces the full effect, the claim is
            # about seeing one's own audit, not about having repaired anything.
            reproduced = abs(av0["effect"]) / abs(ep["effect"]) if ep["effect"] else float("inf")
            if not survives(rep):
                print(f"  -> the REPAIR ADDS NOTHING: AS - AV is null, and the audit alone "
                      f"reproduces {reproduced:.0%} of the episode effect. The effect belongs to "
                      "the prior audit, not to having performed a repair.")
            elif (rep["effect"] > 0) == (ep["effect"] > 0):
                print(f"  -> the repair carries part of it: AS - AV moves the same way as the "
                      f"episode effect, with the audit alone reproducing {reproduced:.0%}")
            else:
                print("  -> the repair moves FAR the OPPOSITE way to the episode effect; the "
                      "audit overshoots and the repair pulls back")
        if "repair_contribution_as_minus_an" in row:
            rep_an = row["repair_contribution_as_minus_an"]
            print(f"\n  IDENTIFIED repair contribution (AS - AN): {fmt(rep_an)}")
            if "continuation_presence_av_minus_an" in row:
                print(f"  continuation presence (AV - AN): "
                      f"{fmt(row['continuation_presence_av_minus_an'])}")
            if "audit_plus_inert_vs_filler" in row:
                print(f"  audit + inert continuation vs filler (AN - AF): "
                      f"{fmt(row['audit_plus_inert_vs_filler'])}")
            # The contrast the paper's central claim needs: both sides carry an audit and a
            # continuation, and differ only in whether the continuation repairs anything.
            if not survives(rep_an):
                print("  -> the REPAIR ADDS NOTHING once a continuation is held present: "
                      "the effect is the prior audit, not having repaired")
            else:
                print(f"  -> the repair contributes {rep_an['effect']:+.4f} beyond an "
                      "inert continuation of the same or greater length")
        if "verdict_composition_ax_minus_as" in row:
            vc, ax0 = row["verdict_composition_ax_minus_as"], row["incorrect_verdict_vs_r0"]
            print(f"\n  incorrect-verdict episode (AX) vs R0: {fmt(ax0)}")
            print(f"  verdict composition (AX - AS): {fmt(vc)}")
            # Under verdict priming the model agrees with whatever verdict it just gave, so an
            # episode reporting an error should push FAR *up* relative to one reporting none.
            if not survives(vc):
                print("  -> NOT VERDICT PRIMING: flipping the episode's verdict does not move "
                      "FAR, so the effect does not run through agreeing with itself")
            elif vc["effect"] > 0:
                print(f"  -> VERDICT PRIMING is at least partly responsible: an incorrect-verdict "
                      f"episode raises FAR by {vc['effect']:+.4f} against AS, the direction "
                      "self-consistency predicts")
            else:
                print(f"  -> AX lowers FAR *further* than AS ({vc['effect']:+.4f}); the direction "
                      "is opposite to verdict priming, so something other than agreeing with "
                      "itself is at work")
        if "genuine_repair_ax_minus_axn" in row:
            gr = row["genuine_repair_ax_minus_axn"]
            print(f"\n  GENUINE repair contribution (AX - AXN): {fmt(gr)}")
            if not survives(gr):
                print("  -> correcting a real error adds nothing over restating the same verdict: "
                      "what the ordinary AS - AN picks up is the repair REQUEST, not the repair")
            else:
                print(f"  -> correcting a real error contributes {gr['effect']:+.4f} beyond an "
                      "inert continuation of the same audit")
        for label, key in (("placement", "placement_assistant_minus_user"),
                           ("attribution", "attribution_self_minus_peer")):
            d = row[key]
            print(f"  {label}: {fmt(d)}"
                  + ("" if d["p"] <= ALPHA else "  (null)"))

    Path(args.out).write_text(json.dumps(
        {"alpha": ALPHA, "n_boot": N_BOOT, "results": out}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
