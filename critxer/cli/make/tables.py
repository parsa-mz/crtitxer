#!/usr/bin/env python
"""Emit the paper's two artefact-driven LaTeX tables from the scored JSON.

A table a human retypes is a table that disagrees with its artefacts eventually; both of these did.
The counts quoted in the prose are printed too, so a sentence claiming "15 of 15" can be checked
against the artefacts rather than against memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from critxer.core.audit import SCREENED_OUT, WORDINGS
from critxer.core.paths import artefact

# Display names, and the order the paper uses. Screened-out models come last, below a rule.
MODEL_ORDER = ("ministral-14B", "qwen3.6-27B", "qwen3.6-35B-A3B", "gemma-4-26B", "gemma-4-31B")
PRETTY = {
    "ministral-14B": "Ministral-3-14B",
    "qwen3.6-27B": "Qwen3.6-27B",
    "qwen3.6-35B-A3B": "Qwen3.6-35B-A3B",
    "gemma-4-26B": "gemma-4-26B-A4B-it",
    "gemma-4-31B": "gemma-4-31B-it",
}


def factorial_path(family: str) -> Path:
    """F1 is the primary wording and keeps the unsuffixed name; the sweep is suffixed."""
    return artefact("r4_factorial.json" if family == "F1" else f"r4_factorial_{family}.json")


def load_all() -> dict[str, dict]:
    out = {}
    for fam in WORDINGS:
        p = factorial_path(fam)
        if p.exists():
            out[fam] = json.loads(p.read_text())["results"]
    return out


def cell(d: dict | None) -> str:
    """One effect in percentage points, starred when its interval excludes zero."""
    if d is None:
        return "---"
    star = "$^{*}$" if d["p"] <= 0.05 else ""
    return f"${d['effect'] * 100:+.2f}$" + star


def allcombos_table(by_fam: dict[str, dict]) -> str:
    rows, prev_model = [], None
    for model in MODEL_ORDER:
        for fam in WORDINGS:
            r = by_fam.get(fam, {}).get(model)
            if r is None:
                continue
            if prev_model is not None and model != prev_model:
                rows.append(r"\midrule")
            prev_model = model
            rows.append(
                f"{PRETTY[model]} & {fam} & {cell(r.get('episode_vs_filler'))} "
                f"& {cell(r.get('placement_assistant_minus_user'))} "
                f"& {cell(r.get('attribution_self_minus_peer'))} \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{llrrr}\n\\toprule\n"
        "model & wording & episode $-$ AF & placement & attribution \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Every model $\\times$ wording combination we ran, in FAR percentage points, "
        "clustered on the frozen episode. $^{*}$ marks an interval excluding zero. The two models "
        "below the second-to-last rule are the ones the sensitivity screen rejects "
        "(\\S\\ref{sec:gate0}) and are excluded from the counts quoted in \\S\\ref{sec:r4}. "
        "Nothing here is aggregated. Generated from the scored artefacts by "
        "\\texttt{critxer tables}, not transcribed.}\n"
        "\\label{tab:allcombos}\n\\end{table}\n"
    )


def counts(by_fam: dict[str, dict]) -> list[str]:
    """The prose claims, recomputed. A count in a sentence should come from here."""
    lines = []
    for name, key in (("episode - AF", "episode_vs_filler"),
                      ("placement", "placement_assistant_minus_user"),
                      ("attribution", "attribution_self_minus_peer")):
        vals = [r[key] for fam in WORDINGS for m, r in by_fam.get(fam, {}).items()
                if m not in SCREENED_OUT and key in r]
        sig = [v for v in vals if v["p"] <= 0.05]
        neg = [v for v in vals if v["effect"] < 0]
        effs = [v["effect"] * 100 for v in sig]
        span = f"{min(effs):+.2f} to {max(effs):+.2f}pp" if effs else "n/a"
        lines.append(f"  {name:14s}: {len(vals)} combinations, {len(sig)} exclude zero, "
                     f"{len(neg)} negative, significant range {span}, "
                     f"max p among significant {max((v['p'] for v in sig), default=0):.4f}")
    return lines


def detection_path(family: str) -> Path:
    return artefact(f"detection_scored_{family}.json")


DETECTION_ROWS = (
    ("qwen3.6-27B", ("AS", "R2", "R3", "R3u")),
    ("qwen3.6-35B-A3B", ("AS", "R2", "R3", "R3u")),
    ("ministral-14B", ("AS", "R2", "R3", "R3u")),
)


def detection_wordings_table(by_fam: dict[str, dict]) -> str:
    """Delta-d' per wording with survival marks, plus the mean criterion move."""
    rows, first = [], True
    for model, conds in DETECTION_ROWS:
        if not first:
            rows.append(r"\midrule")
        first = False
        for cond in conds:
            cells, dps, dcs = [], [], []
            for fam in WORDINGS:
                r = by_fam.get(fam, {}).get(model, {}).get(cond)
                dp = r.get("delta_d_prime") if r else None
                if dp is None:
                    cells.append("---")
                    continue
                dps.append(dp["effect"])
                dag = "$^{\\dagger}$" if dp.get("holm", {}).get("reject") else ""
                cells.append(f"${dp['effect']:+.3f}$" + dag)
                dc = r.get("delta_criterion")
                if dc:
                    dcs.append(dc["effect"])
            surv = sum(
                1 for fam in WORDINGS
                if (by_fam.get(fam, {}).get(model, {}).get(cond) or {})
                .get("delta_d_prime", {}).get("holm", {}).get("reject")
            )
            mean_dp = f"${sum(dps) / len(dps):+.3f}$" if dps else "---"
            mean_dc = f"${sum(dcs) / len(dcs):+.3f}$" if dcs else "---"
            rows.append(f"{PRETTY[model]} & {cond} & " + " & ".join(cells)
                        + f" & {mean_dp} & {surv}/{len(dps)} & {mean_dc} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n\\footnotesize\n"
        "\\begin{tabular}{llrrrrrrrr}\n\\toprule\n"
        "& & \\multicolumn{5}{c}{$\\Delta d'$ by wording} & & & \\\\\n\\cmidrule(lr){3-7}\n"
        "model & cond & F1 & F2 & F3 & F4 & F5 & mean & surv. & mean $\\Delta c$ \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{The detection arm at all five wordings: 929 labelled-incorrect traces per cell. "
        "$^{\\dagger}$ marks survival of Holm--Bonferroni within that wording's families; each "
        "wording is corrected as its own set of families, over the same declared membership at "
        "every wording, and the cross-wording summary is descriptive, as elsewhere in this paper. "
        "\\textbf{Read the sign column, not the count}: across the 30 Qwen ladder-rung contrasts, "
        "every contrast that survives correction is a degradation --- no rung on either Qwen model "
        "ever produces a surviving improvement, at any wording. Ministral's R3u, the one "
        "improvement we reported at F1, survives at F1 alone and is negative at two other "
        "wordings. Generated from the scored artefacts by \\texttt{critxer tables}.}\n"
        "\\label{tab:detection-wordings}\n\\end{table}\n"
    )


def write_table(path: str | Path, tex: str) -> None:
    """Write ``tex`` to ``path``, creating the directory if it is not there.

    The default output directory is the paper tree, which is not part of the repository, so in a
    fresh clone it does not exist and a plain ``write_text`` raises.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/table_allcombos.tex")
    ap.add_argument("--detection-out", default="paper/table_detection_wordings.tex")
    args = ap.parse_args()

    by_fam = load_all()
    if not by_fam:
        raise SystemExit("no r4_factorial*.json found; run `critxer analyse-r4` first")
    write_table(args.out, allcombos_table(by_fam))
    print(f"wrote {args.out} from {len(by_fam)} wordings")
    print("\ncounts as the artefacts have them (check the prose against these):")
    for line in counts(by_fam):
        print(line)

    det = {f: json.loads(detection_path(f).read_text())["results"]
           for f in WORDINGS if detection_path(f).exists()}
    if det:
        write_table(args.detection_out, detection_wordings_table(det))
        print(f"\nwrote {args.detection_out} from {len(det)} wordings")
        print(detection_counts(det))


def detection_counts(det: dict[str, dict]) -> str:
    """The paper's directional detection claim, recomputed rather than remembered."""
    rungs, ctx = [], []
    for fam, models in det.items():
        for m, rows in models.items():
            for cond, r in rows.items():
                dp = r.get("delta_d_prime")
                if dp is None:
                    continue
                (ctx if cond in ("AS", "AV", "AF") else rungs).append(
                    (m, fam, cond, dp["effect"], dp.get("holm", {}).get("reject", False)))
    qwen = [x for x in rungs if x[0].startswith("qwen")]
    surv = [x for x in qwen if x[4]]
    ctx_as = [x for x in ctx if x[2] == "AS"]
    return (
        f"  Qwen ladder rungs: {len(qwen)} contrasts, {sum(1 for x in qwen if x[3] < 0)} negative, "
        f"{len(surv)} survive, of which {sum(1 for x in surv if x[3] < 0)} are degradations\n"
        f"  prior-context AS:  {len(ctx_as)} contrasts, "
        f"{sum(1 for x in ctx_as if x[4])} survive"
    )


if __name__ == "__main__":
    main()
