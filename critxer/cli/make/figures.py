#!/usr/bin/env python
"""Figures for the paper. Everything reads persisted JSON: no model inference, no GPU, no network.

* **Figure 0** (schematic): the design before any result -- what is held byte-identical, what varies
  between rungs, what R4 adds, and how the two arms combine into d' and c.
* **Figure 1** (dot-and-interval): the R4 contrasts on one axis, so the identified ones (AS - AN,
  AX - AXN) sit beside the coarser ones that motivate looking at them. Horizontal, because nine
  stacked forest rows ran two thirds of a page for nine numbers.
* **Figure 2** (two panels): the ladder's *identified* decomposition across wordings. R3 - R2 is
  deliberately NOT plotted -- it sums two large opposite-signed components that nearly cancel, so
  plotting it invites the reading the text argues against.
* **Figure 3** (quadrant): each condition by its false-alarm change against its d' change, the one
  view that separates "fewer false alarms because it discriminates better" from "because it flags
  less". Defined below between figures 1 and 2, in the order the two were added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from critxer.core.audit import WORDINGS
from critxer.core.paths import data_root

OUT_DIR = Path("paper/figures")

# Colourblind-safe categorical triple, each paired with a distinct marker so identity survives in
# greyscale.
MODELS = ("qwen3.6-27B", "qwen3.6-35B-A3B", "ministral-14B")
MODEL_STYLE = {
    "qwen3.6-27B": {"color": "#2a78d6", "marker": "o", "label": "Qwen3.6-27B"},
    "qwen3.6-35B-A3B": {"color": "#eb6834", "marker": "s", "label": "Qwen3.6-35B-A3B"},
    "ministral-14B": {"color": "#1baf7a", "marker": "^", "label": "Ministral-3-14B"},
}
PP = 100.0  # proportions -> percentage points

INK = "#20201d"
MUTED = "#6d6c64"
RULE = "#c3c2b7"
FIXED_FILL = "#eceae1"
VARY_FILL = "#dce9f7"
CTX_FILL = "#fce6da"


def _box(ax, x, y, w, h, face, text, *, fontsize=6.4, weight="normal", edge=None, style="round"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"{style},pad=0,rounding_size=0.012",
        facecolor=face, edgecolor=edge or face, linewidth=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=3, fontweight=weight, linespacing=1.35)


def _arrow(ax, xy_from, xy_to, *, color=MUTED, lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=7,
        color=color, linewidth=lw, shrinkA=1, shrinkB=1, zorder=4))


PANEL_ORDER = ("pipeline", "ladder", "arms")
# Relative widths, keyed by panel, so dropping one does not leave the others at the wrong ratio.
PANEL_WIDTH = {"pipeline": 1.05, "ladder": 1.0, "arms": 0.95}


def panel_letter(panels: tuple[str, ...], panel: str) -> str:
    """The letter a panel carries, from its position in *this* figure rather than its identity.

    The ALTA write-up drops the ladder panel, which makes the two-arm panel "(b)". Hard-coding each
    panel's letter instead renders "(a)" beside "(c)" on a subset and contradicts the caption.
    """
    return "abc"[panels.index(panel)]


def make_fig0(out_stem: Path, panels: tuple[str, ...] = PANEL_ORDER) -> None:
    """The design, in one picture: the pipeline, the ladder, and how the two arms combine.

    Panels left to right in the order the paper needs them: the pipeline and where the manipulation
    sits, what the rungs vary, and how the two arms combine into d' and c. ``panels`` selects a
    subset, keeping the declared order regardless of the order asked for.
    """
    if unknown := [p for p in panels if p not in PANEL_ORDER]:
        raise SystemExit(f"unknown fig0 panel(s) {unknown}; expected some of {list(PANEL_ORDER)}")
    panels = tuple(p for p in PANEL_ORDER if p in panels)
    if not panels:
        raise SystemExit("fig0 needs at least one panel")

    fig = plt.figure(figsize=(2.3 * len(panels), 2.75))
    gs = fig.add_gridspec(1, len(panels),
                          width_ratios=[PANEL_WIDTH[p] for p in panels], wspace=0.13)
    axes = {}
    for i, name in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        axes[name] = ax
    axa, axb, axc = (axes.get(n) for n in PANEL_ORDER)

    # ---- the pipeline, and where the manipulation lives -------------------------------------
    if axa is not None:
        axa.text(0.0, 1.0, f"({panel_letter(panels, 'pipeline')}) the pipeline",
                 fontsize=7.8, fontweight="bold", color=INK, va="top")

        _box(axa, 0.06, 0.755, 0.88, 0.115, "#e8f1e9", "candidate solution", edge="#b9d6bd")
        _arrow(axa, (0.5, 0.755), (0.5, 0.685))
        _box(axa, 0.06, 0.545, 0.88, 0.14, VARY_FILL,
             "$\\bf{AUDITOR}$\ncorrect, or first bad step?", edge="#9dc2e8")
        _arrow(axa, (0.5, 0.545), (0.5, 0.475))
        _box(axa, 0.06, 0.335, 0.88, 0.14, CTX_FILL, "$\\bf{REPAIRER}$\nfixes what was flagged",
             edge="#f0b998")

        # The question, drawn as the feedback edge it is: the repairer's identity reaching
        # backwards.
        axa.annotate("", xy=(0.045, 0.615), xytext=(0.045, 0.405),
                     arrowprops={"arrowstyle": "-|>", "mutation_scale": 8, "color": "#2a78d6",
                                 "linewidth": 1.1, "connectionstyle": "arc3,rad=0.62",
                                 "shrinkA": 1, "shrinkB": 1})
        axa.text(0.5, 0.245,
                 "does $\\it{who\\ repairs\\ next}$\nchange what is reported $\\it{now}$?",
                 ha="center", va="top", fontsize=6.6, color="#2a78d6")
        axa.text(0.5, 0.05,
                 "929 ProcessBench traces\nverified correct  →  any flag is a false alarm",
                 ha="center", va="bottom", fontsize=6.1, color=MUTED, style="italic")

    # ---- the ladder ------------------------------------------------------------------------
    if axb is not None:
        axb.text(0.0, 1.0, f"({panel_letter(panels, 'ladder')}) how real the obligation is",
                 fontsize=7.8, fontweight="bold", color=INK, va="top")

        rungs = [
            ("R0", "no future task"),
            ("R0p", "unrelated future task"),
            ("R1", "another model repairs"),
            ("R2", "you repair, later"),
            ("R3/R3u", "you repair, now"),
            ("R4", "you already did"),
        ]
        # A staircase: each rung one step up and one step wider, so "more real" is a direction
        # rather than a list order the reader has to take on trust. Bars stay short and the gloss
        # sits inside the panel -- an earlier version widened them until the labels crossed over.
        n = len(rungs)
        for i, (name, gloss) in enumerate(rungs):
            y = 0.135 + i * 0.128
            w = 0.17 + 0.036 * i
            # A single-hue ramp, light to saturated, rather than greys: the greys read as disabled.
            t = i / (n - 1)
            face = (1 - 0.30 * t, 1 - 0.16 * t, 1 - 0.05 * t) if i < n - 1 else CTX_FILL
            _box(axb, 0.0, y, w, 0.088, face, "", edge="#f0b998" if i == n - 1 else None)
            axb.text(0.014, y + 0.044, name, fontsize=6.3, va="center", ha="left",
                     color=INK, fontweight="bold")
            axb.text(w + 0.028, y + 0.044, gloss, fontsize=6.1, va="center", ha="left", color=MUTED)

        _arrow(axb, (0.955, 0.13), (0.955, 0.90), color="#8a8880", lw=1.0)
        axb.text(0.925, 0.515, "more real", fontsize=6.3, color="#8a8880", rotation=90,
                 va="center", ha="right")
        axb.text(0.0, 0.055,
                 "R0–R3u differ by $\\bf{one\\ sentence}$;\neverything else byte-identical",
                 fontsize=6.1, color=MUTED, va="top", ha="left")

    # ---- the two arms ----------------------------------------------------------------------
    if axc is not None:
        axc.text(0.0, 1.0, f"({panel_letter(panels, 'arms')}) is a lower rate better, or lenient?",
                 fontsize=7.8, fontweight="bold", color=INK, va="top")

        _box(axc, 0.0, 0.70, 0.46, 0.155, "#e8f1e9", "929\n$\\bf{correct}$", edge="#b9d6bd")
        _box(axc, 0.54, 0.70, 0.46, 0.155, "#f7e6e6", "929\n$\\bf{incorrect}$", edge="#e0b9b9")
        _arrow(axc, (0.23, 0.70), (0.23, 0.565))
        _arrow(axc, (0.77, 0.70), (0.77, 0.565))
        _box(axc, 0.0, 0.425, 0.46, 0.14, FIXED_FILL, "false-alarm\nrate")
        _box(axc, 0.54, 0.425, 0.46, 0.14, FIXED_FILL, "detection\nrate")
        _arrow(axc, (0.23, 0.425), (0.40, 0.30))
        _arrow(axc, (0.77, 0.425), (0.60, 0.30))
        _box(axc, 0.10, 0.16, 0.80, 0.14, VARY_FILL, "$d'$ discrimination\n$c$ threshold",
             edge="#9dc2e8", weight="bold")
        axc.text(0.5, 0.075, "every condition run on both arms,\nat five prompt wordings",
                 ha="center", va="top", fontsize=6.1, color=MUTED, style="italic")

    fig.subplots_adjust(left=0.012, right=0.99, top=0.97, bottom=0.02)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# (key, short axis label, is this a contrast a claim rests on, cluster tag)
FIG1_ROWS = [
    ("episode_presence_vs_r0", "episode\n− R0", False, "g0"),
    ("episode_vs_filler", "episode\n− AF", False, "g0"),
    ("audit_only_vs_filler", "AV\n− AF", False, "g0"),
    # AS - AV is NOT the identified repair contrast: it varies whether a continuation exists at all
    # as well as whether that continuation repairs anything. AS - AN holds a continuation present on
    # both sides; AV - AN is its paired null. Plotted instead so the figure matches the paper.
    ("repair_contribution_as_minus_an", "AS\n− AN", True, "g1"),
    ("continuation_presence_av_minus_an", "AV\n− AN", False, "g1"),
    # The same contrast on the incorrect-verdict pool, where the removed continuation is a real
    # correction of a real fault -- which AS - AN mostly is not on the Qwen models (88%/86% of their
    # episodes' audits report "correct").
    ("genuine_repair_ax_minus_axn", "AX\n− AXN", True, "g1"),
    ("verdict_composition_ax_minus_as", "AX\n− AS", True, "g1"),
    ("placement_assistant_minus_user", "place-\nment", False, "g2"),
    ("attribution_self_minus_peer", "attri-\nbution", False, "g2"),
]
GROUP_TITLES = {"g0": "episode effect + length control",
                "g1": "what inside the episode carries it",
                "g2": "2×2 replication"}


def make_fig1(r4_path: Path, out_stem: Path) -> list[dict]:
    """Dot-and-interval plot, contrasts along x. Returns the plotted records for the report."""
    results = json.loads(r4_path.read_text())["results"]
    plotted: list[dict] = []

    # x positions, with a gap wherever the cluster tag changes so the three groups read as groups.
    xs, prev, x = [], None, 0.0
    for _, _, _, group in FIG1_ROWS:
        if prev is not None and group != prev:
            x += 0.85
        xs.append(x)
        x += 1.0
        prev = group

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.axhline(0.0, color=RULE, linewidth=1, linestyle="--", zorder=1)

    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    for idx, (key, _label, _claim, _group) in enumerate(FIG1_ROWS):
        for m_idx, model in enumerate(MODELS):
            d = results.get(model, {}).get(key)
            if d is None:
                continue  # e.g. AX - AS is absent for ministral; omit rather than plot a zero.
            effect, lo, hi = d["effect"] * PP, d["lo"] * PP, d["hi"] * PP
            xx = xs[idx] + offsets[m_idx]
            s = MODEL_STYLE[model]
            ax.plot([xx, xx], [lo, hi], color=s["color"], linewidth=1.4,
                    solid_capstyle="round", zorder=2)
            ax.plot(xx, effect, marker=s["marker"], color=s["color"], markersize=4.6,
                    markeredgecolor="#fcfcfb", markeredgewidth=0.7, zorder=3)
            plotted.append({"row": key, "model": model, "effect_pp": effect,
                            "lo_pp": lo, "hi_pp": hi, "n": d.get("n"), "p": d.get("p")})

    # Group headers, drawn above the data so the reader sees why the columns are grouped.
    ymax = max(r["hi_pp"] for r in plotted)
    ymin = min(r["lo_pp"] for r in plotted)
    head_y = ymax + (ymax - ymin) * 0.09
    for group, title in GROUP_TITLES.items():
        members = [xs[i] for i, r in enumerate(FIG1_ROWS) if r[3] == group]
        lo_x, hi_x = min(members) - 0.42, max(members) + 0.42
        ax.plot([lo_x, hi_x], [head_y, head_y], color=RULE, linewidth=0.8,
                clip_on=False, zorder=1)
        ax.text((lo_x + hi_x) / 2, head_y + (ymax - ymin) * 0.035, title, ha="center",
                va="bottom", fontsize=6.3, color=MUTED, clip_on=False)

    ax.set_xticks(xs)
    ax.set_xticklabels([r[1] for r in FIG1_ROWS], fontsize=7.4)
    for tick, row in zip(ax.get_xticklabels(), FIG1_ROWS, strict=True):
        if row[2]:
            tick.set_fontweight("bold")
    ax.set_ylabel("effect on FAR (pp)\npositive = more false alarms", fontsize=7.6)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_ylim(ymin - (ymax - ymin) * 0.06, head_y + (ymax - ymin) * 0.16)
    ax.tick_params(axis="both", labelsize=7.4)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([0], [0], marker=MODEL_STYLE[m]["marker"], color=MODEL_STYLE[m]["color"],
                      linestyle="", markersize=4.6, label=MODEL_STYLE[m]["label"])
               for m in MODELS]
    ax.legend(handles=handles, loc="lower left", ncol=3, fontsize=6.8, frameon=False,
              handletextpad=0.3, columnspacing=1.2, borderpad=0.1)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plotted


FIG2_PANELS = [("R3u_R2", "R3u − R2   enactment"),
               ("R3_R3u", "R3 − R3u   conditionality")]


# Conditions on the detection arm, and how they group for the quadrant figure. The prior-context
# cell and the prospective rungs are the paper's two answers, so they get shape and colour; the
# wording is not encoded at all, because the point of the figure is that it does not matter here.
#
# Labels are the paper's names, not the experiment's. This read "R4 prior audit context", and R4 is
# internal nomenclature that appears in neither venue's .tex -- so the legend named a condition the
# reader had no definition for, while R2/R3/R3u beside it are defined in the ladder appendix.
QUADRANT_CONDS = {
    "AS": {"color": "#2a78d6", "marker": "o", "label": "prior audit context"},
    # The reasoning-enabled check, kept visually separate because it is not comparable to the rest
    # on magnitude: thinking and token budget both differ, so only its position relative to the two
    # axes means anything. It is the one condition here measured at F1 alone.
    "AS_reasoning": {"color": "#1a4f8a", "marker": "*",
                     "label": "prior audit context, reasoning on"},
    "R2": {"color": "#c74a3c", "marker": "s", "label": "R2 stated"},
    "R3": {"color": "#eb6834", "marker": "^", "label": "R3 immediate, conditional"},
    "R3u": {"color": "#f0a03c", "marker": "D", "label": "R3u immediate"},
}


def make_fig3(det_paths: dict[str, Path], out_stem: Path,
              reasoning_path: Path | None = None) -> list[dict]:
    """Every detection-arm contrast as one point in (threshold move, discrimination change).

    Quadrants are labelled by what the auditor visibly *does*, not by sign and not by the
    lenient/conservative pair: a higher criterion is "conservative" about asserting an error, which
    is leniency toward the audited work, so printing either word made the figure contradict the
    title. The axis says which way the flag count moves instead.

    ``reasoning_path`` adds the reasoning-enabled AS cells under their own marker. Every point in
    this figure is a contrast against its own arm's R0, so the two are on the same baseline even
    though their magnitudes are not comparable.
    """
    pts: list[dict] = []
    sources = dict(det_paths)
    if reasoning_path is not None:
        sources["reasoning"] = reasoning_path
    for fam, path in sources.items():
        if not path.exists():
            continue
        for model, rows in json.loads(path.read_text())["results"].items():
            for cond, r in rows.items():
                key = "AS_reasoning" if fam == "reasoning" and cond == "AS" else cond
                dp, dc = r.get("delta_d_prime"), r.get("delta_criterion")
                if dp is None or dc is None or key not in QUADRANT_CONDS:
                    continue
                pts.append({"family": fam, "model": model, "cond": key,
                            "dc": dc["effect"], "dd": dp["effect"],
                            "survives": bool(dp.get("holm", {}).get("reject"))})

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    xs = [p["dc"] for p in pts]
    ys = [p["dd"] for p in pts]
    xpad = (max(xs) - min(xs)) * 0.12
    ypad = (max(ys) - min(ys)) * 0.16
    xlim = (min(xs) - xpad, max(xs) + xpad)
    ylim = (min(ys) - ypad, max(ys) + ypad)

    # The split that carries the result is left/right, not by quadrant: every prior-context point
    # is right of zero and almost every rung point is left of it. Tint that, faintly, and let the
    # quadrant labels do the rest.
    ax.axvspan(0, xlim[1], color="#f2f6fb", zorder=0)
    ax.axvline(0, color=RULE, linewidth=1, linestyle="--", zorder=1)
    ax.axhline(0, color=RULE, linewidth=1, linestyle="--", zorder=1)

    corner = {"fontsize": 6.3, "color": MUTED, "zorder": 1, "style": "italic"}
    ax.text(xlim[1], ylim[1], "flags fewer errors, better  ", ha="right", va="top", **corner)
    ax.text(xlim[0], ylim[1], "  flags more errors, better", ha="left", va="top", **corner)
    ax.text(xlim[1], ylim[0], "flags fewer errors, worse  ", ha="right", va="bottom", **corner)
    ax.text(xlim[0], ylim[0], "  flags more errors, worse", ha="left", va="bottom", **corner)

    for cond, style in QUADRANT_CONDS.items():
        sub = [p for p in pts if p["cond"] == cond]
        if not sub:
            continue
        # Filled = survives its declared family; hollow = does not. One glance separates "we can
        # claim this" from "the point estimate leans this way", which the table needed a dagger
        # column and a caption sentence to say.
        for filled in (False, True):
            g = [p for p in sub if p["survives"] is filled]
            if not g:
                continue
            ax.scatter([p["dc"] for p in g], [p["dd"] for p in g],
                       marker=style["marker"], s=27 if filled else 22,
                       facecolor=style["color"] if filled else "none",
                       edgecolor=style["color"], linewidth=1.0,
                       alpha=0.95 if filled else 0.75, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\Delta c$   threshold move  (right = flags fewer errors)", fontsize=7.8)
    ax.set_ylabel(r"$\Delta d'$   discrimination" "\n(up = better)", fontsize=7.8)
    ax.tick_params(axis="both", labelsize=7.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([0], [0], marker=s["marker"], color=s["color"], linestyle="",
                      markersize=4.6, label=s["label"]) for s in QUADRANT_CONDS.values()]
    handles += [Line2D([0], [0], marker="o", color=MUTED, linestyle="", markersize=4.6,
                       markerfacecolor="none", label="hollow: does not survive correction")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=6.6, frameon=False, columnspacing=1.1, handletextpad=0.35)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pts


def make_fig2(r1_path: Path, r3_path: Path, out_stem: Path) -> list[dict]:
    """Two panels, shared y. Bands rather than caps: five points x three models x two panels was
    thirty error bars competing with the lines they belonged to."""
    sources = {"R3u_R2": json.loads(r1_path.read_text())["results"],
               "R3_R3u": json.loads(r3_path.read_text())["results"]}

    plotted: list[dict] = []
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.15), sharey=True)
    x = np.arange(len(WORDINGS))

    for (src_key, title), ax in zip(FIG2_PANELS, axes, strict=True):
        results = sources[src_key]
        ax.axhline(0.0, color=RULE, linewidth=1, linestyle="--", zorder=1)

        for model in MODELS:
            per_family = results.get(model, {}).get("per_family", {})
            eff = np.array([per_family.get(f, {}).get("effect", np.nan) * PP for f in WORDINGS])
            lo = np.array([per_family.get(f, {}).get("lo", np.nan) * PP for f in WORDINGS])
            hi = np.array([per_family.get(f, {}).get("hi", np.nan) * PP for f in WORDINGS])
            for i, f in enumerate(WORDINGS):
                if f in per_family:
                    plotted.append({"panel": title, "model": model, "family": f,
                                    "effect_pp": eff[i], "lo_pp": lo[i], "hi_pp": hi[i]})
            s = MODEL_STYLE[model]
            # A shaded interval band reads as one object per model; error caps read as five.
            ax.fill_between(x, lo, hi, color=s["color"], alpha=0.16, linewidth=0, zorder=2)
            ax.plot(x, eff, color=s["color"], marker=s["marker"], markersize=4.2,
                    linewidth=1.3, label=s["label"], zorder=3,
                    markeredgecolor="#fcfcfb", markeredgewidth=0.6)

        # F1 is the pre-registered wording. A tick marker rather than the full-height shaded band an
        # earlier draft used, which drew the eye to the background instead of the curves.
        ax.plot([0], [ax.get_ylim()[0]], marker="^", markersize=4, color=MUTED,
                clip_on=False, zorder=5)
        ax.set_title(title, fontsize=7.6, color=INK, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(WORDINGS, fontsize=7.4)
        ax.set_xlim(-0.35, len(WORDINGS) - 0.65)
        ax.tick_params(axis="both", labelsize=7.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("effect on FAR (pp)", fontsize=7.6)
    axes[0].text(0.0, axes[0].get_ylim()[0], "  pre-registered", fontsize=6.2, color=MUTED,
                 va="bottom", ha="left")
    for ax in axes:
        ax.set_xlabel("prompt wording", fontsize=7.4, labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.10),
               ncol=len(MODELS), fontsize=6.8, frameon=False, columnspacing=1.6,
               handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plotted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(data_root()))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})

    make_fig0(out_dir / "fig0_design")
    # The CURRENT scored artefact, not the r4_factorial_F1_AN.json snapshot this read before: that
    # file predates the estimator correction, so the figure was drawing the superseded intervals
    # while every table beside it carried the corrected ones.
    fig1_rows = make_fig1(data_dir / "r4_factorial.json", out_dir / "fig1_r4")
    fig2_rows = make_fig2(data_dir / "template_robustness_R3u_R2.json",
                          data_dir / "template_robustness_R3_R3u.json",
                          out_dir / "fig2_templates")
    fig3_rows = make_fig3({f: data_dir / f"detection_scored_{f}.json" for f in WORDINGS},
                          out_dir / "fig3_quadrant",
                          reasoning_path=data_dir / "detection_scored_reasoning.json")

    for name in ("fig0_design", "fig1_r4", "fig2_templates", "fig3_quadrant"):
        p = out_dir / f"{name}.pdf"
        print(f"wrote {p} ({p.stat().st_size} bytes)")

    print("\n--- figure 1 data (percentage points) ---")
    for rec in fig1_rows:
        print(f"{rec['row']:38s} {rec['model']:18s} {rec['effect_pp']:8.3f} "
              f"[{rec['lo_pp']:7.3f},{rec['hi_pp']:7.3f}] p={rec['p']:.5f}")
    print("\n--- figure 2 data (percentage points) ---")
    for rec in fig2_rows:
        print(f"{rec['panel']:28s} {rec['model']:18s} {rec['family']:4s} "
              f"{rec['effect_pp']:7.3f} [{rec['lo_pp']:7.3f},{rec['hi_pp']:7.3f}]")
    print(f"\n--- figure 3: {len(fig3_rows)} points ---")
    for cond in QUADRANT_CONDS:
        g = [p for p in fig3_rows if p["cond"] == cond]
        if g:
            print(f"  {cond:4s} n={len(g):2d}  dc>0 in {sum(p['dc'] > 0 for p in g):2d}  "
                  f"dd'>0 in {sum(p['dd'] > 0 for p in g):2d}  "
                  f"survives in {sum(p['survives'] for p in g):2d}")


if __name__ == "__main__":
    main()
