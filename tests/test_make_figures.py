"""Tests for the design figure's panel selection.

Figures are otherwise untested on purpose: they are presentation code, and the numbers they draw are
pinned where they are computed. Panel *selection* is different -- a silently dropped or mislabelled
panel produces a figure that still compiles, still looks plausible, and no longer matches its
caption.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import pytest

from critxer.cli.make.figures import PANEL_ORDER, make_fig0


def test_the_default_is_all_three_panels_in_the_declared_order():
    assert PANEL_ORDER == ("pipeline", "ladder", "arms")


def test_an_unknown_panel_name_raises_rather_than_being_skipped(tmp_path):
    with pytest.raises(SystemExit, match="controls"):
        make_fig0(tmp_path / "fig", panels=("pipeline", "controls"))


def test_a_subset_renders_and_writes_both_formats(tmp_path):
    make_fig0(tmp_path / "fig", panels=("pipeline", "arms"))

    assert (tmp_path / "fig.pdf").exists()
    assert (tmp_path / "fig.png").exists()


def test_panels_are_lettered_by_position_not_by_identity():
    """With the ladder dropped, the two-arm panel must be "(b)", not still "(c)".

    The letters are part of the caption's contract with the reader. Hard-coding them in each panel's
    own drawing code is what made this a real risk.
    """
    from critxer.cli.make.figures import panel_letter

    assert panel_letter(("pipeline", "arms"), "arms") == "b"
    assert panel_letter(PANEL_ORDER, "arms") == "c"


def test_requesting_no_panels_raises(tmp_path):
    with pytest.raises(SystemExit):
        make_fig0(tmp_path / "fig", panels=())


# --- the quadrant figure's reasoning-enabled overlay --------------------------------------------
#
# The reasoning check is one condition measured at one wording, so it gets its own marker rather
# than joining the main series: thinking and token budget both differ, and a reader who reads its
# magnitude against the others is reading a two-way confound. What is comparable is which side of
# each axis it falls on, which is the whole point of this figure.

def _scored(cond_deltas):
    return {"results": {"qwen3.6-27B": cond_deltas}}


def test_the_reasoning_cell_is_a_separate_series(tmp_path):
    from critxer.cli.make.figures import make_fig3

    cell = {"delta_d_prime": {"effect": -0.005}, "delta_criterion": {"effect": 0.087}}
    main = tmp_path / "detection_scored_F1.json"
    main.write_text(json.dumps(_scored({"AS": cell})))
    reasoning = tmp_path / "detection_scored_reasoning.json"
    reasoning.write_text(json.dumps(_scored({"AS": cell})))

    rows = make_fig3({"F1": main}, tmp_path / "fig3", reasoning_path=reasoning)

    assert sorted(r["cond"] for r in rows) == ["AS", "AS_reasoning"]


def test_a_missing_reasoning_artefact_is_not_an_error(tmp_path):
    """The figure predates this arm and must still build without it."""
    from critxer.cli.make.figures import make_fig3

    main = tmp_path / "detection_scored_F1.json"
    main.write_text(json.dumps(_scored(
        {"AS": {"delta_d_prime": {"effect": 0.09}, "delta_criterion": {"effect": 0.06}}})))

    rows = make_fig3({"F1": main}, tmp_path / "fig3", reasoning_path=tmp_path / "absent.json")

    assert [r["cond"] for r in rows] == ["AS"]
