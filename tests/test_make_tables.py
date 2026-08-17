"""Tests for the artefact-driven LaTeX tables.

Both of the paper's cross-wording tables were maintained by hand and both went stale: the
combination count read 11 when 15 combinations existed, and the abstract's effect range never picked
up an $-11.50$ printed in the table beneath it. The generator exists so a number in the prose can be
checked against the artefacts; these tests cover the parts of it that could go wrong quietly.
"""

from __future__ import annotations

from pathlib import Path

from critxer.cli.make.tables import allcombos_table, cell, counts, write_table


def _row(effect, p):
    return {"effect": effect, "p": p}


def _fam(**models):
    return {m: {"episode_vs_filler": _row(*v[0]),
                "placement_assistant_minus_user": _row(*v[1]),
                "attribution_self_minus_peer": _row(*v[2])}
            for m, v in models.items()}


class TestCounts:
    """The counts the prose quotes, recomputed from what is on disk."""

    def test_screened_out_models_are_not_counted(self):
        """The screen is pre-hypothesis, so those models are reported but never counted.

        Left in, "15 of 15" would silently become "17 of 17" the moment a gemma cell was scored at
        another wording -- and would be wrong in the direction that inflates the claim.
        """
        by_fam = {"F1": _fam(**{
            "qwen3.6-27B": [(-0.04, 0.0001), (0.0, 0.9), (0.0, 0.9)],
            "gemma-4-31B": [(+0.02, 0.0001), (0.0, 0.9), (0.0, 0.9)],
        })}

        line = next(x for x in counts(by_fam) if "episode - AF" in x)

        assert "1 combinations" in line
        assert "1 exclude zero" in line

    def test_the_reported_range_spans_every_significant_combination(self):
        """The specific stale number: an -11.50 sat in the table while the abstract said -8.8."""
        by_fam = {
            "F1": _fam(**{"m": [(-0.088, 0.0001), (0.0, 0.9), (0.0, 0.9)]}),
            "F4": _fam(**{"m": [(-0.115, 0.0001), (0.0, 0.9), (0.0, 0.9)]}),
        }

        line = next(x for x in counts(by_fam) if "episode - AF" in x)

        assert "-11.50" in line

    def test_a_nonsignificant_combination_is_counted_but_not_ranged(self):
        by_fam = {"F1": _fam(**{
            "a": [(-0.04, 0.0001), (0.0, 0.9), (0.0, 0.9)],
            "b": [(-0.99, 0.6000), (0.0, 0.9), (0.0, 0.9)],
        })}

        line = next(x for x in counts(by_fam) if "episode - AF" in x)

        assert "2 combinations" in line
        assert "1 exclude zero" in line
        assert "-99" not in line


class TestTableRendering:
    def test_an_effect_is_starred_exactly_when_its_interval_excludes_zero(self):
        assert cell(_row(-0.04, 0.01)).endswith("$^{*}$")
        assert not cell(_row(-0.04, 0.20)).endswith("$^{*}$")

    def test_a_missing_contrast_renders_as_a_dash_rather_than_a_zero(self):
        """A cell we did not run is not a cell where the effect was zero."""
        assert cell(None) == "---"

    def test_effects_are_rendered_in_percentage_points(self):
        assert "-4.00" in cell(_row(-0.04, 0.01))

    def test_models_are_separated_by_a_rule_so_wordings_group_visibly(self):
        by_fam = {"F1": _fam(**{
            "qwen3.6-27B": [(-0.04, 0.01), (0.0, 0.9), (0.0, 0.9)],
            "ministral-14B": [(-0.08, 0.01), (0.0, 0.9), (0.0, 0.9)],
        })}

        tex = allcombos_table(by_fam)

        assert tex.count(r"\midrule") == 2  # the header rule, plus one between the two models
        assert "Ministral-3-14B" in tex and "Qwen3.6-27B" in tex

    def test_a_wording_with_no_artefact_is_skipped_rather_than_rendered_empty(self):
        by_fam = {"F1": _fam(**{"qwen3.6-27B": [(-0.04, 0.01), (0.0, 0.9), (0.0, 0.9)]})}

        tex = allcombos_table(by_fam)

        assert "F1" in tex
        assert "F3" not in tex


class TestWriteTable:
    """The default output directory is not in the repository, so it may not exist yet."""

    def test_a_missing_output_directory_is_created_rather_than_raising(self, tmp_path: Path):
        out = tmp_path / "paper" / "table.tex"

        write_table(out, "\\begin{table}\\end{table}\n")

        assert out.read_text().startswith("\\begin{table}")

    def test_an_existing_file_is_overwritten(self, tmp_path: Path):
        out = tmp_path / "table.tex"
        out.write_text("stale")

        write_table(out, "fresh")

        assert out.read_text() == "fresh"
