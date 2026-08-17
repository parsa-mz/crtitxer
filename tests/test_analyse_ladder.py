"""Tests for the cell-vs-R0 analysis, which serves two directories with different clustering needs.

`analyse-ladder --dir` is pointed at the ladder by default, whose rungs have no episode and are
correctly resampled by item. But `run_r4.py` ends by recommending `analyse-ladder --dir <outdir>`
for the R4 view, and those cells cycle 465 targets over 50 frozen episodes. Item-only resampling
there is the mistake the project has already paid for once, so it is pinned here rather than left
to the reader of two commands to notice.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest


def _cell(tmpdir, cond, probs, *, auditor="m", family="F1", episode_ids=None):
    """One persisted condition record, with only the fields the analysis reads."""
    n = len(probs)
    rec = {
        "auditor": auditor,
        "condition": cond,
        "family": family,
        "far": float(np.mean(probs)),
        "item_ids": [f"i{k}" for k in range(n)],
        "per_item_probs": list(probs),
        "per_item_flags": [[int(p > 0.5)] * 8 for p in probs],
    }
    if episode_ids is not None:
        rec["episode_ids"] = list(episode_ids)
    (tmpdir / f"{auditor}__{cond}__{family}.json").write_text(json.dumps(rec))


def _run(d, tmp_path, monkeypatch):
    out = tmp_path / "verdict.json"
    monkeypatch.setattr(sys, "argv", [
        "analyse-ladder", "--dir", str(d), "--family", "F1", "--out", str(out),
    ])
    monkeypatch.setattr("critxer.cli.analyse.ladder.N_BOOT", 400)
    from critxer.cli.analyse.ladder import main

    main()
    return json.loads(out.read_text())["results"]


class TestClusteringFollowsTheRecord:
    """A cell carrying `episode_ids` must be resampled on the episode, not on the item.

    465 targets over 50 episodes is 9.3x reuse, so item-only intervals are too narrow by roughly the
    design effect and the p-value moves across decision thresholds in both directions. The ladder's
    own rungs carry no episode and must stay item-only, so the rule has to be read off the record
    rather than off a flag the caller remembers to pass.
    """

    def _dir_with_episodes(self, tmp_path, *, with_episodes: bool):
        d = tmp_path / "cells"
        d.mkdir()
        n, n_ep = 200, 10
        eps = [f"ep{k % n_ep}" for k in range(n)] if with_episodes else None
        # Effect concentrated by episode: items sharing an episode move together, which is exactly
        # what item-only resampling cannot see.
        rng = np.random.default_rng(4)
        ep_shift = rng.normal(0, 0.08, n_ep)
        base = np.clip(rng.normal(0.30, 0.02, n), 0, 1)
        cond = np.clip(base + np.array([ep_shift[k % n_ep] for k in range(n)]), 0, 1)
        _cell(d, "R0", base, episode_ids=eps)
        _cell(d, "AS", cond, episode_ids=eps)
        return d

    def test_a_cell_with_episode_ids_reports_that_it_was_clustered(self, tmp_path, monkeypatch):
        d = self._dir_with_episodes(tmp_path, with_episodes=True)

        row = _run(d, tmp_path, monkeypatch)["m"]["rungs"]["AS"]

        assert row["clustered"] is True, "an episode-bearing cell was resampled by item"
        assert row["n_clusters"] == 10

    def test_a_ladder_rung_without_episodes_stays_item_only(self, tmp_path, monkeypatch):
        d = self._dir_with_episodes(tmp_path, with_episodes=False)

        row = _run(d, tmp_path, monkeypatch)["m"]["rungs"]["AS"]

        assert row["clustered"] is False
        assert row["n_paired"] == 200

    def test_clustering_widens_the_interval_when_the_effect_is_episode_borne(
        self, tmp_path, monkeypatch, tmp_path_factory
    ):
        """The point of the fix: the same data, clustered, is less precise and honestly so."""
        clustered = _run(self._dir_with_episodes(tmp_path, with_episodes=True),
                         tmp_path, monkeypatch)["m"]["rungs"]["AS"]
        b = tmp_path_factory.mktemp("b")
        item_only = _run(self._dir_with_episodes(b, with_episodes=False),
                         b, monkeypatch)["m"]["rungs"]["AS"]

        assert clustered["effect"] == pytest.approx(item_only["effect"], abs=1e-12)
        assert (clustered["hi"] - clustered["lo"]) > (item_only["hi"] - item_only["lo"])


class TestTheIntervalDoesNotDependOnIterationOrder:
    """Rungs and models drew from one walking generator, so each depended on what preceded it.

    Third instance of the bug `core.resample.Streams` was written for. A single-model fixture cannot
    see it, so the first model here contributes a different number of rungs.
    """

    def _dir(self, tmp_path, *, first_has_extra_rung: bool):
        d = tmp_path / "cells"
        d.mkdir()
        n = 120
        for model in ("aaa-first", "zzz-second"):
            conds = ["R0", "AS"] + (["AV"] if first_has_extra_rung or model == "zzz-second" else [])
            for cond in conds:
                # Seeded per cell so dropping a rung cannot change another model's data.
                cell = np.random.default_rng([11, *[ord(c) for c in f"{model}/{cond}"]])
                _cell(d, cond, np.clip(cell.normal(0.3, 0.03, n), 0, 1), auditor=model)
        return d

    def test_a_models_interval_is_unchanged_by_a_rung_another_model_does_not_have(
        self, tmp_path, monkeypatch, tmp_path_factory
    ):
        with_extra = _run(self._dir(tmp_path, first_has_extra_rung=True),
                          tmp_path, monkeypatch)["zzz-second"]["rungs"]["AS"]
        b = tmp_path_factory.mktemp("b")
        without = _run(self._dir(b, first_has_extra_rung=False), b, monkeypatch)[
            "zzz-second"]["rungs"]["AS"]

        assert with_extra == without, (
            "the second model's interval moved because the first model contributed a different "
            "number of rungs to a shared generator"
        )
