"""Tests for the R4 contrast assembly in ``critxer analyse-r4``.

The metrics and the cell construction are covered elsewhere (``test_r4.py``, ``test_resample.py``).
What this file covers is the part where a wrong answer is silent: which contrasts get built at all,
and which clustering each one is reported under. A contrast missing from the output cannot be
claimed; a contrast clustered on one pool when it spans two reports an interval that is too narrow,
and nothing about the number looks wrong.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from critxer.core.r4 import CELLS


def _cell(d, model, cond, probs, *, family="F1", episodes=None, fingerprint="pool-a"):
    """One persisted R4 cell, with only the fields the analysis reads."""
    n = len(probs)
    rec = {
        "auditor": model,
        "condition": cond,
        "family": family,
        "item_ids": [f"i{k}" for k in range(n)],
        "per_item_probs": list(probs),
        "far": float(np.mean(probs)),
        "episode_ids": list(episodes) if episodes is not None else [k % 10 for k in range(n)],
        "episode_fingerprint": fingerprint,
    }
    (d / f"{model}__{cond}__{family}.json").write_text(json.dumps(rec))


def _run(tmp_path, monkeypatch, cells: dict[str, list[float]]):
    """Score one model's worth of cells and return its row of contrasts."""
    d = tmp_path / "r4"
    d.mkdir()
    # The clean pool and the incorrect-verdict pool are different frozen episode sets, and
    # `check_pool_fingerprints` enforces that they carry different fingerprints.
    incorrect = {"AX", "AXN"}
    for cond, probs in cells.items():
        _cell(d, "m", cond, probs,
              fingerprint="pool-x" if cond in incorrect else "pool-a")
    out = tmp_path / "factorial.json"
    monkeypatch.setattr(sys, "argv", [
        "analyse-r4", "--dir", str(d), "--family", "F1", "--out", str(out),
    ])
    monkeypatch.setattr("critxer.cli.analyse.r4.N_BOOT", 400)
    from critxer.cli.analyse.r4 import main

    main()
    return json.loads(out.read_text())["results"]["m"]


def _flat(rng, n, value):
    return list(np.clip(rng.normal(value, 0.02, n), 0, 1))


@pytest.fixture
def cells():
    """A full set of cells: the 2x2, the three controls, and the incorrect-verdict pair."""
    rng = np.random.default_rng(0)
    n = 60
    base = {c: _flat(rng, n, 0.20) for c in CELLS}
    return {
        "R0": _flat(rng, n, 0.24),
        **base,
        "AF": _flat(rng, n, 0.24),
        "AV": _flat(rng, n, 0.22),
        "AN": _flat(rng, n, 0.23),
        "AX": _flat(rng, n, 0.14),
        "AXN": _flat(rng, n, 0.15),
    }


class TestVerdictPolarityIsItsOwnContrast:
    """AXN - AN varies the episode's verdict and nothing else, and the paper claims from it.

    The decomposition needs three separable components: what a repair *request* elicits (AS - AN),
    what the repair *content* adds (AX - AXN), and what the *verdict* alone does. A contrast a claim
    rests on has to be built here, where it is corrected, not beside the paper --
    `continuation_presence_av_minus_an` asks a different question and does not substitute.
    """

    def test_the_polarity_contrast_is_reported(self, tmp_path, monkeypatch, cells):
        row = _run(tmp_path, monkeypatch, cells)

        assert "polarity_axn_minus_an" in row

    def test_it_is_the_mean_difference_between_the_two_inert_continuation_cells(
        self, tmp_path, monkeypatch, cells
    ):
        """Both sides carry an inert continuation, so only the audit's verdict differs."""
        row = _run(tmp_path, monkeypatch, cells)

        expected = float(np.mean(cells["AXN"]) - np.mean(cells["AN"]))
        assert row["polarity_axn_minus_an"]["effect"] == pytest.approx(expected, abs=1e-9)

    def test_it_is_not_narrower_than_either_single_pool_clustering(
        self, tmp_path, monkeypatch, cells
    ):
        """It spans two disjoint episode pools, so neither grouping alone covers the dependence.

        `verdict_composition_ax_minus_as` already reports the wider of the two for exactly this
        reason. Clustering this one on a single pool would understate the interval, which is the
        failure that looks like precision.
        """
        row = _run(tmp_path, monkeypatch, cells)

        d = row["polarity_axn_minus_an"]
        width = d["hi"] - d["lo"]
        single = row["genuine_repair_ax_minus_axn"]
        assert width >= (single["hi"] - single["lo"]) * 0.5, (d, single)
        assert d["lo"] <= d["effect"] <= d["hi"]

    def test_it_is_absent_rather_than_null_when_the_incorrect_pool_was_not_run(
        self, tmp_path, monkeypatch, cells
    ):
        """AX/AXN exist only where the natural-fault pool was generated; say so by omission."""
        without = {k: v for k, v in cells.items() if k not in ("AX", "AXN")}

        row = _run(tmp_path, monkeypatch, without)

        assert "polarity_axn_minus_an" not in row


class TestEveryClaimedContrastIsAFamilyMember:
    """A contrast the paper draws a conclusion from must be corrected, or it is uncorrected.

    Two were not. `audit_plus_inert_vs_filler` (AN - AF) is the *identified* audit-alone contrast --
    the paper reads it in preference to AV - AF precisely because it holds a continuation present on
    both sides -- and it had no declared family, so its significance was being read off the interval
    alone. That is the second criterion the statistical conventions forbid, and it decided a "two of
    three" count. `polarity_axn_minus_an` had the same problem by not existing.
    """

    def test_every_declared_member_is_actually_produced(self, tmp_path, monkeypatch, cells):
        from critxer.cli.analyse.r4 import CLAIM_CONTRASTS

        row = _run(tmp_path, monkeypatch, cells)

        missing = [c for c in CLAIM_CONTRASTS if c not in row]
        assert missing == [], f"declared but never computed: {missing}"

    @pytest.mark.parametrize("contrast", ["audit_plus_inert_vs_filler", "polarity_axn_minus_an"])
    def test_the_contrasts_the_paper_reads_carry_a_holm_decision(
        self, tmp_path, monkeypatch, cells, contrast
    ):
        row = _run(tmp_path, monkeypatch, cells)

        assert "holm" in row[contrast], f"{contrast} is claimed from but never corrected"
        assert row[contrast]["holm"]["family"] == "r4-claims"


class TestAContrastsIntervalDependsOnlyOnItsOwnData:
    """One shared RNG makes every interval depend on what was analysed before it.

    Models are iterated in sorted order, so draws consumed by whichever sorts first shift the stream
    for the rest -- enough to move a p-value across a Holm threshold with the model's own data
    untouched. The property to hold is the strong one: a model's numbers must not depend on which
    other models were in the directory. A single-model fixture cannot see this, which is why the
    first version of this test passed against the bug.
    """

    def _two_models(self, tmp_path, cells, *, first_has_pool: bool):
        d = tmp_path / "r4"
        d.mkdir()
        incorrect = {"AX", "AXN"}
        for model in ("aaa-first", "zzz-second"):
            for cond, probs in cells.items():
                if model == "aaa-first" and not first_has_pool and cond in incorrect:
                    continue
                _cell(d, model, cond, probs,
                      fingerprint="pool-x" if cond in incorrect else "pool-a")
        return d

    def _score(self, d, tmp_path, monkeypatch):
        out = tmp_path / "factorial.json"
        monkeypatch.setattr(sys, "argv", [
            "analyse-r4", "--dir", str(d), "--family", "F1", "--out", str(out),
        ])
        monkeypatch.setattr("critxer.cli.analyse.r4.N_BOOT", 400)
        from critxer.cli.analyse.r4 import main

        main()
        return json.loads(out.read_text())["results"]

    def test_one_models_numbers_do_not_move_when_another_model_gains_contrasts(
        self, tmp_path, monkeypatch, cells, tmp_path_factory
    ):
        a_dir = self._two_models(tmp_path, cells, first_has_pool=True)
        b_root = tmp_path_factory.mktemp("b")
        b_dir = self._two_models(b_root, cells, first_has_pool=False)

        a = self._score(a_dir, tmp_path, monkeypatch)["zzz-second"]
        b = self._score(b_dir, b_root, monkeypatch)["zzz-second"]

        for name in ("episode_vs_filler", "audit_only_vs_filler", "polarity_axn_minus_an",
                     "repair_contribution_as_minus_an", "repair_contribution_as_minus_av"):
            assert a[name]["p"] == pytest.approx(b[name]["p"], abs=1e-12), (
                name, a[name]["p"], b[name]["p"])
            assert a[name]["lo"] == pytest.approx(b[name]["lo"], abs=1e-12), name
            assert a[name]["hi"] == pytest.approx(b[name]["hi"], abs=1e-12), name

    def test_two_identical_runs_agree_exactly(self, tmp_path, monkeypatch, cells,
                                              tmp_path_factory):
        """The weaker property, which held already; kept so a reseed cannot break it."""
        a = _run(tmp_path, monkeypatch, cells)
        b = _run(tmp_path_factory.mktemp("c"), monkeypatch, cells)

        assert a["episode_vs_filler"]["p"] == pytest.approx(b["episode_vs_filler"]["p"], abs=1e-12)
