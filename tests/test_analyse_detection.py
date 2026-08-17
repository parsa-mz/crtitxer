"""Tests for the detection-arm analysis's failure modes.

The scoring itself is covered by `tests/test_detection.py`, which tests the metrics. This file
covers the part that bit: what happens when the artefacts on disk do not support the contrast being
asked for, and whether each contrast is built against the baseline it claims to be built against.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest


class TestMissingBaselineIsNamed:
    """A condition with no clean-arm R0 on its own item set must be reported, not crash.

    The detection arm at wordings F2-F5 hit this: the ladder was swept across wordings without
    re-measuring R0, because the template contrasts are rung-versus-rung and do not need it. The
    analysis then died with `KeyError: ('ministral-14B', '.../ladder')` -- which says nothing about
    what is missing or how to produce it, and had to be diagnosed by listing directories.
    """

    def test_the_message_names_the_model_the_wording_and_the_command_to_fix_it(self):
        from critxer.cli.analyse.detection import missing_baseline_message

        msg = missing_baseline_message("ministral-14B", "F2", ["R2", "R3", "R3u"], "/data/ladder")

        assert "ministral-14B" in msg
        assert "F2" in msg
        assert "R2" in msg
        assert "run-ladder" in msg
        assert "--conditions R0" in msg


def _cell(tmpdir, cond, probs, *, auditor="m", family="F1", episode_ids=None):
    """One persisted condition record, with only the fields the analysis reads."""
    n = len(probs)
    rec = {
        "auditor": auditor,
        "condition": cond,
        "family": family,
        "item_ids": [f"i{k}" for k in range(n)],
        "per_item_probs": list(probs),
        "per_item_confidence": [0.8] * n,
        "per_item_steps": [1] * n,
        "gold_labels": [1] * n,
    }
    if episode_ids is not None:
        rec["episode_ids"] = list(episode_ids)
    (tmpdir / f"{auditor}__{cond}__{family}.json").write_text(json.dumps(rec))


class TestEachContrastUsesTheBaselineOnItsOwnItemSet:
    """d' pairs a detection rate with a false-alarm rate, and both must come from the same arm.

    The clean arm exists at two sizes -- the ladder's 929 items and R4's 465-target subset -- and R0
    exists in both. The interval was built against the R0 from the condition's own directory while
    the *point estimate* differenced two rows of a table whose R0 row took its false-alarm rate from
    whichever directory was read first. So an R4 cell's estimate and its interval were centred on
    different baselines. On the real data that reported the 35B's headline AS cell as $-0.012$
    ("flat") where the consistent figure is $+0.030$.
    """

    def _run(self, tmp_path, monkeypatch, *, ladder_r0_far, r4_r0_far):
        """Score one R4-style cell against two clean directories with different R0 rates."""
        inc, ladder, r4 = tmp_path / "inc", tmp_path / "ladder", tmp_path / "r4"
        for d in (inc, ladder, r4):
            d.mkdir()
        rng = np.random.default_rng(0)
        n_clean, n_inc = 60, 60
        eps = [int(k) % 10 for k in range(n_clean)]

        # Incorrect arm: R0 and the AS cell, AS detecting slightly more.
        _cell(inc, "R0", rng.uniform(0.4, 0.6, n_inc))
        _cell(inc, "AS", rng.uniform(0.45, 0.65, n_inc), episode_ids=eps)
        # Two clean directories whose R0 false-alarm rates differ -- the situation that caused it.
        _cell(ladder, "R0", np.full(n_clean, ladder_r0_far))
        _cell(r4, "R0", np.full(n_clean, r4_r0_far))
        # AS's clean record lives in r4, so r4's R0 is the baseline it must be scored against.
        _cell(r4, "AS", np.full(n_clean, r4_r0_far - 0.03), episode_ids=eps)

        out = tmp_path / "scored.json"
        monkeypatch.setattr(sys, "argv", [
            "analyse-detection", "--incorrect-dir", str(inc),
            "--clean-dirs", f"{ladder},{r4}", "--family", "F1", "--out", str(out),
        ])
        monkeypatch.setattr("critxer.cli.analyse.detection.N_BOOT_SDT", 300)
        monkeypatch.setattr("critxer.cli.analyse.detection.N_BOOT", 300)
        from critxer.cli.analyse.detection import main

        main()
        return json.loads(out.read_text())["results"]["m"]["AS"]

    def test_the_point_estimate_does_not_move_when_an_unrelated_baseline_changes(
        self, tmp_path, monkeypatch, tmp_path_factory
    ):
        """AS is baselined on r4's R0, so the ladder's R0 must not enter its delta at all.

        This is the regression: the ladder's R0 is a *different item set*, present only because the
        ladder arm also lives on disk. Changing it changed the reported delta d' for an R4 cell.
        """
        a = self._run(tmp_path, monkeypatch, ladder_r0_far=0.20, r4_r0_far=0.25)
        b = self._run(tmp_path_factory.mktemp("b"), monkeypatch,
                      ladder_r0_far=0.40, r4_r0_far=0.25)

        assert a["delta_d_prime"]["effect"] == pytest.approx(b["delta_d_prime"]["effect"], abs=1e-9)
        assert a["delta_criterion"]["effect"] == pytest.approx(
            b["delta_criterion"]["effect"], abs=1e-9)

    def test_the_point_estimate_sits_inside_its_own_interval(self, tmp_path, monkeypatch):
        """The cheap end-to-end consistency check the mismatch would break.

        A plug-in estimate need not equal the replicate mean, but an estimate centred on one
        baseline and an interval centred on another can fall outside it entirely.
        """
        row = self._run(tmp_path, monkeypatch, ladder_r0_far=0.45, r4_r0_far=0.25)

        for key in ("delta_d_prime", "delta_criterion", "delta_balanced_accuracy"):
            d = row[key]
            assert d["lo"] <= d["effect"] <= d["hi"], (key, d)


class TestBalancedAccuracyChangeCarriesAnInterval:
    """The operating point is a claim, so it needs an interval like every other contrast.

    `balanced_accuracy` was persisted as a bare point estimate while `delta_d_prime` and
    `delta_criterion` beside it both carried clustered intervals. A false-alarm reduction that also
    raises balanced accuracy is the difference between "the auditor got more useful" and "the
    auditor got more lenient and we cannot tell", and with no interval the paper could only state
    the first descriptively. The three models' AS - R0 values are +2.29, +1.15 and +0.29pp; the last
    is small enough that whether it excludes zero decides how the sentence is written.
    """

    def _score(self, tmp_path, monkeypatch, *, det_base, det_cond, far_base, far_cond):
        inc, clean = tmp_path / "inc", tmp_path / "clean"
        for d in (inc, clean):
            d.mkdir()
        n = 60
        eps = [k % 10 for k in range(n)]
        _cell(inc, "R0", np.full(n, det_base))
        _cell(inc, "AS", np.full(n, det_cond), episode_ids=eps)
        _cell(clean, "R0", np.full(n, far_base))
        _cell(clean, "AS", np.full(n, far_cond), episode_ids=eps)

        out = tmp_path / "scored.json"
        monkeypatch.setattr(sys, "argv", [
            "analyse-detection", "--incorrect-dir", str(inc),
            "--clean-dirs", str(clean), "--family", "F1", "--out", str(out),
        ])
        monkeypatch.setattr("critxer.cli.analyse.detection.N_BOOT_SDT", 300)
        monkeypatch.setattr("critxer.cli.analyse.detection.N_BOOT", 300)
        from critxer.cli.analyse.detection import main

        main()
        return json.loads(out.read_text())["results"]["m"]["AS"]

    def test_the_effect_is_half_the_detection_gain_minus_half_the_false_alarm_gain(
        self, tmp_path, monkeypatch
    ):
        """Balanced accuracy is (TPR + TNR)/2, so its change is fixed by the two rate changes.

        Pinning the identity rather than the number is what catches the two ways this goes wrong
        quietly: differencing against the wrong directory's R0 (the bug `delta_d_prime` already has
        a regression test for), and dropping the factor of one half.
        """
        row = self._score(tmp_path, monkeypatch,
                          det_base=0.60, det_cond=0.66, far_base=0.30, far_cond=0.22)

        expected = 0.5 * ((0.66 - 0.60) - (0.22 - 0.30))
        assert row["delta_balanced_accuracy"]["effect"] == pytest.approx(expected, abs=1e-9)

    def test_a_flat_condition_gives_an_interval_that_contains_zero(self, tmp_path, monkeypatch):
        """The null case, so the interval is not vacuously wide or vacuously narrow."""
        row = self._score(tmp_path, monkeypatch,
                          det_base=0.60, det_cond=0.60, far_base=0.30, far_cond=0.30)

        d = row["delta_balanced_accuracy"]
        assert d["effect"] == pytest.approx(0.0, abs=1e-9)
        assert d["lo"] <= 0.0 <= d["hi"]

    def test_it_records_that_both_arms_were_clustered_on_the_episode(self, tmp_path, monkeypatch):
        """Same clustering as d' and c, and it says so, because 465 targets cycle 50 episodes."""
        row = self._score(tmp_path, monkeypatch,
                          det_base=0.60, det_cond=0.66, far_base=0.30, far_cond=0.22)

        d = row["delta_balanced_accuracy"]
        assert d["clustered_signal"] is True
        assert d["clustered_noise"] is True


class TestEveryQuantityAConclusionRestsOnGetsAFamily:
    """One evidentiary standard across the three signal-detection quantities.

    Correcting only `delta_d_prime` applies a stricter test to the hypothesis being rejected than to
    the one being kept, which declaring families in code is supposed to prevent. All three carry a
    Holm decision under their own family, and it rescues nothing: criterion still survives 13 of 15
    combinations against d-prime's 0 of 15.
    """

    def _score(self, tmp_path, monkeypatch):
        inc, clean = tmp_path / "inc", tmp_path / "clean"
        for d in (inc, clean):
            d.mkdir()
        n = 60
        eps = [k % 10 for k in range(n)]
        rng = np.random.default_rng(3)
        for cond, det, far in (("R0", 0.60, 0.30), ("AS", 0.66, 0.22), ("AV", 0.63, 0.26),
                               ("AF", 0.60, 0.30)):
            ep = None if cond == "R0" else eps
            _cell(inc, cond, np.clip(rng.normal(det, 0.03, n), 0, 1), episode_ids=ep)
            _cell(clean, cond, np.clip(rng.normal(far, 0.03, n), 0, 1), episode_ids=ep)
        out = tmp_path / "scored.json"
        monkeypatch.setattr(sys, "argv", [
            "analyse-detection", "--incorrect-dir", str(inc),
            "--clean-dirs", str(clean), "--family", "F1", "--out", str(out),
        ])
        monkeypatch.setattr("critxer.cli.analyse.detection.N_BOOT_SDT", 300)
        monkeypatch.setattr("critxer.cli.analyse.detection.N_BOOT", 300)
        from critxer.cli.analyse.detection import main

        main()
        return json.loads(out.read_text())["results"]["m"]["AS"]

    @pytest.mark.parametrize("quantity", ["delta_d_prime", "delta_criterion",
                                          "delta_balanced_accuracy"])
    def test_each_quantity_carries_a_holm_decision(self, tmp_path, monkeypatch, quantity):
        row = self._score(tmp_path, monkeypatch)

        assert "holm" in row[quantity], f"{quantity} is reported without a family correction"
        assert row[quantity]["holm"]["k"] >= 1

    def test_the_three_quantities_are_corrected_as_separate_families(self, tmp_path, monkeypatch):
        """Pooling them would test 27 hypotheses where the paper advances three sets of nine.

        Their family names must differ, or a survivor of one is being charged for the others' tests.
        """
        row = self._score(tmp_path, monkeypatch)

        names = {q: row[q]["holm"]["family"] for q in
                 ("delta_d_prime", "delta_criterion", "delta_balanced_accuracy")}
        assert len(set(names.values())) == 3, names


class TestTheDetectionContrastDoesNotDependOnIterationOrder:
    """`detection_vs_r0` drew from one shared generator, so a model's interval moved with its peers.

    This is the bug `resample.Streams` was written for, still live in this file after the d' and
    criterion contrasts beside it were converted: the models are iterated in sorted order, so every
    draw the first model consumes shifts the stream for the rest. A single-model fixture cannot see
    it, so the two models here differ in how many conditions they contribute.
    """

    def _score(self, tmp_path, monkeypatch, *, first_has_av: bool):
        inc, clean = tmp_path / "inc", tmp_path / "clean"
        for d in (inc, clean):
            d.mkdir()
        n = 60
        eps = [k % 10 for k in range(n)]
        for model in ("aaa-first", "zzz-second"):
            for cond, det, far in (("R0", 0.60, 0.30), ("AS", 0.66, 0.22), ("AV", 0.63, 0.26)):
                if model == "aaa-first" and cond == "AV" and not first_has_av:
                    continue
                ep = None if cond == "R0" else eps
                # Seeded per cell, not from one walking generator: drawing them in sequence would
                # make the second model's DATA change when the first model loses a condition, and
                # the test would fail for a reason that has nothing to do with the analysis.
                cell = np.random.default_rng([7, *[ord(c) for c in f"{model}/{cond}"]])
                _cell(inc, cond, np.clip(cell.normal(det, 0.03, n), 0, 1),
                      auditor=model, episode_ids=ep)
                _cell(clean, cond, np.clip(cell.normal(far, 0.03, n), 0, 1),
                      auditor=model, episode_ids=ep)
        out = tmp_path / "scored.json"
        monkeypatch.setattr(sys, "argv", [
            "analyse-detection", "--incorrect-dir", str(inc),
            "--clean-dirs", str(clean), "--family", "F1", "--out", str(out),
        ])
        monkeypatch.setattr("critxer.cli.analyse.detection.N_BOOT_SDT", 300)
        monkeypatch.setattr("critxer.cli.analyse.detection.N_BOOT", 300)
        from critxer.cli.analyse.detection import main

        main()
        return json.loads(out.read_text())["results"]["zzz-second"]["AS"]["detection_vs_r0"]

    def test_a_models_interval_is_unchanged_by_a_contrast_another_model_does_not_have(
        self, tmp_path, monkeypatch, tmp_path_factory
    ):
        with_av = self._score(tmp_path, monkeypatch, first_has_av=True)
        without = self._score(tmp_path_factory.mktemp("b"), monkeypatch, first_has_av=False)

        assert with_av == without, (
            "the second model's interval moved because the first model contributed a different "
            "number of contrasts to a shared generator"
        )

    def test_issuing_the_same_stream_identity_twice_raises(self):
        """The guard is the point of `Streams`: a copy-pasted call site must not go unnoticed."""
        from critxer.core.resample import Streams

        stream = Streams(1)
        stream("m", "detection_vs_r0")
        with pytest.raises(SystemExit, match="already issued"):
            stream("m", "detection_vs_r0")
