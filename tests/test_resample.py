"""Tests for cluster-aware resampling.

R4 pairs 465 targets with 50 frozen episodes, so every episode is reused for about 9.3 targets.
Bootstrapping targets alone treats those 9.3 as independent observations when they share whatever
idiosyncratic effect their episode has, which understates the variance of every R4 contrast. These
tests pin the two properties that make the fix a fix: episodes are resampled as whole units, and
the resulting interval is wider than the item-only one whenever the episode actually matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from critxer.core.resample import (
    cluster_bootstrap,
    cluster_indices,
    episode_ids_for,
    interval_from_reps,
    paired_bootstrap,
    two_stage_bootstrap,
)


def _null_clustered_sample(rng, n_clusters=50, per_cluster=9, cluster_sd=0.05, item_sd=0.15):
    """One draw from a null world with R4's geometry: 50 episodes, ~9 targets each, true effect 0.

    The parameters are the study's, not round numbers: 465 targets over a pool of 50 is 9.3 per
    episode, and the between/within ratio is roughly what the real contrasts show. Coverage is a
    property of a design, so measuring it anywhere else would not tell us about this one.
    """
    offsets = rng.normal(0.0, cluster_sd, size=n_clusters)
    diff = np.repeat(offsets, per_cluster) + rng.normal(0.0, item_sd, n_clusters * per_cluster)
    return diff, np.repeat(np.arange(n_clusters), per_cluster)


def _coverage(estimator, trials=200, n_boot=1200):
    """How often the estimator's 95% interval contains the true effect of zero."""
    rng = np.random.default_rng(20260807)
    hits = 0
    for t in range(trials):
        diff, clusters = _null_clustered_sample(rng)
        out = estimator(diff, clusters, np.random.default_rng(t), n_boot=n_boot)
        hits += out["lo"] <= 0.0 <= out["hi"]
    return hits / trials


class TestTheClusterBootstrapIsCalibrated:
    """The property that matters and that no earlier test checked: does the 95% interval cover 95%?

    Every other test here asks whether the clustered interval is *wider* than the item one, which
    it must be -- but "wider" has no upper bound, and the estimator this replaces was wider than
    correct. An over-covering interval is not the safe direction: it silently withdraws real
    effects, and it withdrew one from this paper (AS - AV, reported as null at p = 0.068, which is
    p = 0.020 once the estimator is right).
    """

    def test_the_interval_covers_at_about_its_nominal_rate(self):
        assert 0.92 <= _coverage(cluster_bootstrap) <= 0.97

    def test_the_two_stage_estimator_it_replaces_over_covers(self):
        """Pinned so the defect stays visible rather than becoming folklore in a commit message.

        Resampling items *within* each drawn cluster adds a second source of variance that the
        sampling distribution of the cluster mean does not have. The cluster mean is already an
        estimate of that cluster's contribution; re-drawing its members perturbs it again.
        """
        assert _coverage(two_stage_bootstrap) > 0.97


class TestTheClusterBootstrapWidensWhenEpisodesMatter:
    """The whole point: if a grouping carries signal, ignoring it fakes precision."""

    def _clustered_data(self, n_clusters=20, per_cluster=10, cluster_sd=0.5, seed=7):
        """Differences whose variation is almost entirely *between* clusters, not within.

        Constructed this way because that is the regime the R4 design is in: two targets sharing
        an episode see the same prior exchange, so their difference from baseline moves together.
        """
        rng = np.random.default_rng(seed)
        offsets = rng.normal(0.0, cluster_sd, size=n_clusters)
        diff, clusters = [], []
        for c, off in enumerate(offsets):
            diff.extend(off + rng.normal(0.0, 0.02, size=per_cluster))
            clusters.extend([c] * per_cluster)
        return np.array(diff), np.array(clusters)

    def test_the_interval_is_wider_than_ignoring_clusters(self):
        diff, clusters = self._clustered_data()
        rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)

        naive = paired_bootstrap(diff, rng_a, n_boot=4000)
        clustered = cluster_bootstrap(diff, clusters, rng_b, n_boot=4000)

        naive_width = naive["hi"] - naive["lo"]
        clustered_width = clustered["hi"] - clustered["lo"]
        assert clustered_width > 2 * naive_width

    def test_the_point_estimate_is_unchanged(self):
        """Only the uncertainty is wrong in the naive version, not the effect."""
        diff, clusters = self._clustered_data()

        naive = paired_bootstrap(diff, np.random.default_rng(1), n_boot=2000)
        clustered = cluster_bootstrap(diff, clusters, np.random.default_rng(1), n_boot=2000)

        assert clustered["effect"] == pytest.approx(naive["effect"])

    def test_it_reports_how_many_clusters_it_had(self):
        """A contrast resting on 50 episodes must not be read as resting on 465 items."""
        diff, clusters = self._clustered_data(n_clusters=20)

        out = cluster_bootstrap(diff, clusters, np.random.default_rng(1), n_boot=500)

        assert out["n_clusters"] == 20
        assert out["n"] == 200

    def test_nans_are_dropped_with_their_cluster_labels(self):
        """A NaN is a parse failure, i.e. missing data; dropping it must not shift labels."""
        diff, clusters = self._clustered_data(n_clusters=4, per_cluster=5)
        diff[3] = np.nan

        out = cluster_bootstrap(diff, clusters, np.random.default_rng(1), n_boot=500)

        assert out["n"] == 19
        assert out["n_clusters"] == 4

    def test_one_item_per_cluster_reduces_to_the_item_bootstrap(self):
        """A sanity anchor: with no reuse there is no clustering to correct for."""
        rng = np.random.default_rng(3)
        diff = rng.normal(0.1, 1.0, size=200)
        clusters = np.arange(200)

        naive = paired_bootstrap(diff, np.random.default_rng(5), n_boot=6000)
        clustered = cluster_bootstrap(diff, clusters, np.random.default_rng(5), n_boot=6000)

        naive_width = naive["hi"] - naive["lo"]
        clustered_width = clustered["hi"] - clustered["lo"]
        assert clustered_width == pytest.approx(naive_width, rel=0.2)

    def test_a_single_cluster_is_refused(self):
        """With one episode there is nothing to resample and the CI would be meaningless."""
        with pytest.raises(ValueError, match="at least 2 clusters"):
            cluster_bootstrap(np.zeros(10), np.zeros(10), np.random.default_rng(1))


class TestEpisodeIdsAreRecoverableForRunsThatDidNotRecordThem:
    """`run_r4.py` assigns target *i* the episode at ``i % len(pool)`` and did not persist it.

    The rule is deterministic, so the assignment for already-generated cells can be reconstructed
    rather than regenerated -- but only if the reconstruction is pinned by a test, because getting
    it silently wrong would produce confident, wrongly-clustered intervals.
    """

    def test_it_cycles_through_the_pool_in_order(self):
        assert episode_ids_for(n_items=7, n_episodes=3).tolist() == [0, 1, 2, 0, 1, 2, 0]

    def test_every_episode_is_used_about_equally(self):
        ids = episode_ids_for(n_items=465, n_episodes=50)
        counts = np.bincount(ids)

        assert counts.max() - counts.min() <= 1
        assert len(counts) == 50

    def test_a_pool_larger_than_the_targets_uses_each_episode_once(self):
        assert episode_ids_for(n_items=3, n_episodes=10).tolist() == [0, 1, 2]

    def test_an_empty_pool_is_refused(self):
        with pytest.raises(ValueError, match="n_episodes"):
            episode_ids_for(n_items=5, n_episodes=0)


class TestTheReportedPValueAgreesWithTheReportedInterval:
    """One significance criterion, not two that can disagree.

    Both a percentile interval and a null-centred p-value were reported for every contrast, and
    they are different tests: the interval asks whether zero falls outside the middle 95% of the
    bootstrap distribution, the centred p-value asks how often a distribution shifted to zero
    exceeds the observed effect. They coincide only when the replicate distribution is symmetric.
    On the real data they disagreed -- AS - AV on Qwen3.6-27B came out [-2.65, -0.07] with
    p = 0.065 -- so a table could mark the same contrast significant by its CI and null by its p,
    which is what happened in an earlier draft of the paper.

    The reported ``p`` is therefore the achieved significance level *of the percentile interval*:
    twice the smaller tail mass on either side of zero. It cannot contradict the interval by
    construction. The old quantity is kept as ``p_centred`` so the two remain comparable and the
    change is auditable rather than silent.
    """

    def _skewed(self, seed: int, shift: float) -> np.ndarray:
        """Right-skewed differences whose mean sits near alpha -- the regime where the two differ.

        Exponential rather than normal because that is what makes the replicate distribution
        asymmetric; on symmetric data the two definitions agree and the test would prove nothing.
        """
        return np.random.default_rng(seed).exponential(1.0, size=150) - shift

    def test_p_is_not_below_alpha_when_the_interval_includes_zero(self):
        """The concrete disagreement, in the direction that overstates significance.

        Under the centred definition this case reports p = 0.044 with an interval of
        [-0.363, +0.003] -- significant by the p-value, null by the CI printed beside it.
        """
        out = paired_bootstrap(self._skewed(1, 1.25), np.random.default_rng(11), n_boot=4000)
        assert out["lo"] < 0 < out["hi"], out

        assert out["p"] >= 0.05, out

    def test_p_below_alpha_exactly_when_the_interval_excludes_zero(self):
        for seed, shift in ((1, 1.25), (7, 1.25), (8, 1.15), (23, 1.35), (25, 1.31),
                            (0, 0.5), (2, 1.6), (4, 2.0)):
            out = paired_bootstrap(self._skewed(seed, shift), np.random.default_rng(11),
                                   n_boot=4000)
            excludes_zero = out["lo"] > 0 or out["hi"] < 0

            assert excludes_zero == (out["p"] < 0.05), (seed, shift, out)

    def test_the_centred_p_value_is_still_reported_under_its_own_name(self):
        out = paired_bootstrap(self._skewed(1, 1.25), np.random.default_rng(11), n_boot=4000)

        assert out["p_centred"] < 0.05 <= out["p"], out

    def test_the_clustered_interval_agrees_with_its_own_p_value_too(self):
        rng = np.random.default_rng(4)
        offsets = rng.normal(0.35, 0.4, size=25)
        diff = np.concatenate([off + rng.normal(0, 0.05, size=8) for off in offsets])
        clusters = np.repeat(np.arange(25), 8)

        out = cluster_bootstrap(diff, clusters, np.random.default_rng(9), n_boot=4000)

        assert (out["lo"] > 0 or out["hi"] < 0) == (out["p"] < 0.05), out


class TestIntervalFromReps:
    """The same interval-and-p logic, exposed for replicates this module did not generate.

    The detection arm builds its own replicates -- each one recomputes d' from a resampled clean
    arm and a resampled incorrect arm, which is not a paired difference over one item set -- but it
    must not therefore grow its own significance criterion. It had one, and it disagreed with this
    module's by construction.
    """

    def test_it_agrees_with_the_paired_bootstrap_on_the_same_replicates(self):
        rng = np.random.default_rng(2)
        d = rng.normal(0.3, 1.0, size=100)
        reps = d[np.random.default_rng(5).integers(0, 100, size=(4000, 100))].mean(axis=1)

        direct = interval_from_reps(reps, float(d.mean()), n_boot=4000)

        assert direct["p"] < 0.05
        assert direct["lo"] > 0
        assert direct["effect"] == float(d.mean())

    def test_the_point_estimate_is_the_callers_not_the_replicate_mean(self):
        """d' is a nonlinear function of two rates, so the plug-in estimate is not the boot mean."""
        reps = np.linspace(-1.0, 3.0, 4000)

        assert interval_from_reps(reps, 0.5, n_boot=4000)["effect"] == 0.5


class TestClusterIndices:
    """The cluster draw exposed as indices, for statistics that are not a paired mean.

    d' is a function of two rates measured on two disjoint arms, so it cannot go through
    `cluster_bootstrap`, which averages one difference vector. The detection analysis therefore
    built its own loop -- and resampled items, while the paper's statistical conventions claimed
    every R4 interval was clustered. The R4 detection cells reuse 50 episodes over 929 targets,
    18.6x, worse than the 9.3x on the clean arm that the paper says cost it a claim.
    """

    def test_it_draws_whole_clusters(self):
        clusters = np.repeat(np.arange(4), 3)
        idx = cluster_indices(clusters, np.random.default_rng(0))

        # Every drawn index belongs to one of the clusters, and the count matches.
        assert idx.size == clusters.size
        assert set(np.unique(clusters[idx])).issubset(set(np.unique(clusters)))

    def test_the_drawn_cluster_count_equals_the_original(self):
        clusters = np.repeat(np.arange(7), 5)

        idx = cluster_indices(clusters, np.random.default_rng(1))

        assert idx.size == clusters.size

    def test_a_drawn_cluster_arrives_with_its_members_intact(self):
        """The property that separates this from the estimator it replaced.

        Clusters are the sampling unit; their contents are not resampled. If a cluster is drawn
        twice its members appear twice, each time complete -- so every drawn cluster's contribution
        to the replicate mean is that cluster's own mean, not a perturbation of it.
        """
        clusters = np.repeat(np.arange(6), 4)
        idx = cluster_indices(clusters, np.random.default_rng(2))

        drawn, counts = np.unique(clusters[idx], return_counts=True)
        assert set(counts.tolist()) <= {4, 8, 12, 16, 20, 24}, dict(zip(drawn, counts, strict=True))

    def test_it_reproduces_the_cluster_bootstrap_interval_on_a_paired_mean(self):
        """Cross-check that both callers share one draw: same machinery, same answer."""
        rng = np.random.default_rng(3)
        offsets = rng.normal(0.4, 0.5, size=20)
        d = np.concatenate([o + rng.normal(0, 0.05, size=8) for o in offsets])
        clusters = np.repeat(np.arange(20), 8)

        ref = cluster_bootstrap(d, clusters, np.random.default_rng(9), n_boot=3000)
        r2 = np.random.default_rng(9)
        reps = np.array([d[cluster_indices(clusters, r2)].mean() for _ in range(3000)])
        mine = interval_from_reps(reps, float(d.mean()), n_boot=3000)

        assert mine["lo"] == pytest.approx(ref["lo"], abs=0.01)
        assert mine["hi"] == pytest.approx(ref["hi"], abs=0.01)

    def test_it_is_wider_than_item_resampling_when_clusters_carry_signal(self):
        rng = np.random.default_rng(5)
        offsets = rng.normal(0.0, 0.6, size=15)
        d = np.concatenate([o + rng.normal(0, 0.02, size=10) for o in offsets])
        clusters = np.repeat(np.arange(15), 10)

        r = np.random.default_rng(2)
        clus = np.array([d[cluster_indices(clusters, r)].mean() for _ in range(2000)])
        item = np.array([d[r.integers(0, d.size, d.size)].mean() for _ in range(2000)])

        assert clus.std() > 2 * item.std()


class TestIndependentStreamsPerContrast:
    """Each contrast needs its own RNG, or its interval depends on what ran before it.

    Both analyses in this project shared one generator across every model x contrast, and both were
    bitten. In the detection arm the 27B's delta-d' read p = 0.036 in one run and 0.028 in the next,
    straddling a Holm threshold of 0.0167. In the R4 arm the models are iterated in sorted order, so
    `ministral-14B` sorts first and every contrast it consumed shifted the stream for both Qwen
    models -- adding one missing contrast to Ministral moved the 27B's AS - AV from p = 0.0067 to
    0.0090 with its own data untouched.

    The reuse guard is the other half. Deriving the seed from a name is only reproducible if no two
    contrasts share a name, and the obvious way to break that is a copy-pasted call site, which is
    invisible in review and silently recouples the two contrasts.
    """

    def test_the_same_identity_gives_the_same_draws(self):
        from critxer.core.resample import Streams

        a = Streams(seed=7)("m", "episode_vs_filler").integers(0, 100, 5)
        b = Streams(seed=7)("m", "episode_vs_filler").integers(0, 100, 5)

        assert list(a) == list(b)

    def test_different_contrasts_get_different_draws(self):
        from critxer.core.resample import Streams

        s = Streams(seed=7)
        a = s("m", "episode_vs_filler").integers(0, 1_000_000, 8)
        b = s("m", "audit_only_vs_filler").integers(0, 1_000_000, 8)

        assert list(a) != list(b)

    def test_different_models_get_different_draws(self):
        from critxer.core.resample import Streams

        s = Streams(seed=7)
        a = s("aaa", "episode_vs_filler").integers(0, 1_000_000, 8)
        b = s("zzz", "episode_vs_filler").integers(0, 1_000_000, 8)

        assert list(a) != list(b)

    def test_a_contrasts_draws_do_not_depend_on_what_was_issued_before_it(self):
        """The property the shared generator lacked, stated directly."""
        from critxer.core.resample import Streams

        plain = Streams(seed=7)("m", "target").integers(0, 1_000_000, 8)
        crowded = Streams(seed=7)
        for name in ("a", "b", "c", "d"):
            crowded("other", name).integers(0, 10, 50)
        after = crowded("m", "target").integers(0, 1_000_000, 8)

        assert list(plain) == list(after)

    def test_reusing_one_identity_raises_rather_than_recoupling_two_contrasts(self):
        from critxer.core.resample import Streams

        s = Streams(seed=7)
        s("m", "episode_vs_filler")

        with pytest.raises(SystemExit, match="episode_vs_filler"):
            s("m", "episode_vs_filler")

    def test_the_same_name_under_a_different_model_is_not_a_reuse(self):
        from critxer.core.resample import Streams

        s = Streams(seed=7)
        s("aaa", "episode_vs_filler")

        s("zzz", "episode_vs_filler")
