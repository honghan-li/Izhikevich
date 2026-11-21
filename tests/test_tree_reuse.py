"""§17.9 -- reused single-linkage tree must label identically to a full fit.

Guards the semi-internal `hdbscan.hdbscan_._tree_to_labels` API used by the
§12.4.1 sweep: a version bump that changes it must fail loudly here.
"""
import numpy as np
import pytest

from izhiclust.cluster import build_tree, full_fit, labels_from_tree


@pytest.fixture(scope="module")
def embedding():
    rng = np.random.default_rng(0)
    cent = rng.normal(0, 6, (12, 10))
    lab = rng.integers(0, 12, 3000)
    return (cent[lab] + rng.normal(0, 1.1, (3000, 10))).astype(np.float32)


@pytest.mark.parametrize("min_samples", [5, 10])
def test_reused_tree_labels_identically(embedding, min_samples):
    slt, mst = build_tree(embedding, min_samples, core_dist_n_jobs=4)
    for mcs in (30, 60, 120):
        for method in ("leaf", "eom"):
            got, _ = labels_from_tree(embedding, slt, mcs, method)
            c = full_fit(embedding, min_cluster_size=mcs,
                         min_samples=min_samples, method=method,
                         prediction_data=False, core_dist_n_jobs=4)
            assert np.array_equal(got, c.labels_), (min_samples, mcs, method)


def test_relative_validity_shim_matches_library(embedding):
    from izhiclust.cluster import relative_validity, relative_validity_contributions
    c = full_fit(embedding, min_cluster_size=60, min_samples=10, method="leaf",
                 prediction_data=False, core_dist_n_jobs=4)
    rv = relative_validity(c._min_spanning_tree, c.labels_)
    assert abs(rv - c.relative_validity_) < 1e-12
    contrib = relative_validity_contributions(c._min_spanning_tree, c.labels_)
    assert abs(contrib.sum() - c.relative_validity_) < 1e-10
