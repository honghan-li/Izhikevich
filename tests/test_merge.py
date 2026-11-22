"""C8 — the §8.4 nameability merge and the name vocabulary."""
import numpy as np

from izhiclust.characterize import (MERGE_DESCRIPTORS, name_from_medians,
                                    name_key, nameability_merge)
from izhiclust.config import DEFAULT


def _desc(labels, rng, shift):
    """Descriptors drawn per cluster, shifted by `shift[cluster]`."""
    out = {}
    for j, d in enumerate(MERGE_DESCRIPTORS):
        v = rng.normal(1.0, 0.1, labels.size) + shift[labels] * (j + 1) * 0.0
        out[d] = v
    # the first descriptor carries the separation
    out[MERGE_DESCRIPTORS[0]] = rng.normal(1.0, 0.1, labels.size) \
        + shift[labels]
    return out


def test_indistinguishable_clusters_merge():
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2], 300)
    # clusters 0 and 1 are drawn from the same distribution; 2 is far away
    desc = _desc(labels, rng, np.array([0.0, 0.0, 5.0]))
    merged, rows, summary = nameability_merge(labels, desc, DEFAULT)
    assert summary["n_before"] == 3
    assert summary["n_after"] == 2
    assert np.unique(merged[labels == 0]).tolist() == \
        np.unique(merged[labels == 1]).tolist()
    assert np.unique(merged[labels == 2]).size == 1
    assert len(rows) == 1


def test_distinguishable_clusters_survive():
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1, 2], 300)
    desc = _desc(labels, rng, np.array([0.0, 3.0, 6.0]))
    merged, rows, summary = nameability_merge(labels, desc, DEFAULT)
    assert summary["n_after"] == 3
    assert rows == []


def test_merge_is_transitive():
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1, 2], 300)
    desc = _desc(labels, rng, np.array([0.0, 0.0, 0.0]))
    merged, rows, summary = nameability_merge(labels, desc, DEFAULT)
    assert summary["n_after"] == 1
    assert np.unique(merged[merged >= 0]).size == 1


def test_names_distinguish_bursting_from_tonic():
    cfg = DEFAULT
    tonic = {"rate": 40.0, "cv_isi": 0.01, "burst_index": 0.0,
             "spikes_per_burst": 1.0, "adapt_index": 0.0}
    burst = {"rate": 60.0, "cv_isi": 1.5, "burst_index": 0.6,
             "spikes_per_burst": 2.2, "adapt_index": 0.0}
    assert name_from_medians(tonic, cfg) == "medium tonic"
    nb = name_from_medians(burst, cfg)
    assert "bursting" in nb and "doublet" in nb
    # the name key strips the numbers but keeps the qualitative content
    assert name_key(nb) == "medium doublet bursting"
    adapting = dict(tonic, adapt_index=0.4)
    assert name_key(name_from_medians(adapting, cfg)) == "medium tonic, +adapting"
    irregular = dict(tonic, cv_isi=0.9)
    assert "+irregular" in name_from_medians(irregular, cfg)
    # the v1 name space had 6 keys for 102 clusters; the v2 vocabulary is wider
    keys = set()
    for rate in (10.0, 40.0, 150.0):
        for bi, spb in ((0.0, 1.0), (0.6, 2.2), (0.6, 3.0), (0.6, 5.0)):
            for ad in (0.0, 0.4, -0.4):
                keys.add(name_key(name_from_medians(
                    {"rate": rate, "cv_isi": 0.1, "burst_index": bi,
                     "spikes_per_burst": spb, "adapt_index": ad}, cfg)))
    assert len(keys) >= 8, sorted(keys)
