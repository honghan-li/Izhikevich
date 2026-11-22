"""§8.1 — the representation-health gates, against the real run output.

v1 needed 92 of 128 components for 99% of the variance (§2.4) and accommodated
it by raising `n_components`; §8.1 turns that into a blocking gate at <= 20.
This test asserts the gate on whatever the last run actually produced.
"""
import json
import os

import numpy as np
import pytest

from izhiclust.cluster import fit_pca_adaptive
from izhiclust.config import DEFAULT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")


def _load():
    npz = os.path.join(DATA, "run_data.npz")
    if os.path.exists(npz):
        with np.load(npz, allow_pickle=False) as z:
            if "features_isi" in z:
                return z["features_isi"], z["class_Z"] | z["class_S"] | \
                    z["class_W"] | z["class_X"]
    return None, None


def test_pca_reaches_the_criterion_within_the_gate():
    F, held = _load()
    if F is None:
        pytest.skip("no v2 run output in ./data yet")
    cl = DEFAULT.clustering
    F = F[~held]
    scores, pca, curve = fit_pca_adaptive(F, 2, cl.pca_min_evr)
    k99 = int(np.searchsorted(curve, cl.pca_min_evr) + 1)
    print(f"\n  {k99} components for {cl.pca_min_evr:.0%} "
          f"(gate: <= {DEFAULT.gates.pca_max_components})")
    assert k99 <= DEFAULT.gates.pca_max_components, k99
    assert scores.shape == (F.shape[0], pca.n_components_)


def test_manifest_records_the_gates():
    path = os.path.join(OUT, "run_manifest.json")
    if not os.path.exists(path):
        pytest.skip("no run_manifest.json yet")
    with open(path) as fh:
        man = json.load(fh)
    if "health_gates" not in man:
        pytest.skip("manifest predates the §8.1 gates")
    h = man["health_gates"]
    assert h["pca_components_99"] <= h["pca_max_components"]
    assert h["near_over_far"] <= DEFAULT.gates.monotonicity_ratio
    assert h["octave_mean_distance"] >= h["ratio15_mean_distance"]
    assert man["pca"]["cumulative_at_selected"] >= man["pca"]["min_evr_criterion"]
