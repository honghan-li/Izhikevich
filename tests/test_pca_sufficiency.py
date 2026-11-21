"""§17.10 -- PCA must retain >= 99% of the variance of the real features.

§12.1 asserts 50 components suffice and instructs that if they fall short,
`n_components` is to be raised rather than the check relaxed.  On the real
feature matrix (with the §19 A1 noise floor removed, which decorrelates the
bands) 50 components retain ~85%, so the pipeline raises the count; this test
asserts the guarantee that actually matters -- that the component count the
pipeline uses reaches the criterion -- and records where 50 lands.
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
            return z["features"]
    csv = os.path.join(DATA, "features_bands.csv")
    if os.path.exists(csv):
        import pandas as pd
        df = pd.read_csv(csv)
        return df.drop(columns=["sample_id"]).to_numpy(dtype=np.float32)
    return None


def test_pca_reaches_the_criterion_on_real_features():
    F = _load()
    if F is None:
        pytest.skip("no run output in ./data yet; the pipeline asserts this at "
                    "run time and records the curve in run_manifest.json")
    cl = DEFAULT.clustering
    scores, pca, curve = fit_pca_adaptive(F, cl.pca_components, cl.pca_min_evr)
    evr = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    assert evr >= cl.pca_min_evr, (pca.n_components_, evr)
    assert scores.shape == (F.shape[0], pca.n_components_)
    print(f"\n  {pca.n_components_} components for "
          f"{cl.pca_min_evr:.0%}; at {cl.pca_components} components: "
          f"{curve[cl.pca_components - 1]:.4f}")


def test_manifest_records_the_variance_curve():
    path = os.path.join(OUT, "run_manifest.json")
    if not os.path.exists(path):
        pytest.skip("no run_manifest.json yet")
    with open(path) as fh:
        man = json.load(fh)
    pca = man["pca"]
    assert pca["cumulative_at_selected"] >= pca["min_evr_criterion"]
    assert len(pca["cumulative_explained_variance_curve"]) >= pca["n_components"]
    assert 0.0 < pca["cumulative_at_50"] <= 1.0
