"""Tests for the §19.B amendments: noise floor, adaptive PCA, naming rules."""
import numpy as np
import pytest
from dataclasses import replace

from izhiclust.characterize import name_from_medians
from izhiclust.cluster import fit_pca_adaptive
from izhiclust.config import DEFAULT
from izhiclust.features import band_features, make_plan, spectra


def _signal(bands_hz, amps, n, dt_out_ms):
    t = np.arange(n) * dt_out_ms * 1e-3
    x = np.zeros(n)
    for f, a in zip(bands_hz, amps):
        x += a * np.sin(2 * np.pi * f * t)
    return x.astype(np.float32)[None, :]


def test_noise_floor_removes_sub_floor_bands():
    """A1: content more than 40 dB below the peak must not reach the metric."""
    cfg = DEFAULT
    g = cfg.grid
    plan = make_plan(cfg, g, alpha=0.0)
    assert plan.noise_floor_frac == 0.01
    strong = _signal([120.0], [10.0], g.n_out, g.dt_out)
    rng = np.random.default_rng(0)
    # add a forest of tones 60 dB down -- below the -40 dB floor
    weak = np.zeros_like(strong)
    for f in rng.uniform(200.0, 900.0, 40):
        weak += _signal([float(f)], [10.0 * 1e-3], g.n_out, g.dt_out)
    F0 = np.asarray(band_features(spectra(strong, plan), plan))
    F1 = np.asarray(band_features(spectra(strong + weak, plan), plan))
    assert float(F0[0] @ F1[0]) > 0.999
    # and without the floor the same perturbation moves the vector a lot more
    plan_raw = replace(plan, noise_floor_frac=0.0)
    G0 = np.asarray(band_features(spectra(strong, plan_raw), plan_raw))
    G1 = np.asarray(band_features(spectra(strong + weak, plan_raw), plan_raw))
    assert float(G0[0] @ G1[0]) < float(F0[0] @ F1[0])


def test_noise_floor_keeps_real_structure():
    cfg = DEFAULT
    g = cfg.grid
    plan = make_plan(cfg, g, alpha=0.0)
    x = _signal([60.0, 180.0, 300.0], [10.0, 5.0, 2.5], g.n_out, g.dt_out)
    F = np.asarray(band_features(spectra(x, plan), plan))[0]
    nz = np.where(F > 0)[0]
    centres = plan.band_centres[nz]
    for f in (60.0, 180.0, 300.0):
        assert np.abs(centres - f).min() < 0.1 * f, (f, centres)


def test_adaptive_pca_reaches_the_criterion():
    rng = np.random.default_rng(0)
    # a matrix whose spectrum needs well over 50 components for 99%
    F = rng.normal(size=(800, 128)) * np.linspace(1.0, 0.25, 128)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    scores, pca, curve = fit_pca_adaptive(F, 50, 0.99)
    assert scores.shape[1] == pca.n_components_
    assert np.cumsum(pca.explained_variance_ratio_)[-1] >= 0.99
    assert pca.n_components_ >= 50
    assert curve[49] < 0.99 <= curve[pca.n_components_ - 1]


def test_adaptive_pca_does_not_shrink_below_the_floor():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(400, 3))
    F = base @ rng.normal(size=(3, 128))          # rank 3
    scores, pca, curve = fit_pca_adaptive(F, 50, 0.99)
    assert pca.n_components_ == 50


@pytest.mark.parametrize("med,expect", [
    ({"rate": 0.0, "n_spikes": 0, "ac_power": 1e9}, "subthreshold/resonant"),
    ({"rate": 0.0, "n_spikes": 0, "ac_power": 1.0}, "silent"),
    ({"rate": 3.0, "n_spikes": 6}, "very slow tonic"),
    ({"rate": 3.0, "n_spikes": 6, "burst_index": 0.6}, "very slow bursting"),
    ({"rate": 12.0, "n_spikes": 24}, "slow tonic"),
    ({"rate": 50.0, "n_spikes": 100}, "medium tonic"),
    ({"rate": 150.0, "n_spikes": 300}, "fast tonic"),
])
def test_naming_rules(med, expect):
    """A9: a slow spiker is named as a spiker, not as a resonator."""
    name = name_from_medians(med, DEFAULT, p_floor=1e5)
    assert name.startswith(expect), (name, expect)


def test_naming_tags():
    base = {"rate": 50.0, "n_spikes": 100}
    assert "+adapting" in name_from_medians({**base, "adapt_index": 0.4}, DEFAULT, 1e5)
    assert "+accelerating" in name_from_medians({**base, "adapt_index": -0.4}, DEFAULT, 1e5)
    assert "+irregular" in name_from_medians({**base, "cv_isi": 0.9}, DEFAULT, 1e5)
    n = name_from_medians({**base, "burst_index": 0.5, "spikes_per_burst": 4.0},
                          DEFAULT, 1e5)
    assert "bursting" in n and "4.0 spikes/burst" in n
    # a slow spiker still gets its adaptation tag (A9)
    assert "+adapting" in name_from_medians(
        {"rate": 2.0, "n_spikes": 4, "adapt_index": 0.5}, DEFAULT, 1e5)
