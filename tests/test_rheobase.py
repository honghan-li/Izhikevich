"""C3 — the rheobase-relative drive (IMPROVEMENTS.md §4)."""
import numpy as np
import pytest

from izhiclust import rheobase
from izhiclust.config import (DEFAULT, canonical_array, excitability_class,
                              i_andronov_hopf, i_saddle_node)


def test_analytic_values_match_the_spec_worked_example():
    """§4.1: at a = 0.02, b = 0.2, I_AH = 3.80 and I_SN = 4.00."""
    assert abs(float(i_saddle_node(0.2)) - 4.00) < 0.005  # scalar input
    assert abs(float(i_andronov_hopf(0.02, 0.2)) - 3.80) < 0.005
    # the Hopf precedes the fold wherever b > a, and is absent otherwise
    assert float(i_andronov_hopf(0.02, 0.2)) < float(i_saddle_node(0.2))
    assert np.isnan(float(i_andronov_hopf(0.3, 0.2)))
    assert excitability_class(np.array([0.3]), np.array([0.2]))[0] == 1
    assert excitability_class(np.array([0.02]), np.array([0.2]))[0] == 2


def test_saddle_node_is_where_the_equilibrium_disappears():
    """`I_SN` must be the `I` at which 0.04v^2 + (5-b)v + 140 + I loses roots."""
    for b in (-0.1, 0.0, 0.2, 0.35):
        I = float(i_saddle_node(np.array([b]))[0])
        for eps, expect_roots in ((-0.05, True), (+0.05, False)):
            disc = (5.0 - b) ** 2 - 0.16 * (140.0 + I + eps)
            assert (disc > 0) == expect_roots, (b, eps, disc)


@pytest.mark.gpu
def test_bisection_brackets_the_hybrid_rheobase(gpu, sim):
    cfg = DEFAULT
    P = canonical_array()
    r = rheobase.find_rheobase(sim, P, cfg)
    g = rheobase.bisection_grid(cfg)
    assert np.all(np.isfinite(r["i_star"])), r["i_star"]
    # just above I* the system fires in steady state; just below it does not
    above = sim.spike_counts(P, r["i_star"] + 1.0, g)
    below = sim.spike_counts(P, r["i_star"] - 1.0, g)
    assert np.all(above > 0), above
    assert np.all(below == 0), (below, r["i_star"])
    # the measured rheobase sits near the analytic bracket, not far from it
    assert np.all(r["i_star"] <= cfg.drive.i_max)


@pytest.mark.gpu
def test_production_drive_puts_the_canonical_seeds_at_i_10(gpu, sim):
    cfg = DEFAULT
    P = canonical_array()
    r = rheobase.find_rheobase(sim, P, cfg)
    delta = np.full(P.shape[0], 3.0)
    I = rheobase.production_drive(r["i_star"], delta, cfg,
                                  np.arange(P.shape[0]))
    assert np.allclose(I, cfg.drive.canonical_drive)
    I2 = rheobase.production_drive(r["i_star"], delta, cfg, None)
    assert np.allclose(I2, r["i_star"] + 3.0)


@pytest.mark.gpu
def test_drive_removes_the_quiescent_population(gpu, sim):
    """§4.3: every sample fires by construction, where v1 lost 45.1%."""
    cfg = DEFAULT
    rng = np.random.default_rng(0)
    x = rng.random((64, 5))
    p = cfg.space.to_native(x)
    r = rheobase.find_rheobase(sim, p[:, :4], cfg)
    I = rheobase.production_drive(r["i_star"], p[:, 4], cfg)
    g = rheobase.bisection_grid(cfg)
    cnt = sim.spike_counts(p[:, :4], I, g)
    ok = np.isfinite(r["i_star"])
    assert ok.mean() > 0.9, ok.mean()
    assert (cnt[ok] > 0).mean() > 0.95, (cnt[ok] == 0).sum()
