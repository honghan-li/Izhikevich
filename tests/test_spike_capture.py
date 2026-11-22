"""C2 — in-kernel spike-time capture (IMPROVEMENTS.md §3.1)."""
import numpy as np
import pytest
from dataclasses import replace

from izhiclust.config import DEFAULT, canonical_array
from izhiclust.simulate import grid_for_dt


@pytest.mark.gpu
def test_counts_and_times_agree_with_the_diagnostics(gpu, sim):
    cfg = DEFAULT
    P = canonical_array()
    res = sim.run(P, cfg.model.I_legacy, store_traces=False,
                  max_spikes=cfg.isi.s_max, refine=False)
    sp = res.spikes
    assert np.array_equal(sp.counts, res.diag["n_spikes_all"].astype(np.int64))
    assert not sp.overflow.any()
    for i in range(P.shape[0]):
        t = sp.get(i)
        assert np.all(np.diff(t) > 0)                      # monotonic
        assert t.min() > 0.0 and t.max() <= cfg.grid.t_total
        # window count matches the in-kernel steady-window counter
        m = (t >= cfg.grid.t_transient) & (t <= cfg.grid.t_steady_end)
        assert abs(int(m.sum()) - int(res.diag["n_spikes"][i])) <= 1


@pytest.mark.gpu
def test_times_are_at_dt_int_resolution(gpu, sim):
    """§3.1: times are written at `dt_int`, not at `dt_out = 0.1 ms`."""
    cfg = DEFAULT
    g = grid_for_dt(cfg.grid, 0.01)
    res = sim.run(np.array([[0.02, 0.2, -50.0, 2.0]]), cfg.model.I_legacy,
                  grid=g, store_traces=False, max_spikes=cfg.isi.s_max,
                  refine=False)
    t = res.spikes.get(0)
    # every time is a multiple of dt_int, and not of dt_out
    k = t / g.dt_int
    assert np.allclose(k, np.round(k), atol=5e-3)
    frac = np.abs(t / g.dt_out - np.round(t / g.dt_out))
    assert frac.max() > 0.05, "times look quantised to dt_out"


@pytest.mark.gpu
def test_overflow_flag_is_set_when_the_buffer_is_too_small(gpu, sim):
    cfg = DEFAULT
    res = sim.run(np.array([[0.02, 0.2, -50.0, 2.0]]), cfg.model.I_legacy,
                  store_traces=False, max_spikes=8, refine=False)
    assert res.spikes.overflow[0]
    assert res.spikes.counts[0] == 8
    assert res.diag["n_spikes_all"][0] > 8


@pytest.mark.gpu
def test_s_max_covers_the_fastest_trains(gpu, sim):
    """§3.1 sizing: `S_max = 4096` covers 2 kHz over the whole run."""
    cfg = DEFAULT
    rng = np.random.default_rng(1)
    p = cfg.space.to_native(rng.random((256, 5)))
    res = sim.run(p[:, :4], 20.0, store_traces=False,
                  max_spikes=cfg.isi.s_max, refine=False)
    assert not res.spikes.overflow.any()
    assert res.spikes.counts.max() < cfg.isi.s_max
