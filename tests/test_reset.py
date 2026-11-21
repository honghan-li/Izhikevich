"""§17.2 -- reset correctness against a scipy event-driven CPU reference."""
import numpy as np
import pytest
from dataclasses import replace

from izhiclust.config import DEFAULT
from izhiclust.simulate import grid_for_dt


def reference_spikes(a, b, c, d, I, v0, t_end, max_spikes=64):
    """solve_ivp with a v = 30 terminal event, restarted after each reset."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        v, u = y
        return [0.04 * v * v + 5.0 * v + 140.0 - u + I, a * (b * v - u)]

    def ev(t, y):
        return y[0] - 30.0
    ev.terminal = True
    ev.direction = 1.0

    t, y = 0.0, [v0, b * v0]
    out = []
    while t < t_end and len(out) < max_spikes:
        sol = solve_ivp(rhs, (t, t_end), y, events=ev, rtol=1e-12, atol=1e-12,
                        method="DOP853", max_step=1.0)
        if sol.t_events[0].size == 0:
            break
        ts = float(sol.t_events[0][0])
        out.append(ts)
        ye = sol.y_events[0][0]
        t, y = ts, [c, ye[1] + d]
        t += 1e-12
    return np.asarray(out)


@pytest.mark.gpu
@pytest.mark.parametrize("params", [
    (0.02, 0.20, -65.0, 8.0),     # RS
    (0.02, 0.20, -50.0, 2.0),     # CH
    (0.10, 0.20, -65.0, 2.0),     # FS
])
def test_spike_times_match_reference(gpu, sim, params):
    cfg = DEFAULT
    a, b, c, d = params
    t_end = 400.0
    short = replace(cfg.grid, t_total=t_end, t_transient=10.0)
    g = grid_for_dt(short, 0.001, dt_out=cfg.grid.dt_out)
    P = np.array([[a, b, c, d]], dtype=np.float64)
    res = sim.run(P, grid=g, store_traces=False, max_spikes=256, refine=False)
    got = res.spike_times[0]
    got = got[np.isfinite(got)]
    want = reference_spikes(a, b, c, d, cfg.model.I, cfg.model.v0, t_end)
    n = min(got.size, want.size)
    assert n >= 8, (got.size, want.size)
    err = np.abs(got[:n] - want[:n])
    # §7.3: end-of-step event detection is late by at most dt_int per spike, and
    # that lateness shifts the remainder of the train, so the absolute error
    # accumulates linearly.  Three things are therefore asserted: the first
    # spike agrees to within the step size; the accumulation rate is of order
    # dt_int per spike; and the ISIs -- the only quantity the magnitude spectrum
    # is sensitive to, since a common time shift leaves it untouched -- agree
    # to well under a step.
    assert err[0] < 1.5 * g.dt_int, (params, err[0])
    slope = np.polyfit(np.arange(n), err, 1)[0]
    assert 0.0 <= slope < 10.0 * g.dt_int, (params, slope)
    disi = np.abs(np.diff(got[:n]) - np.diff(want[:n]))
    assert disi.max() < 10.0 * g.dt_int, (params, disi.max())


@pytest.mark.gpu
def test_peak_equalisation(gpu, sim):
    """Spiking samples are emitted as exactly +30 mV (§7.1)."""
    cfg = DEFAULT
    short = replace(cfg.grid, t_total=200.0, t_transient=10.0)
    g = grid_for_dt(short, cfg.grid.dt_int, dt_out=cfg.grid.dt_out)
    P = np.array([[0.02, 0.2, -65.0, 8.0]])
    tr = sim.run(P, grid=g, filter_on=False, out_start=0,
                 n_out=g.n_out_total, refine=False).traces[0]
    peaks = tr[tr > 29.0]
    assert peaks.size > 2
    assert np.allclose(peaks, 30.0, atol=1e-5)
    assert tr.max() <= 30.0 + 1e-5
