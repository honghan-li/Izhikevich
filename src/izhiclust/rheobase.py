"""C3 — drive relative to the numerically determined rheobase (§4).

`I` is no longer fixed.  Per parameter set `theta = (a,b,c,d)` the rheobase of
the *hybrid* system (ODE + reset) is bisected on "does it spike in steady
state", bracketed by the analytic saddle-node / Andronov-Hopf values (§4.1),
and the production drive is `I = I*(theta) + Delta` with `Delta` the fifth
design axis (§4.2).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from .config import (Config, Grid, excitability_class, i_andronov_hopf,
                     i_saddle_node)

# status codes recorded per sample in samples.csv
OK = 0
NO_RHEOBASE = 1        # no spiking at any I <= I_max
BRACKET_LOW = 2        # already spiking at the lower bracket (bistable corner)


def bisection_grid(cfg: Config) -> Grid:
    """The §4.2 search grid: 500 ms, spikes in the last 200 ms decide."""
    dr = cfg.drive
    t_transient = dr.t_total - dr.t_measure
    L = max(1, int(round(0.1 / dr.dt_int)))
    return Grid(t_total=dr.t_total, t_transient=t_transient,
                t_steady_end=dr.t_total, window_shift=0.0,
                dt_int=dr.dt_int, L=L)


def analytic_bracket(params: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """`I_SN`, `I_AH`, the excitability class and the lower bracket (§4.1)."""
    a, b = params[:, 0], params[:, 1]
    i_sn = i_saddle_node(b)
    i_ah = i_andronov_hopf(a, b)
    lo = np.where(np.isfinite(i_ah), np.minimum(i_sn, i_ah), i_sn)
    return {"i_sn": i_sn, "i_ah": i_ah,
            "exc_class": excitability_class(a, b),
            "bracket_lo": lo - cfg.drive.bracket_pad}


def find_rheobase(sim, params: np.ndarray, cfg: Config,
                  log=None) -> Dict[str, np.ndarray]:
    """Vectorised bisection over one batch of parameter sets.

    Returns `i_star` (the smallest drive at which the hybrid system fires in
    steady state, to `drive.tol`), the analytic brackets, the onset firing rate
    just above `i_star` (the f-I onset-shape classifier of §4.3) and a status
    code per sample.
    """
    dr = cfg.drive
    g = bisection_grid(cfg)
    n = params.shape[0]
    br = analytic_bracket(params, cfg)
    lo = br["bracket_lo"].astype(np.float64).copy()
    hi = np.full(n, dr.i_max, dtype=np.float64)
    status = np.zeros(n, dtype=np.int8)

    fires_hi = sim.spike_counts(params, hi, g) > 0
    status[~fires_hi] = NO_RHEOBASE

    # The reset supplies a finite kick, so spiking can persist below the
    # analytic onset (§4.1 item 3).  Walk the lower bracket down a bounded
    # number of times before giving up on it (assumption V2-A8).
    fires_lo = sim.spike_counts(params, lo, g) > 0
    for _ in range(2):
        if not (fires_lo & fires_hi).any():
            break
        m = fires_lo & fires_hi
        lo[m] -= 5.0
        fires_lo[m] = sim.spike_counts(params[m], lo[m], g) > 0
    status[fires_lo & fires_hi] = BRACKET_LOW

    live = fires_hi & ~fires_lo
    n_iter = 0
    while live.any() and n_iter < dr.max_iter:
        width = hi - lo
        if np.all(width[live] <= dr.tol):
            break
        mid = 0.5 * (lo + hi)
        idx = np.where(live & (width > dr.tol))[0]
        if idx.size == 0:
            break
        fires = sim.spike_counts(params[idx], mid[idx], g) > 0
        hi[idx[fires]] = mid[idx[fires]]
        lo[idx[~fires]] = mid[idx[~fires]]
        n_iter += 1

    i_star = np.where(fires_hi, hi, np.nan)
    i_star = np.where(status == BRACKET_LOW, lo, i_star)

    # f-I onset shape (§4.3): the rate just above threshold
    onset_rate = np.zeros(n, dtype=np.float64)
    ok = np.isfinite(i_star)
    if ok.any():
        cnt = sim.spike_counts(params[ok], i_star[ok], g)
        onset_rate[ok] = cnt / (dr.t_measure / 1000.0)

    if log is not None:
        log(f"  rheobase: {int(ok.sum())}/{n} bracketed "
            f"({int((status == NO_RHEOBASE).sum())} no rheobase <= "
            f"{dr.i_max:g}, {int((status == BRACKET_LOW).sum())} bistable "
            f"lower bracket), {n_iter} bisection rounds, median I* "
            f"{np.nanmedian(i_star):.2f}")
    return {"i_star": i_star, "i_sn": br["i_sn"], "i_ah": br["i_ah"],
            "exc_class": br["exc_class"], "onset_rate": onset_rate,
            "status": status, "bracket_lo": lo, "bracket_hi": hi,
            "n_bisections": np.full(n, n_iter, dtype=np.int32)}


def production_drive(i_star: np.ndarray, delta: np.ndarray, cfg: Config,
                     canonical_rows: Optional[np.ndarray] = None) -> np.ndarray:
    """`I = I* + Delta`, except the canonical seeds, which keep v1's `I = 10`.

    Samples with no rheobase in range get `I = I_max` so they are still
    integrated once and reported as `Z`/`S` (§3.4, §4.2) rather than skipped.
    """
    I = i_star + delta
    I = np.where(np.isfinite(I), I, cfg.drive.i_max)
    if canonical_rows is not None and len(canonical_rows):
        rows = np.asarray(canonical_rows, dtype=np.int64)
        keep = np.isfinite(i_star[rows]) & (i_star[rows] < cfg.drive.canonical_drive)
        I[rows[keep]] = cfg.drive.canonical_drive
    return I
