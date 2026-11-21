"""RawModule wrapper: chunked GPU integration plus the dt/4 refinement pass.

SPEC.md §7 (integrator), §7.3 (per-sample adaptivity), §7.5 (kernel layout).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from typing import Optional

import numpy as np

from .config import Config, Grid

NDIAG = 12
_DIAG_FIELDS = (
    "n_spikes", "isi_sum", "isi_sumsq", "isi_min", "isi_max", "first_isi",
    "last_isi", "mean_v", "sd_v", "max_local_error", "finite", "n_spikes_all",
)

_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "kernel.cu")


def grid_for_dt(base: Grid, dt: float, dt_out: Optional[float] = None) -> Grid:
    """A Grid at integration step `dt`, keeping `dt_out` (default: base's).

    `L` must be an integer and the transient must land on an output sample;
    both hold for every dt used by the convergence gate (§7.4).
    """
    target_out = base.dt_out if dt_out is None else dt_out
    L = max(1, int(round(target_out / dt)))
    g = replace(base, dt_int=dt, L=L)
    n_steps = g.t_total / g.dt_int
    if abs(n_steps - round(n_steps)) > 1e-9:
        raise ValueError(f"dt={dt} does not divide T_total={g.t_total}")
    k = g.t_transient / g.dt_out
    if abs(k - round(k)) > 1e-9:
        raise ValueError(f"transient {g.t_transient} not a multiple of dt_out={g.dt_out}")
    return g


@dataclass
class SimResult:
    traces: Optional[np.ndarray]        # (n, n_out) float32, analysis window
    u_traces: Optional[np.ndarray]
    diag: dict                          # name -> (n,) array
    isi_hist: np.ndarray                # (n, 32) int32
    spike_times: Optional[np.ndarray]   # (n, max_spikes) float32, NaN-padded
    grid: Grid
    n_refined: int = 0


class Simulator:
    """Compiles kernel.cu once and runs it in chunks."""

    def __init__(self, cfg: Config):
        import cupy as cp

        self.cp = cp
        self.cfg = cfg
        with open(_KERNEL_PATH, "r") as fh:
            src = fh.read()
        self.module = cp.RawModule(code=src, backend="nvrtc",
                                   options=("-std=c++14",))
        self.kernel = self.module.get_function("izhi_sweep")

    # ------------------------------------------------------------------
    def _launch(self, par_d, grid: Grid, *, store_traces, store_u, filter_on,
                max_spikes, traces_d=None, u_d=None, out_start=None, n_out=None):
        cp = self.cp
        cfg = self.cfg
        n = par_d.shape[0]
        dg = cfg.diag
        diag_d = cp.zeros((n, NDIAG), dtype=cp.float64)
        hist_d = cp.zeros((n, dg.n_isi_bins), dtype=cp.int32)
        dummy = cp.zeros(1, dtype=cp.float32)
        out_start = grid.out_start if out_start is None else out_start
        n_out = grid.n_out if n_out is None else n_out
        if store_traces and traces_d is None:
            traces_d = cp.empty((n_out, n), dtype=cp.float32)
        if store_u and u_d is None:
            u_d = cp.empty((n_out, n), dtype=cp.float32)
        spk_d = (cp.full((max_spikes, n), cp.nan, dtype=cp.float32)
                 if max_spikes > 0 else dummy)

        tpb = cfg.numerics.threads_per_block
        blocks = max(1, (n + tpb - 1) // tpb)
        log_lo = math.log10(dg.isi_lo)
        inv_span = 1.0 / (math.log10(dg.isi_hi) - log_lo)
        self.kernel(
            (blocks,), (tpb,),
            (np.int32(n), par_d,
             np.float64(cfg.model.I), np.float64(cfg.model.v0),
             np.float64(grid.dt_int), np.int64(grid.n_steps), np.int32(grid.L),
             np.int32(out_start), np.int32(n_out),
             np.int32(1 if store_traces else 0),
             np.int32(1 if filter_on else 0),
             np.int32(1 if store_u else 0),
             np.int64(grid.win_start_step), np.float64(cfg.model.v_peak),
             np.float64(log_lo), np.float64(inv_span), np.int32(dg.n_isi_bins),
             traces_d if store_traces else dummy,
             u_d if store_u else dummy,
             spk_d, np.int32(max_spikes), diag_d, hist_d),
        )
        return traces_d, u_d, diag_d, hist_d, spk_d

    # ------------------------------------------------------------------
    def run(self, params: np.ndarray, *, grid: Optional[Grid] = None,
            store_traces: bool = True, store_u: bool = False,
            filter_on: bool = True, max_spikes: int = 0,
            refine: bool = True, want_device_traces: bool = False,
            out_start: Optional[int] = None, n_out: Optional[int] = None):
        """Integrate one batch (<= chunk) of parameter sets.

        `params` is (n, 4) = (a, b, c, d).  Returns a SimResult; if
        `want_device_traces` the traces come back as a device array of shape
        (n, n_out) instead of a host array (saves a round trip before the FFT).
        """
        cp = self.cp
        grid = grid or self.cfg.grid
        par = np.ascontiguousarray(params, dtype=np.float64)
        n = par.shape[0]
        par_d = cp.asarray(par)

        traces_d, u_d, diag_d, hist_d, spk_d = self._launch(
            par_d, grid, store_traces=store_traces, store_u=store_u,
            filter_on=filter_on, max_spikes=max_spikes,
            out_start=out_start, n_out=n_out)

        diag = cp.asnumpy(diag_d)
        n_refined = 0
        if refine:
            tol = self.cfg.numerics.local_error_tol
            bad = np.where((diag[:, 9] > tol) | (diag[:, 10] < 0.5))[0]
            n_refined = int(bad.size)
            if n_refined:
                fine = grid_for_dt(grid, grid.dt_int / self.cfg.numerics.refine_factor,
                                   dt_out=grid.dt_out)
                sub = cp.asarray(np.ascontiguousarray(par[bad]))
                t2, u2, d2, h2, _ = self._launch(
                    sub, fine, store_traces=store_traces, store_u=store_u,
                    filter_on=filter_on, max_spikes=0,
                    out_start=out_start, n_out=n_out)
                diag[bad] = cp.asnumpy(d2)
                hist_d[bad] = h2
                if store_traces:
                    traces_d[:, bad] = t2
                if store_u:
                    u_d[:, bad] = u2
                del t2, u2, d2, h2, sub

        out_traces = None
        out_u = None
        if store_traces:
            out_traces = cp.ascontiguousarray(traces_d.T)
            del traces_d
            if not want_device_traces:
                h = cp.asnumpy(out_traces)
                del out_traces
                out_traces = h
        if store_u:
            out_u = cp.asnumpy(cp.ascontiguousarray(u_d.T))
            del u_d

        spikes = None
        if max_spikes > 0:
            spikes = cp.asnumpy(spk_d).T.copy()

        dd = {name: diag[:, j] for j, name in enumerate(_DIAG_FIELDS)}
        res = SimResult(traces=out_traces, u_traces=out_u, diag=dd,
                        isi_hist=cp.asnumpy(hist_d), spike_times=spikes,
                        grid=grid, n_refined=n_refined)
        del par_d, diag_d, hist_d, spk_d
        cp.get_default_memory_pool().free_all_blocks()
        return res


# ------------------------------------------------------- host-side diagnostics


def host_diagnostics(diag: dict, isi_hist: np.ndarray, cfg: Config) -> dict:
    """Derive rate, CV(ISI), burst index, spikes/burst, adaptation (§7.6)."""
    dg = cfg.diag
    win_s = cfg.grid.window_seconds
    n_spikes = diag["n_spikes"]
    n_isi = np.maximum(n_spikes - 1.0, 0.0)
    rate = n_spikes / win_s

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_isi = np.where(n_isi > 0, diag["isi_sum"] / np.maximum(n_isi, 1), np.nan)
        var_isi = np.where(
            n_isi > 1,
            diag["isi_sumsq"] / np.maximum(n_isi, 1) - mean_isi ** 2,
            np.nan)
        var_isi = np.where(np.isfinite(var_isi), np.maximum(var_isi, 0.0), np.nan)
        cv_isi = np.sqrt(var_isi) / mean_isi

    # log-ISI histogram bin edges (§7.6): 32 bins, 0.1 ms .. 2000 ms
    log_lo, log_hi = math.log10(dg.isi_lo), math.log10(dg.isi_hi)
    edges = np.logspace(log_lo, log_hi, dg.n_isi_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])

    thresh = cfg.naming.burst_isi_frac * mean_isi            # 0.25 * mean ISI
    below = np.zeros_like(rate)
    above = np.zeros_like(rate)
    hist = isi_hist.astype(np.float64)
    # fractional credit for the bin straddling the threshold
    for k in range(dg.n_isi_bins):
        lo, hi = edges[k], edges[k + 1]
        frac = np.clip((thresh - lo) / (hi - lo), 0.0, 1.0)
        frac = np.where(np.isfinite(frac), frac, 0.0)
        below += hist[:, k] * frac
        above += hist[:, k] * (1.0 - frac)
    total = below + above
    with np.errstate(invalid="ignore", divide="ignore"):
        burst_index = np.where(total > 0, below / np.maximum(total, 1e-12), 0.0)
        spikes_per_burst = np.where(above > 0, below / above + 1.0,
                                    np.where(below > 0, total + 1.0, 1.0))
        median_isi = _hist_median(hist, centres)
        f_i, l_i = diag["first_isi"], diag["last_isi"]
        adapt = np.where((f_i > 0) & (l_i > 0), (l_i - f_i) / (l_i + f_i), 0.0)

    return {
        "rate": rate,
        "mean_isi": mean_isi,
        "median_isi": median_isi,
        "cv_isi": cv_isi,
        "burst_index": burst_index,
        "spikes_per_burst": spikes_per_burst,
        "adapt_index": adapt,
        "n_isi": n_isi,
    }


def _hist_median(hist: np.ndarray, centres: np.ndarray) -> np.ndarray:
    tot = hist.sum(axis=1)
    cum = np.cumsum(hist, axis=1)
    out = np.full(hist.shape[0], np.nan)
    ok = tot > 0
    if ok.any():
        half = 0.5 * tot[ok][:, None]
        idx = (cum[ok] >= half).argmax(axis=1)
        out[ok] = centres[idx]
    return out
