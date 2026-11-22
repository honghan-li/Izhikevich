"""RawModule wrapper: chunked GPU integration plus the dt/4 refinement pass.

SPEC.md v1 §7 (integrator), v1 §7.3 (per-sample adaptivity), v1 §7.5 (kernel
layout).  IMPROVEMENTS.md §3.1 (spike-time capture), §4 (per-sample drive).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from typing import List, Optional, Sequence

import numpy as np

from .config import Config, Grid

NDIAG = 14
_DIAG_FIELDS = (
    "n_spikes", "isi_sum", "isi_sumsq", "isi_min", "isi_max", "first_isi",
    "last_isi", "mean_v", "sd_v", "max_local_error", "finite", "n_spikes_all",
    "n_spikes_recorded", "overflow",
)

_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "kernel.cu")


def grid_for_dt(base: Grid, dt: float, dt_out: Optional[float] = None) -> Grid:
    """A Grid at integration step `dt`, keeping `dt_out` (default: base's)."""
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


# ------------------------------------------------------------- spike trains


@dataclass
class SpikeTrains:
    """Ragged spike times (ms), CSR-style: `times[offsets[i]:offsets[i+1]]`."""
    times: np.ndarray            # float32, concatenated
    offsets: np.ndarray          # int64, (n + 1,)
    overflow: np.ndarray         # bool, (n,)

    @property
    def n(self) -> int:
        return self.offsets.size - 1

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def get(self, i: int) -> np.ndarray:
        return self.times[self.offsets[i]:self.offsets[i + 1]]

    def subset(self, idx: Sequence[int]) -> "SpikeTrains":
        idx = np.asarray(idx, dtype=np.int64)
        parts = [self.times[self.offsets[i]:self.offsets[i + 1]] for i in idx]
        counts = np.array([p.size for p in parts], dtype=np.int64)
        off = np.zeros(idx.size + 1, dtype=np.int64)
        np.cumsum(counts, out=off[1:])
        times = (np.concatenate(parts) if parts else
                 np.zeros(0, dtype=np.float32))
        return SpikeTrains(times.astype(np.float32, copy=False), off,
                           self.overflow[idx].copy())

    @staticmethod
    def concat(parts: Sequence["SpikeTrains"]) -> "SpikeTrains":
        parts = [p for p in parts if p is not None]
        if not parts:
            return SpikeTrains(np.zeros(0, np.float32), np.zeros(1, np.int64),
                               np.zeros(0, bool))
        times = np.concatenate([p.times for p in parts])
        counts = np.concatenate([p.counts for p in parts])
        off = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=off[1:])
        return SpikeTrains(times, off,
                           np.concatenate([p.overflow for p in parts]))

    @staticmethod
    def from_dense(dense: np.ndarray, counts: np.ndarray,
                   overflow: np.ndarray) -> "SpikeTrains":
        """`dense` is (max_spikes, n) as written by the kernel."""
        counts = counts.astype(np.int64)
        off = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=off[1:])
        times = np.empty(int(off[-1]), dtype=np.float32)
        for i in range(counts.size):
            times[off[i]:off[i + 1]] = dense[:counts[i], i]
        return SpikeTrains(times, off, overflow.astype(bool))

    def replace_rows(self, rows: np.ndarray, other: "SpikeTrains") -> "SpikeTrains":
        """Return a copy with `rows` taken from `other` (row k <- other row k)."""
        rows = np.asarray(rows, dtype=np.int64)
        parts: List[np.ndarray] = []
        counts = self.counts.copy()
        pos = {int(r): k for k, r in enumerate(rows)}
        for i in range(self.n):
            if i in pos:
                parts.append(other.get(pos[i]))
            else:
                parts.append(self.get(i))
            counts[i] = parts[-1].size
        off = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(counts, out=off[1:])
        ov = self.overflow.copy()
        ov[rows] = other.overflow
        return SpikeTrains(np.concatenate(parts).astype(np.float32), off, ov)


@dataclass
class SimResult:
    traces: Optional[np.ndarray]        # (n, n_out) float32
    u_traces: Optional[np.ndarray]
    diag: dict                          # name -> (n,) array
    isi_hist: np.ndarray                # (n, 32) int32
    spikes: Optional[SpikeTrains]
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
    def _launch(self, par_d, drive_d, grid: Grid, *, store_traces, store_u,
                peak_hold, max_spikes, traces_d=None, u_d=None,
                out_start=None, n_out=None):
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
        spk_d = (cp.zeros((max_spikes, n), dtype=cp.float32)
                 if max_spikes > 0 else dummy)

        tpb = cfg.numerics.threads_per_block
        blocks = max(1, (n + tpb - 1) // tpb)
        log_lo = math.log10(dg.isi_lo)
        inv_span = 1.0 / (math.log10(dg.isi_hi) - log_lo)
        self.kernel(
            (blocks,), (tpb,),
            (np.int32(n), par_d, drive_d,
             np.float64(cfg.model.v0),
             np.float64(grid.dt_int), np.int64(grid.n_steps), np.int32(grid.L),
             np.int32(out_start), np.int32(n_out),
             np.int32(1 if store_traces else 0),
             np.int32(1 if peak_hold else 0),
             np.int32(1 if store_u else 0),
             np.int64(grid.win_start_step), np.int64(grid.win_end_step),
             np.float64(cfg.model.v_peak),
             np.float64(log_lo), np.float64(inv_span), np.int32(dg.n_isi_bins),
             traces_d if store_traces else dummy,
             u_d if store_u else dummy,
             spk_d, np.int32(max_spikes), diag_d, hist_d),
        )
        return traces_d, u_d, diag_d, hist_d, spk_d

    # ------------------------------------------------------------------
    def run(self, params: np.ndarray, drive, *, grid: Optional[Grid] = None,
            store_traces: bool = True, store_u: bool = False,
            peak_hold: bool = True, max_spikes: int = 0,
            refine: bool = True, want_device_traces: bool = False,
            out_start: Optional[int] = None, n_out: Optional[int] = None):
        """Integrate one batch (<= chunk) of parameter sets.

        `params` is (n, 4) = (a, b, c, d); `drive` is the per-sample `I`
        (scalar or (n,)).  Returns a SimResult.
        """
        cp = self.cp
        grid = grid or self.cfg.grid
        par = np.ascontiguousarray(params[:, :4], dtype=np.float64)
        n = par.shape[0]
        dr = np.broadcast_to(np.asarray(drive, dtype=np.float64), (n,))
        dr = np.ascontiguousarray(dr)
        par_d = cp.asarray(par)
        drive_d = cp.asarray(dr)

        traces_d, u_d, diag_d, hist_d, spk_d = self._launch(
            par_d, drive_d, grid, store_traces=store_traces, store_u=store_u,
            peak_hold=peak_hold, max_spikes=max_spikes,
            out_start=out_start, n_out=n_out)

        diag = cp.asnumpy(diag_d)
        spikes = None
        if max_spikes > 0:
            spikes = SpikeTrains.from_dense(cp.asnumpy(spk_d), diag[:, 12],
                                            diag[:, 13] > 0.5)
        n_refined = 0
        if refine:
            tol = self.cfg.numerics.local_error_tol
            bad = np.where((diag[:, 9] > tol) | (diag[:, 10] < 0.5))[0]
            n_refined = int(bad.size)
            if n_refined:
                fine = grid_for_dt(grid, grid.dt_int / self.cfg.numerics.refine_factor,
                                   dt_out=grid.dt_out)
                sub = cp.asarray(np.ascontiguousarray(par[bad]))
                sub_dr = cp.asarray(np.ascontiguousarray(dr[bad]))
                t2, u2, d2, h2, s2 = self._launch(
                    sub, sub_dr, fine, store_traces=store_traces,
                    store_u=store_u, peak_hold=peak_hold, max_spikes=max_spikes,
                    out_start=out_start, n_out=n_out)
                d2h = cp.asnumpy(d2)
                if max_spikes > 0:
                    fine_sp = SpikeTrains.from_dense(cp.asnumpy(s2), d2h[:, 12],
                                                     d2h[:, 13] > 0.5)
                    spikes = spikes.replace_rows(bad, fine_sp)
                diag[bad] = d2h
                hist_d[bad] = h2
                if store_traces:
                    traces_d[:, bad] = t2
                if store_u:
                    u_d[:, bad] = u2
                del t2, u2, d2, h2, s2, sub, sub_dr

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

        dd = {name: diag[:, j] for j, name in enumerate(_DIAG_FIELDS)}
        res = SimResult(traces=out_traces, u_traces=out_u, diag=dd,
                        isi_hist=cp.asnumpy(hist_d), spikes=spikes,
                        grid=grid, n_refined=n_refined)
        del par_d, drive_d, diag_d, hist_d, spk_d
        cp.get_default_memory_pool().free_all_blocks()
        return res

    # ------------------------------------------------------------------
    def spike_counts(self, params: np.ndarray, drive, grid: Grid) -> np.ndarray:
        """Spikes in `grid`'s window only -- the rheobase test (§4.2)."""
        res = self.run(params, drive, grid=grid, store_traces=False,
                       refine=False, max_spikes=0)
        return res.diag["n_spikes"]


# ------------------------------------------------------- host-side diagnostics


def host_diagnostics(diag: dict, isi_hist: np.ndarray, cfg: Config) -> dict:
    """The v1 §7.6 in-kernel diagnostics, retained as a cross-check (§3.1).

    The authoritative descriptors are computed from the exact spike times in
    `isi.descriptors`; these are the quantised, in-register versions.
    """
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

    log_lo, log_hi = math.log10(dg.isi_lo), math.log10(dg.isi_hi)
    edges = np.logspace(log_lo, log_hi, dg.n_isi_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])

    thresh = cfg.naming.burst_isi_frac * mean_isi            # 0.25 * mean ISI
    below = np.zeros_like(rate)
    above = np.zeros_like(rate)
    hist = isi_hist.astype(np.float64)
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
