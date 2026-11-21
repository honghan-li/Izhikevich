"""Feature pipeline (SPEC.md §9) and the ac_power definition (§10).

Works on either numpy or cupy arrays: the array module is taken from the
input, so the sweep runs on the GPU and the tests/round-trip checks on the
host with identical code.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from .config import Config, Features, Grid


def _xp(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp
    except Exception:
        pass
    return np


# --------------------------------------------------------------- band plan


@dataclass
class BandPlan:
    freqs: np.ndarray        # (n_freq,) Hz, one-sided
    h2_inv: np.ndarray       # (n_freq,) 1/|H(f)|^2, the §7.2 deconvolution
    tilt: np.ndarray         # (n_freq,) (f_ref/f)^alpha, 1 at f=0
    bl_hi: int               # last band-limited bin index (inclusive)
    band_edges: np.ndarray   # (n_bands+1,) bin indices, [start, stop)
    band_matrix: np.ndarray  # (bl_hi, n_bands) averaging matrix
    band_centres: np.ndarray  # (n_bands,) Hz
    alpha: float
    n_out: int
    dt_out: float
    noise_floor_frac: float = 0.0

    @property
    def n_bands(self) -> int:
        return self.band_centres.size


def _boxcar_response(f_hz: np.ndarray, L: int, dt_ms: float, stages: int) -> np.ndarray:
    """|H(f)| for `stages` cascaded length-L boxcars at step dt (§7.2)."""
    T = dt_ms * 1e-3                      # s
    x = math.pi * f_hz * T
    num = np.sin(L * x)
    den = L * np.sin(x)
    h = np.ones_like(f_hz)
    nz = np.abs(den) > 0
    h[nz] = num[nz] / den[nz]
    h[f_hz == 0.0] = 1.0
    return np.abs(h) ** stages


def make_plan(cfg: Config, grid: Optional[Grid] = None,
              alpha: Optional[float] = None,
              deconvolve: bool = True) -> BandPlan:
    grid = grid or cfg.grid
    ft: Features = cfg.features
    alpha = cfg.features.alpha if alpha is None else alpha

    n = grid.n_out
    dt_s = grid.dt_out * 1e-3
    freqs = np.fft.rfftfreq(n, d=dt_s)
    df = freqs[1] - freqs[0]

    h2 = _boxcar_response(freqs, grid.L, grid.dt_int, grid.boxcar_stages)
    h2_inv = (1.0 / h2) if deconvolve else np.ones_like(h2)

    tilt = np.ones_like(freqs)
    if alpha:
        with np.errstate(divide="ignore"):
            tilt[1:] = (ft.f_ref / freqs[1:]) ** alpha

    # band-limit to (0, f_band_hi]
    bl_hi = int(round(ft.f_band_hi / df))          # inclusive bin index
    k_native_hi = int(round(ft.f_native_hi / df))  # inclusive, = 40 at df=0.5

    if k_native_hi < ft.n_native:
        raise ValueError("native band count exceeds available bins")
    # 40 native bins of width df, then n_log log-spaced bands to f_band_hi
    native_edges = np.arange(1, ft.n_native + 1, dtype=np.int64)
    start = int(native_edges[-1]) + 1
    r = (ft.f_band_hi / ft.f_native_hi) ** (1.0 / ft.n_log)
    log_edges = [start]
    for j in range(1, ft.n_log):
        e = int(math.ceil(ft.f_native_hi * (r ** j) / df))
        log_edges.append(max(log_edges[-1] + 1, e))
    edges = np.concatenate([native_edges, np.asarray(log_edges, dtype=np.int64),
                            np.asarray([bl_hi + 1], dtype=np.int64)])
    if np.any(np.diff(edges) < 1):
        raise ValueError("empty band in the hybrid band plan")

    n_bands = ft.n_bands
    assert edges.size == n_bands + 1, (edges.size, n_bands)

    # averaging matrix over band-limited bins 1 .. bl_hi
    W = np.zeros((bl_hi, n_bands), dtype=np.float32)
    centres = np.empty(n_bands, dtype=np.float64)
    for j in range(n_bands):
        lo, hi = int(edges[j]), int(edges[j + 1])
        W[lo - 1:hi - 1, j] = 1.0 / (hi - lo)
        centres[j] = float(np.sqrt(freqs[lo] * freqs[hi - 1])) if lo != hi - 1 \
            else float(freqs[lo])

    return BandPlan(freqs=freqs, h2_inv=h2_inv, tilt=tilt, bl_hi=bl_hi,
                    band_edges=edges, band_matrix=W, band_centres=centres,
                    alpha=alpha, n_out=n, dt_out=grid.dt_out,
                    noise_floor_frac=ft.noise_floor_frac)


# ------------------------------------------------------------- the pipeline


def spectra(traces, plan: BandPlan):
    """Steps 1-5 of §9: mean-subtract, Hann, rfft, magnitude, deconvolve.

    Returns the deconvolved one-sided magnitude spectrum (n, n_freq).
    """
    xp = _xp(traces)
    x = traces.astype(xp.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    w = xp.asarray(np.hanning(plan.n_out).astype(np.float32))
    x = x * w
    X = xp.fft.rfft(x, axis=1)
    mag = xp.abs(X).astype(xp.float32)
    mag *= xp.asarray(plan.h2_inv.astype(np.float32))
    return mag


def ac_power(mag, plan: BandPlan):
    """§10: sum |X(f)|^2 over the band-limited bins f in (0, 1000] Hz."""
    xp = _xp(mag)
    seg = mag[:, 1:plan.bl_hi + 1].astype(xp.float64)
    return (seg * seg).sum(axis=1)


def band_features(mag, plan: BandPlan):
    """Steps 6-10 of §9: tilt, band-limit, hybrid banding, floor, log1p, L2.

    The relative-noise-floor subtraction between banding and `log1p` is
    amendment A1 (SPEC §19): `log1p` on raw magnitudes ties its compression
    point to absolute loudness, which let the leakage/jitter floor of the
    near-empty bands occupy a large share of the unit feature vector.
    """
    xp = _xp(mag)
    seg = mag[:, 1:plan.bl_hi + 1]
    if plan.alpha:
        seg = seg * xp.asarray(plan.tilt[1:plan.bl_hi + 1].astype(np.float32))
    p = seg.astype(xp.float32)
    p = p * p
    W = xp.asarray(plan.band_matrix)
    banded = xp.sqrt(p @ W)                  # RMS magnitude per band
    rho = plan.noise_floor_frac
    if rho:
        peak = banded.max(axis=1, keepdims=True)
        banded = xp.maximum(banded - rho * peak, 0.0)
    F = xp.log1p(banded)
    nrm = xp.linalg.norm(F, axis=1, keepdims=True)
    nrm = xp.where(nrm > 0, nrm, 1.0)
    return (F / nrm).astype(xp.float32)


def full_pipeline(traces, plan: BandPlan) -> Tuple:
    """(features, ac_power, magnitude) -- magnitude is the §16.2 spectra_full."""
    mag = spectra(traces, plan)
    return band_features(mag, plan), ac_power(mag, plan), mag
