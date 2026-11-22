"""Host-side, on-demand spectra (IMPROVEMENTS.md §3.5, C9).

The rFFT is no longer a pipeline stage: nothing batched consumes a spectrum,
so the anti-aliased output stage, the `|H(f)|^2` deconvolution, the hybrid
banding and `spectra_full.csv` are all gone (§3.5).  What remains is a plotting
and characterisation tool applied to ~100 samples — the cluster
representatives and the `S`-class subthreshold resonators, re-simulated on a
10x finer output grid with no decimation filter so that aliasing is negligible
by construction rather than by filtering.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def rfft_magnitude(traces: np.ndarray, dt_out_ms: float,
                   detrend: bool = True, window: bool = True,
                   amplitude: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """(freqs Hz, |rFFT| magnitude) of each row of `traces`.

    Mean-subtracted and Hann-windowed, as v1 §9 steps 1-4; no deconvolution
    (there is no filter left to deconvolve) and no banding.  With
    `amplitude=True` the magnitude is scaled by `4 / n` -- two for the
    one-sided spectrum, two for the Hann coherent gain -- so a pure tone of
    amplitude `A` reads `A` mV instead of growing with the record length.
    """
    x = np.atleast_2d(np.asarray(traces, dtype=np.float32))
    if detrend:
        x = x - x.mean(axis=1, keepdims=True)
    if window:
        x = x * np.hanning(x.shape[1]).astype(np.float32)
    mag = np.abs(np.fft.rfft(x, axis=1)).astype(np.float32)
    if amplitude:
        mag *= np.float32(4.0 / x.shape[1]) if window else np.float32(2.0 / x.shape[1])
    freqs = np.fft.rfftfreq(x.shape[1], d=dt_out_ms * 1e-3)
    return freqs, mag


def peak_frequency(freqs: np.ndarray, mag: np.ndarray, f_lo: float = 0.5,
                   f_hi: float = 1000.0) -> np.ndarray:
    """Dominant in-band frequency of each row -- the `S`-class descriptor (§3.4)."""
    mag = np.atleast_2d(mag)
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    if not sel.any():
        return np.full(mag.shape[0], np.nan)
    j = np.argmax(mag[:, sel], axis=1)
    return freqs[sel][j]


def rms_amplitude(traces: np.ndarray) -> np.ndarray:
    """AC RMS of each row, in mV -- the `sd(v)` of §3.4, host-side."""
    x = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    return np.sqrt(((x - x.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
