"""Host reference for the §7.2 output stage, matching kernel.cu exactly.

The kernel accumulates, per block of L integration samples,
    A_k = sum_{m=1..L} (L + 1 - m) x_{kL+m}
    B_k = sum_{m=1..L} (m - 1)     x_{kL+m}
and emits (A_k + B_{k-1}) / L^2, which is precisely two cascaded length-L
boxcars followed by decimation by L (triangular 2L-1 tap FIR).
"""
import numpy as np


def cascaded_boxcar_decimate(x: np.ndarray, L: int = 10) -> np.ndarray:
    n_blocks = x.size // L
    w_desc = np.arange(L, 0, -1, dtype=np.float64)        # L+1-m, m=1..L
    w_asc = np.arange(0, L, dtype=np.float64)             # m-1
    X = x[:n_blocks * L].reshape(n_blocks, L).astype(np.float64)
    A = X @ w_desc
    B = X @ w_asc
    Bprev = np.concatenate([[0.0], B[:-1]])
    return (A + Bprev) / (L * L)


def triangular_fir(L: int = 10) -> np.ndarray:
    b = np.ones(L) / L
    return np.convolve(b, b)


def filter_only(x: np.ndarray, L: int = 10) -> np.ndarray:
    """Same filter, no decimation (for measuring the response)."""
    return np.convolve(x, triangular_fir(L), mode="full")[:x.size]
