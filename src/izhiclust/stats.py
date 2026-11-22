"""Per-cluster triangular matrices (v1 §14.1).

C6 retires the GP, so every sample is design-unbiased: the v1 "all" vs
"LHS-only (authoritative)" split and the inverse-propensity machinery are gone
(§7).  The design vector now carries the overdrive, so the matrices are 5x5
over `(a, b, c, d, Delta)`.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

COEFFS = ("a", "b", "c", "d", "delta")


def _names(k: int):
    return list(COEFFS[:k])


def triangular_frame(P: np.ndarray) -> pd.DataFrame:
    """Lower triangle = Pearson r, diagonal = SD (ddof=1), upper = empty.

    A `sd_log_a` column carries the SD of `log10(a)`, as v1 §14.1 asks.
    """
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    n, k = P.shape
    M = np.full((k, k), np.nan)
    if n >= 2:
        sd = P.std(axis=0, ddof=1)
        R = np.corrcoef(P, rowvar=False) if n >= 3 else np.full((k, k), np.nan)
        for i in range(k):
            M[i, i] = sd[i]
            for j in range(i):
                M[i, j] = R[i, j]
    df = pd.DataFrame(M, index=_names(k), columns=_names(k))
    sd_log_a = (np.std(np.log10(np.maximum(P[:, 0], 1e-30)), ddof=1)
                if n >= 2 else np.nan)
    df["sd_log_a"] = [sd_log_a] + [np.nan] * (k - 1)
    return df


def render_triangular(df: pd.DataFrame) -> str:
    return df.to_string(na_rep="", float_format=lambda v: f"{v: .4f}")


def corr_vector(P: np.ndarray) -> Dict[str, float]:
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    k = P.shape[1]
    out: Dict[str, float] = {}
    R = np.corrcoef(P, rowvar=False) if P.shape[0] >= 3 else np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i):
            out[f"r_{COEFFS[i]}{COEFFS[j]}"] = float(R[i, j])
    return out
