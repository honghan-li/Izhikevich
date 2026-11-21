"""Per-cluster triangular matrices and design-unbiased variants (§14.1)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

COEFFS = ("a", "b", "c", "d")


def triangular_frame(P: np.ndarray, log_a: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Lower triangle = Pearson r, diagonal = SD (ddof=1), upper = NaN.

    `P` is (n, 4) in native units.  A `sd_log_a` column carries the SD of
    log10(a) as the extra column §14.1 asks for.
    """
    n = P.shape[0]
    M = np.full((4, 4), np.nan)
    if n >= 2:
        sd = P.std(axis=0, ddof=1)
        R = np.corrcoef(P, rowvar=False) if n >= 3 else np.full((4, 4), np.nan)
        for i in range(4):
            M[i, i] = sd[i]
            for j in range(i):
                M[i, j] = R[i, j]
    df = pd.DataFrame(M, index=list(COEFFS), columns=list(COEFFS))
    sd_log_a = (np.std(np.log10(P[:, 0]), ddof=1) if n >= 2 else np.nan)
    df["sd_log_a"] = [sd_log_a, np.nan, np.nan, np.nan]
    return df


def render_triangular(df: pd.DataFrame) -> str:
    return df.to_string(na_rep="", float_format=lambda v: f"{v: .4f}")


def corr_vector(P: np.ndarray) -> Dict[str, float]:
    out = {}
    if P.shape[0] >= 3:
        R = np.corrcoef(P, rowvar=False)
        for i in range(4):
            for j in range(i):
                out[f"r_{COEFFS[i]}{COEFFS[j]}"] = float(R[i, j])
    else:
        for i in range(4):
            for j in range(i):
                out[f"r_{COEFFS[i]}{COEFFS[j]}"] = float("nan")
    return out


def ipw_weights(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Inverse-propensity weights from a k-NN density estimate (--ipw-stats).

    Density in the normalised parameter box is estimated as k / (n * V_k) with
    V_k the volume of the k-NN ball; the weight is its reciprocal, normalised
    to mean 1.
    """
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    k = int(min(k, max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    r = np.maximum(d[:, -1], 1e-9)
    dens = k / (n * (np.pi ** 2 / 2.0) * r ** 4)      # 4-D ball volume factor
    w = 1.0 / np.maximum(dens, 1e-12)
    return w / w.mean()


def weighted_triangular(P: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """Weighted SD / Pearson r (for the optional IPW variant)."""
    n = P.shape[0]
    M = np.full((4, 4), np.nan)
    if n >= 2:
        w = w / w.sum()
        mu = (w[:, None] * P).sum(axis=0)
        Z = P - mu
        cov = (w[:, None] * Z).T @ Z / (1.0 - (w ** 2).sum())
        sd = np.sqrt(np.diag(cov))
        for i in range(4):
            M[i, i] = sd[i]
            for j in range(i):
                denom = sd[i] * sd[j]
                M[i, j] = cov[i, j] / denom if denom > 0 else np.nan
    df = pd.DataFrame(M, index=list(COEFFS), columns=list(COEFFS))
    df["sd_log_a"] = [np.nan] * 4
    return df


def compare_matrices(all_df: pd.DataFrame, lhs_df: pd.DataFrame,
                     thresh: float) -> Tuple[bool, float]:
    """(flag, max |delta r|) between the all-sample and LHS-only triangles."""
    a = all_df[list(COEFFS)].to_numpy()
    b = lhs_df[list(COEFFS)].to_numpy()
    mask = np.tril(np.ones((4, 4), dtype=bool), -1)
    d = np.abs(a[mask] - b[mask])
    d = d[np.isfinite(d)]
    if d.size == 0:
        return False, float("nan")
    return bool(d.max() > thresh), float(d.max())
