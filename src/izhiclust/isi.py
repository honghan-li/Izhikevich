"""C1 — the ISI-sequence encoding (IMPROVEMENTS.md §3).

Everything here is host-side and works from the exact spike times captured at
`dt_int` resolution by `kernel.cu` (§3.1).  Four blocks per analysis window:

| block | content | dims |
|---|---|---|
| A | 1-D Gaussian KDE of `lambda_i = log10(I_i / Ibar)` on 32 points | 32 |
| B | 2-D Gaussian KDE of `(lambda_i, lambda_{i+1})` on 12x12 -- the return map | 144 |
| C | drift: `lambda_i` resampled onto 8 points of ordinal time `i/n` | 8 |
| D | `log10(Ibar)` -- rate | 1 |

Blocks A and B are non-negative densities and are L2-normalised *per sample*
exactly as §3.3 specifies.  Blocks C and D are signed and carry their
information in their *magnitude* (a tonic train has `C = 0`, `D` is the rate),
so they are divided by a single dataset-wide scale instead -- per-sample
normalisation of a one-element block would reduce `D` to its sign and would
amplify an arbitrarily small drift in `C` to full weight (assumption V2-A7).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, Isi
from .simulate import SpikeTrains

_EPS = 1e-12


# ------------------------------------------------------------------ ISI CSR


@dataclass
class IsiSet:
    """ISIs of every sample inside one analysis window, CSR-style."""
    isi: np.ndarray              # float64, concatenated
    offsets: np.ndarray          # int64, (n + 1,)
    n_spikes: np.ndarray         # int64, spikes inside the window
    t_lo: float
    t_hi: float

    @property
    def n(self) -> int:
        return self.offsets.size - 1

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def seg(self) -> np.ndarray:
        """Sample index of every ISI."""
        return np.repeat(np.arange(self.n, dtype=np.int64), self.counts)

    def get(self, i: int) -> np.ndarray:
        return self.isi[self.offsets[i]:self.offsets[i + 1]]


def isi_set(spikes: SpikeTrains, t_lo: float, t_hi: float) -> IsiSet:
    """ISIs between consecutive spikes that both lie in `[t_lo, t_hi)`."""
    n = spikes.n
    parts: List[np.ndarray] = []
    counts = np.zeros(n, dtype=np.int64)
    n_spk = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = spikes.get(i)
        t = t[(t >= t_lo) & (t < t_hi)]
        n_spk[i] = t.size
        d = np.diff(t.astype(np.float64)) if t.size > 1 else np.zeros(0)
        counts[i] = d.size
        parts.append(d)
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    isi = np.concatenate(parts) if parts else np.zeros(0)
    return IsiSet(isi.astype(np.float64), off, n_spk, t_lo, t_hi)


# --------------------------------------------------------------- descriptors


def descriptors(s: IsiSet, window_seconds: float, cfg: Config) -> Dict[str, np.ndarray]:
    """Firing descriptors from exact spike times (v1 §7.6 definitions).

    Replaces the quantised in-kernel versions for every downstream use
    (naming, §8.3 canonical consistency, §8.4 merge).
    """
    n = s.n
    cnt = s.counts.astype(np.float64)
    seg = s.seg
    out = {
        "rate": s.n_spikes / window_seconds,
        "n_spikes": s.n_spikes.astype(np.float64),
        "n_isi": cnt,
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = np.bincount(seg, weights=s.isi, minlength=n)
        mean_isi = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        sq = np.bincount(seg, weights=s.isi ** 2, minlength=n)
        var = np.where(cnt > 1, sq / np.maximum(cnt, 1) - mean_isi ** 2, np.nan)
        var = np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan)
        out["mean_isi"] = mean_isi
        out["cv_isi"] = np.sqrt(var) / mean_isi

        thr = cfg.naming.burst_isi_frac * mean_isi
        short = (s.isi < thr[seg]).astype(np.float64)
        below = np.bincount(seg, weights=short, minlength=n)
        above = cnt - below
        out["burst_index"] = np.where(cnt > 0, below / np.maximum(cnt, 1), 0.0)
        out["spikes_per_burst"] = np.where(
            above > 0, below / np.maximum(above, 1) + 1.0,
            np.where(below > 0, cnt + 1.0, 1.0))

        med = np.full(n, np.nan)
        first = np.full(n, np.nan)
        last = np.full(n, np.nan)
        for i in range(n):
            d = s.get(i)
            if d.size:
                med[i] = float(np.median(d))
                first[i] = d[0]
                last[i] = d[-1]
        out["median_isi"] = med
        out["first_isi"] = first
        out["last_isi"] = last
        out["adapt_index"] = np.where(
            np.isfinite(first) & np.isfinite(last) & (first + last > 0),
            (last - first) / np.maximum(first + last, _EPS), 0.0)
    return out


# ------------------------------------------------------------------- blocks


@dataclass
class Blocks:
    A: np.ndarray                # (n, n_a)
    B: np.ndarray                # (n, n_b^2)
    C: np.ndarray                # (n, n_c)
    D: np.ndarray                # (n, 1)
    n_isi: np.ndarray            # (n,)
    usable: np.ndarray           # (n,) bool: enough ISIs for A/B
    h: float

    @property
    def n(self) -> int:
        return self.A.shape[0]


def _kde_1d(lam: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    """(n_elem, n_grid) Gaussian kernel weights."""
    z = (lam[:, None] - grid[None, :]) / h
    return np.exp(-0.5 * z * z).astype(np.float32)


def compute_blocks(s: IsiSet, cfg: Config, h: Optional[float] = None,
                   elem_chunk: int = 2_000_000) -> Blocks:
    """Blocks A-D for one analysis window (§3.2)."""
    ic: Isi = cfg.isi
    h = ic.h if h is None else h
    n = s.n
    ga, gb = ic.grid_a(), ic.grid_b()
    nb = ic.n_b

    A = np.zeros((n, ic.n_a), dtype=np.float32)
    B = np.zeros((n, nb * nb), dtype=np.float32)
    C = np.zeros((n, ic.n_c), dtype=np.float32)
    D = np.zeros((n, 1), dtype=np.float32)

    cnt = s.counts
    tot = np.bincount(s.seg, weights=s.isi, minlength=n) if s.isi.size else np.zeros(n)
    mean_isi = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        D[:, 0] = np.where(cnt > 0, np.log10(np.maximum(mean_isi, _EPS)), 0.0)

    if s.isi.size == 0:
        usable = np.zeros(n, dtype=bool)
        return Blocks(A, B, C, D, cnt.copy(), usable, h)

    # lambda_i = log10(I_i / Ibar), clipped to the §3.2 grid range
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = np.log10(np.maximum(s.isi, _EPS) / np.maximum(mean_isi[s.seg], _EPS))
    lam = np.clip(np.nan_to_num(lam, nan=0.0), ic.clip_lo, ic.clip_hi)

    # ---- block A: 1-D KDE of {lambda_i}, chunked over ISIs
    seg = s.seg
    for lo in range(0, lam.size, elem_chunk):
        hi = min(lo + elem_chunk, lam.size)
        G = _kde_1d(lam[lo:hi], ga, h)
        sg = seg[lo:hi]
        for k in range(ic.n_a):
            A[:, k] += np.bincount(sg, weights=G[:, k],
                                   minlength=n).astype(np.float32)
    norm_a = np.maximum(cnt, 1)[:, None] * (h * np.sqrt(2.0 * np.pi))
    A /= norm_a.astype(np.float32)

    # ---- block B: 2-D KDE of the return map {(lambda_i, lambda_{i+1})}
    pair_counts = np.maximum(cnt - 1, 0)
    poff = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(pair_counts, out=poff[1:])
    n_pairs = int(poff[-1])
    if n_pairs:
        # indices of the first element of each consecutive pair
        base = np.repeat(s.offsets[:-1], pair_counts)
        within = np.arange(n_pairs, dtype=np.int64) - np.repeat(poff[:-1], pair_counts)
        i0 = base + within
        pseg = np.repeat(np.arange(n, dtype=np.int64), pair_counts)
        step = max(elem_chunk // 2, 1)
        for lo in range(0, n_pairs, step):
            hi = min(lo + step, n_pairs)
            G1 = _kde_1d(lam[i0[lo:hi]], gb, h)
            G2 = _kde_1d(lam[i0[lo:hi] + 1], gb, h)
            sg = pseg[lo:hi]
            for p in range(nb):
                w1 = G1[:, p]
                for q in range(nb):
                    B[:, p * nb + q] += np.bincount(
                        sg, weights=w1 * G2[:, q], minlength=n).astype(np.float32)
        normb = (np.maximum(pair_counts, 1)[:, None]
                 * (h * h * 2.0 * np.pi)).astype(np.float32)
        B /= normb

    # ---- block C: drift, lambda resampled onto n_c points of ordinal time.
    # Resampled as equal-count segment *means*, not by interpolation at 8
    # points: an alternating burst sequence aliases under point sampling (the
    # value at i/n depends on the parity of that index), and block C is meant
    # to carry the slow systematic trend (assumption V2-A9).
    within = np.arange(lam.size, dtype=np.int64) - np.repeat(s.offsets[:-1], cnt)
    frac = within / np.maximum(np.repeat(cnt, cnt) - 1, 1)
    binno = np.minimum((frac * ic.n_c).astype(np.int64), ic.n_c - 1)
    flat = np.repeat(np.arange(n, dtype=np.int64), cnt) * ic.n_c + binno
    sums = np.bincount(flat, weights=lam, minlength=n * ic.n_c)
    hits = np.bincount(flat, minlength=n * ic.n_c)
    with np.errstate(invalid="ignore"):
        Cc = np.where(hits > 0, sums / np.maximum(hits, 1), 0.0)
    C[:] = Cc.reshape(n, ic.n_c).astype(np.float32)
    C[cnt < 2] = 0.0

    usable = cnt >= 2
    return Blocks(A, B, C, D, cnt.copy(), usable, h)


# -------------------------------------------------- normalisation / weights


@dataclass
class Scales:
    """Dataset-wide constants needed to reproduce the feature matrix (§3.3)."""
    c_rms: float = 1.0
    d_mean: float = 0.0
    d_sd: float = 1.0
    extra_mean: Tuple[float, ...] = (0.0, 0.0)
    extra_sd: Tuple[float, ...] = (1.0, 1.0)
    c_rms_onset: float = 1.0
    d_mean_onset: float = 0.0
    d_sd_onset: float = 1.0

    def as_dict(self) -> Dict[str, object]:
        return {"c_rms": self.c_rms, "d_mean": self.d_mean, "d_sd": self.d_sd,
                "extra_mean": list(self.extra_mean),
                "extra_sd": list(self.extra_sd),
                "c_rms_onset": self.c_rms_onset,
                "d_mean_onset": self.d_mean_onset, "d_sd_onset": self.d_sd_onset}


def _l2_rows(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(nrm > 0, nrm, 1.0)


def _rms_norm(M: np.ndarray, mask: np.ndarray, floor: float = 0.0) -> float:
    """Dataset-wide RMS block norm, floored.

    The floor matters: on a dataset of perfectly regular trains the measured
    RMS of block C is pure roundoff, and dividing by it would amplify that
    roundoff to unit weight.  The floor is the smallest drift worth
    representing -- `c_scale_floor` decades per point (assumption V2-A9).
    """
    if not mask.any():
        return max(floor, 1.0)
    v = float(np.sqrt((np.linalg.norm(M[mask], axis=1) ** 2).mean()))
    return max(v, floor, _EPS)


def fit_scales(steady: Blocks, onset: Blocks, extras: np.ndarray,
               mask: np.ndarray, cfg: Optional[Config] = None) -> Scales:
    """Dataset-wide scales for the magnitude-carrying blocks, over `mask`."""
    d = steady.D[mask, 0]
    don = onset.D[onset.usable & mask, 0] if onset is not None else np.zeros(0)
    ex = extras[mask]
    ic = cfg.isi if cfg is not None else None
    c_floor = (ic.c_scale_floor * np.sqrt(ic.n_c)) if ic else 0.0
    d_floor = ic.d_scale_floor if ic else _EPS
    return Scales(
        c_rms=_rms_norm(steady.C, mask & steady.usable, c_floor),
        d_mean=float(d.mean()) if d.size else 0.0,
        d_sd=max(float(d.std()) if d.size else 0.0, d_floor),
        extra_mean=tuple(float(v) for v in np.nanmean(ex, axis=0)) if ex.size
        else (0.0, 0.0),
        extra_sd=tuple(max(float(v), d_floor)
                       for v in np.nanstd(ex, axis=0)) if ex.size else (1.0, 1.0),
        c_rms_onset=(_rms_norm(onset.C, mask & onset.usable, c_floor)
                     if onset is not None else max(c_floor, 1.0)),
        d_mean_onset=float(don.mean()) if don.size else 0.0,
        d_sd_onset=max(float(don.std()) if don.size else 0.0, d_floor),
    )


@dataclass
class Weights:
    w_a: float
    w_b: float
    w_c: float
    w_d: float
    w_onset: float
    h: float

    def tag(self) -> str:
        return (f"h={self.h:g} wA={self.w_a:g} wB={self.w_b:g} "
                f"wC={self.w_c:g} wD={self.w_d:g} wOn={self.w_onset:g}")

    @staticmethod
    def default(cfg: Config) -> "Weights":
        ic = cfg.isi
        return Weights(ic.w_a, ic.w_b, ic.w_c, ic.w_d, ic.w_onset, ic.h)


def steady_vector(b: Blocks, sc: Scales, w: Weights) -> np.ndarray:
    """`[w_A A_hat | w_B B_hat | w_C C_hat | w_D D_hat]`, unnormalised (§3.3)."""
    A = w.w_a * _l2_rows(b.A)
    B = w.w_b * _l2_rows(b.B)
    C = w.w_c * (b.C / sc.c_rms)
    D = w.w_d * ((b.D - sc.d_mean) / sc.d_sd)
    return np.hstack([A, B, C, D]).astype(np.float32)


def onset_vector(b: Blocks, extras: np.ndarray, sc: Scales,
                 w: Weights) -> np.ndarray:
    """The §6 onset block: blocks A-D on `[0, t_transient)` plus two scalars."""
    A = w.w_a * _l2_rows(b.A)
    B = w.w_b * _l2_rows(b.B)
    C = w.w_c * (b.C / sc.c_rms_onset)
    D = w.w_d * ((b.D - sc.d_mean_onset) / sc.d_sd_onset)
    ex = (np.nan_to_num(extras, nan=0.0, posinf=0.0, neginf=0.0)
          - np.asarray(sc.extra_mean)) / np.asarray(sc.extra_sd)
    return _l2_rows(np.hstack([A, B, C, D, ex]).astype(np.float32))


def assemble(steady: Blocks, onset: Blocks, extras: np.ndarray, sc: Scales,
             w: Weights) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(clustering matrix, features_isi 185-d, onset 187-d)."""
    raw = steady_vector(steady, sc, w)
    on = onset_vector(onset, extras, sc, w)
    F = _l2_rows(np.hstack([raw, w.w_onset * on]).astype(np.float32))
    return F, _l2_rows(raw), on


def onset_extras(onset_set: IsiSet, spikes: SpikeTrains, cfg: Config,
                 mean_isi_steady: np.ndarray) -> np.ndarray:
    """§6: spikes in the first 100 ms, and log10(first ISI / Ibar_steady)."""
    n = spikes.n
    out = np.zeros((n, 2), dtype=np.float64)
    t_cut = cfg.isi.onset_count_ms
    for i in range(n):
        t = spikes.get(i)
        out[i, 0] = float(np.count_nonzero(t < t_cut))
    d0 = np.array([onset_set.get(i)[0] if onset_set.counts[i] else np.nan
                   for i in range(n)])
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, 1] = np.log10(np.maximum(d0, _EPS)
                             / np.maximum(mean_isi_steady, _EPS))
    out[:, 1] = np.nan_to_num(out[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
    return out


# -------------------------------------------------- §3.4 classes / quarantine


@dataclass
class Classes:
    quiescent: np.ndarray        # Z: no spikes, sd(v) <= V_floor
    subthreshold: np.ndarray     # S: no spikes, sd(v) > V_floor
    sparse: np.ndarray           # W: spiking but n_ISI < 10
    nonfinite: np.ndarray        # X
    active: np.ndarray           # everything that enters the clustering

    def counts(self) -> Dict[str, int]:
        return {"Z": int(self.quiescent.sum()), "S": int(self.subthreshold.sum()),
                "W": int(self.sparse.sum()), "X": int(self.nonfinite.sum()),
                "clustered": int(self.active.sum())}


def classify(diag: Dict[str, np.ndarray], n_isi: np.ndarray, v_floor: float,
             cfg: Config) -> Classes:
    """§3.4: `Z`/`S`/`W`/`X`, gated on `sd(v)` and the ISI count."""
    bad = diag["finite"] < 0.5
    n_spikes = diag["n_spikes_all"]
    silent = (n_spikes <= 0) & ~bad
    quiescent = silent & (diag["sd_v"] <= v_floor)
    sub = silent & (diag["sd_v"] > v_floor)
    sparse = ~bad & ~silent & (n_isi < cfg.isi.n_isi_min)
    active = ~(bad | quiescent | sub | sparse)
    return Classes(quiescent, sub, sparse, bad, active)


# ------------------------------------------- §8.1 representation-health gates


def health_gates(F: np.ndarray, rate: np.ndarray, cv: np.ndarray,
                 bi: np.ndarray, cfg: Config, evr_curve: np.ndarray,
                 rng: Optional[np.random.Generator] = None) -> Dict:
    """The three §8.1 gates, measured exactly as §2.3 measured their v1 values.

    Returns a dict of measurements and pass flags; the pipeline treats a
    failure as blocking before clustering.
    """
    g = cfg.gates
    rng = rng or np.random.default_rng(0)
    k99 = int(np.searchsorted(evr_curve, cfg.clustering.pca_min_evr) + 1)
    res: Dict[str, object] = {
        "pca_components_99": k99,
        "pca_max_components": g.pca_max_components,
        "pca_pass": bool(k99 <= g.pca_max_components),
    }

    tonic = (cv < g.tonic_cv_max) & (bi < g.tonic_bi_max) & (rate > 0)
    idx = np.where(tonic)[0]
    if idx.size > g.tonic_subset:
        idx = rng.choice(idx, size=g.tonic_subset, replace=False)
    res["tonic_subset"] = int(idx.size)
    if idx.size < 50:
        res.update({"monotonicity_pass": None, "octave_pass": None,
                    "note": "too few pure-tonic samples to measure"})
        return res

    V = F[idx]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), _EPS)
    lf = np.log10(rate[idx])
    # pairwise cosine distance and |d log f0| on the subset
    Dm = 1.0 - V @ V.T
    Lm = np.abs(lf[:, None] - lf[None, :])
    iu = np.triu_indices(idx.size, 1)
    d = Dm[iu]
    l = Lm[iu]

    def mean_in(lo, hi):
        m = (l >= lo) & (l < hi)
        return (float(d[m].mean()) if m.any() else float("nan")), int(m.sum())

    near, n_near = mean_in(*g.near_window)
    far, n_far = mean_in(g.far_threshold, np.inf)
    ratio = near / far if far and np.isfinite(far) and far > 0 else float("nan")
    oct_lo, oct_hi = np.log10(2.0) - g.octave_tol, np.log10(2.0) + g.octave_tol
    h_lo, h_hi = np.log10(1.5) - g.octave_tol, np.log10(1.5) + g.octave_tol
    d_oct, n_oct = mean_in(oct_lo, oct_hi)
    d_15, n_15 = mean_in(h_lo, h_hi)
    res.update({
        "near_mean_distance": near, "near_pairs": n_near,
        "far_mean_distance": far, "far_pairs": n_far,
        "near_over_far": ratio,
        "monotonicity_ratio_max": g.monotonicity_ratio,
        "monotonicity_pass": bool(np.isfinite(ratio)
                                  and ratio <= g.monotonicity_ratio),
        "octave_mean_distance": d_oct, "octave_pairs": n_oct,
        "ratio15_mean_distance": d_15, "ratio15_pairs": n_15,
        "octave_pass": bool(np.isfinite(d_oct) and np.isfinite(d_15)
                            and d_oct >= d_15),
        "distance_by_separation": _distance_table(d, l),
    })
    res["passed"] = bool(res["pca_pass"] and res["monotonicity_pass"]
                         and res["octave_pass"])
    return res


_SEP_BINS = ((0.0, 0.002), (0.002, 0.01), (0.01, 0.02), (0.02, 0.05),
             (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.0),
             (1.0, 2.0), (2.0, 5.0))


def _distance_table(d: np.ndarray, l: np.ndarray) -> List[Dict[str, float]]:
    """The §2.3 table, recomputed on the v2 metric."""
    rows = []
    for lo, hi in _SEP_BINS:
        m = (l >= lo) & (l < hi)
        rows.append({"lo": lo, "hi": hi, "pairs": int(m.sum()),
                     "mean_cosine_distance": float(d[m].mean()) if m.any()
                     else float("nan")})
    return rows
