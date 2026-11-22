"""PCA -> UMAP -> HDBSCAN, the staged selection sweep and quarantine.

SPEC.md §10 (quarantine), §12 (dimension reduction, clustering, tree reuse).
"""
from __future__ import annotations

from dataclasses import dataclass
import string
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# `Z` (quiescent), `S` (subthreshold), `W` (sparse) and `X` (non-finite) name
# the held-out classes of §3.4, so the cluster alphabet skips all four.
_ALPHABET = [c for c in string.ascii_uppercase if c not in ("S", "W", "X", "Z")]


def letter_sequence(n: int) -> List[str]:
    out = list(_ALPHABET[:n])
    if n <= len(_ALPHABET):
        return out
    i = 0
    while len(out) < n:
        a = _ALPHABET[i // len(_ALPHABET)]
        b = _ALPHABET[i % len(_ALPHABET)]
        out.append(a + b)
        i += 1
    return out[:n]


def assign_letters(labels: np.ndarray) -> Dict[int, str]:
    """Cluster ids sorted by descending size -> A, B, C, ... (skipping X/Z)."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.lexsort((ids, -counts))
    letters = letter_sequence(len(ids))
    return {int(ids[j]): letters[k] for k, j in enumerate(order)}


# ------------------------------------------------------------- classes

# The `Z`/`S`/`W`/`X` classes of IMPROVEMENTS.md §3.4 live in `isi.classify`:
# they are gated on `sd(v)` and the ISI count, not on `ac_power`, because C9
# removes the batched rFFT that `ac_power` needed.


# ------------------------------------------------------------------- PCA


def fit_pca(F: np.ndarray, n_components: int, seed: int = 0):
    from sklearn.decomposition import PCA
    n_components = int(min(n_components, min(F.shape) - 1))
    pca = PCA(n_components=n_components, whiten=False, svd_solver="randomized",
              random_state=seed)
    scores = pca.fit_transform(F)
    return scores.astype(np.float32), pca


def fit_pca_adaptive(F: np.ndarray, n_min: int, min_evr: float, seed: int = 0):
    """PCA with `n_components` set to where the cumulative EVR reaches `min_evr`.

    §8.1 turns the v1 accommodation into a gate: the component count needed for
    99% of the variance must be <= 20, and that count is what the pipeline then
    uses.  The variance curve comes from a covariance eigendecomposition, which
    is the cheap way to get all of it at N ~ 10^5.
    """
    from sklearn.decomposition import PCA
    cap = int(min(F.shape[1], min(F.shape) - 1))
    full = PCA(n_components=cap, whiten=False, svd_solver="covariance_eigh",
               random_state=seed).fit(F)
    curve = np.cumsum(full.explained_variance_ratio_)
    k = int(np.searchsorted(curve, min_evr) + 1)
    k = int(min(cap, max(n_min, k)))
    scores = full.transform(F)[:, :k].astype(np.float32)
    pca = _PCASlice(full, k)
    return np.ascontiguousarray(scores), pca, curve


class _PCASlice:
    """The first `k` components of a fitted PCA, with a matching `transform`."""

    def __init__(self, pca, k: int):
        self._pca = pca
        self.k = int(k)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_[:k]
        self.n_components_ = int(k)
        self.components_ = pca.components_[:k]
        self.mean_ = pca.mean_

    def transform(self, X):
        return np.ascontiguousarray(
            self._pca.transform(X)[:, :self.k].astype(np.float32))


# ------------------------------------------------------------------ UMAP


def umap_embed(P: np.ndarray, *, n_components: int, n_neighbors: int,
               min_dist: float, n_epochs: int, metric: str = "euclidean",
               random_state: Optional[int] = None, verbose: bool = False):
    import umap
    kw = dict(n_neighbors=n_neighbors, min_dist=min_dist,
              n_components=n_components, metric=metric, n_epochs=n_epochs,
              verbose=verbose)
    # umap-learn 0.6 will switch default pipelines in a future release; pin the
    # 0.5.x layout the SPEC's parameter choices were calibrated against.
    try:
        import umap as _u
        import inspect as _i
        if "compatibility_layout" in _i.signature(_u.UMAP.__init__).parameters:
            kw["compatibility_layout"] = True
    except Exception:
        pass
    if random_state is None:
        kw["n_jobs"] = -1
    else:
        kw["random_state"] = random_state
    reducer = umap.UMAP(**kw)
    return np.ascontiguousarray(reducer.fit_transform(P), dtype=np.float32), reducer


# --------------------------------------------------------------- HDBSCAN


class _RVShim:
    """Minimal stand-in so the library's own DBCV code can score any labelling."""

    def __init__(self, labels, mst):
        self._relative_validity = None
        self.gen_min_span_tree = True
        self.labels_ = labels
        self._min_spanning_tree = mst


def relative_validity(mst: np.ndarray, labels: np.ndarray) -> float:
    import hdbscan
    if (labels >= 0).sum() == 0 or np.unique(labels[labels >= 0]).size < 2:
        return float("-inf")
    try:
        return float(hdbscan.HDBSCAN.relative_validity_.fget(_RVShim(labels, mst)))
    except Exception:
        return float("-inf")


# Penalty weight for the noise fraction above the §12.4 threshold.  Chosen so
# that an all-noise labelling can never win: relative_validity_ is bounded by
# 1, and 2.0 * (1 - 0.25) > 1.
NOISE_PENALTY_WEIGHT = 2.0


def score_labelling(mst: np.ndarray, labels: np.ndarray, cfg) -> Tuple[float, float, float, int]:
    """(score, relative_validity, noise_fraction, n_clusters), §12.4."""
    rv = relative_validity(mst, labels)
    noise = float((labels < 0).mean())
    n_cl = int(np.unique(labels[labels >= 0]).size)
    thr = cfg.clustering.noise_penalty_threshold
    penalty = NOISE_PENALTY_WEIGHT * max(0.0, noise - thr)
    return rv - penalty, rv, noise, n_cl


def build_tree(E: np.ndarray, min_samples: int, core_dist_n_jobs: int = -1,
               mcs_hint: int = 60):
    """One mutual-reachability MST / single-linkage tree build (§12.4.1).

    `mcs_hint` only affects the throw-away condense pass inside the library
    call (a small `min_cluster_size` makes that pass needlessly expensive); the
    returned single-linkage tree does not depend on it.
    """
    import hdbscan
    res = hdbscan.hdbscan(E, min_cluster_size=mcs_hint, min_samples=min_samples,
                          metric="euclidean", gen_min_span_tree=True,
                          core_dist_n_jobs=core_dist_n_jobs,
                          cluster_selection_method="leaf")
    slt, mst = res[4], res[5]
    return slt, mst


def labels_from_tree(E: np.ndarray, slt: np.ndarray, min_cluster_size: int,
                     method: str, allow_single_cluster: bool = False):
    """Condense + extract only -- the cheap half of the algorithm (§12.4.1)."""
    from hdbscan.hdbscan_ import _tree_to_labels
    out = _tree_to_labels(E, slt, min_cluster_size=min_cluster_size,
                          cluster_selection_method=method,
                          allow_single_cluster=allow_single_cluster)
    return out[0], out[1]


def full_fit(E: np.ndarray, *, min_cluster_size: int, min_samples: int,
             method: str, prediction_data: bool = True,
             core_dist_n_jobs: int = -1):
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                         min_samples=min_samples,
                         cluster_selection_method=method,
                         metric="euclidean", allow_single_cluster=False,
                         prediction_data=prediction_data,
                         gen_min_span_tree=True,
                         core_dist_n_jobs=core_dist_n_jobs)
    cl.fit(E)
    return cl


# ------------------------------------------------------------ the sweep


@dataclass
class SweepRow:
    """One row of the two-stage selection sweep (v1 §12.4, amended by §3.3).

    Stage 2 now sweeps the ISI representation parameters (KDE bandwidth and
    block weights) in place of the retired spectral tilt.
    """
    stage: int
    n_neighbors: int
    min_cluster_size: int
    min_samples: int
    method: str
    n_clusters: int
    noise_fraction: float
    relative_validity: float
    score: float
    seconds: float
    h: float = float("nan")
    w_a: float = float("nan")
    w_b: float = float("nan")
    w_c: float = float("nan")
    w_d: float = float("nan")
    w_onset: float = float("nan")
    pca_components: int = 0
    label: str = ""


def sweep_stage1(P: np.ndarray, cfg, log=print) -> Tuple[List[SweepRow], Dict]:
    """Clustering parameters at the default ISI weights, with tree reuse."""
    cl = cfg.clustering
    mss = cl.sweep_fast_min_samples if cfg.sweep_fast else cl.sweep_min_samples
    rows: List[SweepRow] = []
    embeddings: Dict[int, np.ndarray] = {}
    timings = {"umap": 0.0, "trees": 0.0, "extract": 0.0}

    for nn in cl.sweep_neighbors:
        t0 = time.time()
        E, _ = umap_embed(P, n_components=cl.umap_components, n_neighbors=nn,
                          min_dist=cl.umap_min_dist, n_epochs=cl.umap_epochs,
                          metric=cl.umap_metric, random_state=None)
        timings["umap"] += time.time() - t0
        embeddings[nn] = E
        for ms in mss:
            t0 = time.time()
            slt, mst = build_tree(E, ms)
            t_tree = time.time() - t0
            timings["trees"] += t_tree
            for mcs in cl.sweep_min_cluster_size:
                for method in cl.sweep_methods:
                    t1 = time.time()
                    if cfg.tree_reuse:
                        labels, _ = labels_from_tree(E, slt, mcs, method)
                        m = mst
                    else:
                        c = full_fit(E, min_cluster_size=mcs, min_samples=ms,
                                     method=method, prediction_data=False)
                        labels, m = c.labels_, c._min_spanning_tree
                    sc, rv, nf, ncl = score_labelling(m, labels, cfg)
                    dt = time.time() - t1
                    timings["extract"] += dt
                    rows.append(SweepRow(1, nn, mcs, ms, method, ncl, nf,
                                         rv, sc, dt, label="stage1"))
                    log(f"  [stage1] nn={nn:3d} mcs={mcs:4d} ms={ms:3d} "
                        f"{method:4s}: k={ncl:3d} noise={nf:.3f} rv={rv:.4f} "
                        f"score={sc:.4f} ({dt:.1f}s, tree {t_tree:.1f}s)")
            del slt, mst
    return rows, {"embeddings": embeddings, "timings": timings}


def pick_winner(rows: Sequence[SweepRow]) -> SweepRow:
    return max(rows, key=lambda r: (r.score, r.n_clusters))


def relative_validity_contributions(mst: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-cluster terms of `relative_validity_` (DBCV), in cluster-id order.

    Mirrors hdbscan's own `relative_validity_` so the terms sum to it; §14
    reports the per-cluster contribution.
    """
    labels = np.asarray(labels)
    sizes = np.bincount(labels + 1)
    noise_size = sizes[0]
    cluster_size = sizes[1:]
    total = noise_size + np.sum(cluster_size)
    k = len(cluster_size)
    if k == 0:
        return np.zeros(0)
    DSC = np.zeros(k)
    DSPC_wrt = np.ones(k) * np.inf
    min_outlier_sep = np.inf
    edge_from = mst.T[0].astype(np.intp)
    edge_to = mst.T[1].astype(np.intp)
    edge_dist = mst.T[2]
    l1, l2 = labels[edge_from], labels[edge_to]
    max_distance = edge_dist.max() if edge_dist.shape[0] > 0 else 0.0
    both_noise = (l1 == -1) & (l2 == -1)
    one_noise = (l1 == -1) ^ (l2 == -1)
    neither = ~both_noise & ~one_noise
    if one_noise.any():
        min_outlier_sep = edge_dist[one_noise].min()
    same = neither & (l1 == l2)
    diff = neither & (l1 != l2)
    if same.any():
        np.maximum.at(DSC, l1[same], edge_dist[same])
    if diff.any():
        np.minimum.at(DSPC_wrt, l1[diff], edge_dist[diff])
        np.minimum.at(DSPC_wrt, l2[diff], edge_dist[diff])
    min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep
    correction = 2 * (max_distance if k > 1 else min_outlier_sep)
    DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction
    denom = np.maximum(DSPC_wrt, DSC)
    with np.errstate(invalid="ignore", divide="ignore"):
        V = (DSPC_wrt - DSC) / denom
    return cluster_size * V / total


def containment_table(leaf: np.ndarray, eom: np.ndarray) -> List[Dict]:
    """leaf -> eom containment (§12.4: report both taxonomies)."""
    rows = []
    for cid in np.unique(leaf[leaf >= 0]):
        m = leaf == cid
        vals, counts = np.unique(eom[m], return_counts=True)
        j = int(np.argmax(counts))
        rows.append({"leaf": int(cid), "size": int(m.sum()),
                     "eom": int(vals[j]),
                     "containment": float(counts[j] / m.sum())})
    return rows
