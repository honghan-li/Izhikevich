"""Three-term acquisition with greedy batch selection (SPEC.md §11.3).

score(x) = 0.40 * rank[ sum_k w_k * sd_k(x) ]          explore
          + 0.30 * rank[ sum_k w_k * ||grad mu_k(x)|| ] bifurcation surfaces
          + 0.30 * rank[ H(25-NN label posterior)(x) ]  type boundaries / voids

Batch selection is greedy with the posterior variance re-conditioned after
every pick.  The update is exact and y-free: conditioning on x_s subtracts
cov_j(x, x_s)^2 / var_j(x_s), and carrying the resulting u-vectors is exactly
a rank-1 Cholesky update of the training basis.  Because the per-GP basis is
O(n_train * n_candidates), the greedy stage runs on a shortlist of the
top-scoring candidates (see SHORTLIST_FACTOR) rather than the whole pool; the
variance *ranking* term is still computed over the full pool.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

SHORTLIST_FACTOR = 4
SHORTLIST_MIN = 4096


def _ranknorm_np(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    r = rankdata(v, method="average") - 1.0
    return (r / max(len(v) - 1, 1)).astype(np.float64)


def knn_entropy(x_pool: np.ndarray, x_lab: np.ndarray, labels: np.ndarray,
                k: int = 25) -> np.ndarray:
    """Predictive entropy of a k-NN classifier on the interim HDBSCAN labels.

    HDBSCAN noise (-1) is kept as its own class, so both type boundaries and
    unlabelled voids score highly.
    """
    from sklearn.neighbors import KNeighborsClassifier
    uniq = np.unique(labels)
    if uniq.size < 2:
        return np.zeros(x_pool.shape[0])
    k = int(min(k, x_lab.shape[0]))
    clf = KNeighborsClassifier(n_neighbors=k, weights="uniform", n_jobs=-1)
    clf.fit(x_lab, labels)
    p = clf.predict_proba(x_pool)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(p > 0, np.log(p), 0.0)
    return -(p * lp).sum(axis=1)


def pool_sd_and_grad(gps: Sequence, weights: np.ndarray, pool: np.ndarray,
                     chunk: int = 4096):
    """Weighted total posterior SD and weighted mean-gradient norm over a pool."""
    xp = gps[0].xp
    n = pool.shape[0]
    sd = np.zeros(n)
    gr = np.zeros(n)
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        block = pool[sl]
        for gp, w in zip(gps, weights):
            v = gp.var(block)
            v = xp.clip(v, 0.0, None)
            sd[sl] += w * _to_np(xp.sqrt(v))
            _, g = gp.mean_and_grad_norm(block)
            gr[sl] += w * _to_np(g)
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
    return sd, gr


def _to_np(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def select_batch(gps: Sequence, weights: np.ndarray, pool: np.ndarray,
                 x_lab: Optional[np.ndarray], labels: Optional[np.ndarray],
                 batch: int, cfg, rng: Optional[np.random.Generator] = None):
    """Return (indices into pool, diagnostics dict)."""
    sch = cfg.schedule
    xp = gps[0].xp
    n_pool = pool.shape[0]

    sd, gr = pool_sd_and_grad(gps, weights, pool)
    if x_lab is not None and labels is not None:
        ent = knn_entropy(pool, x_lab, labels, sch.knn_entropy)
    else:
        ent = np.zeros(n_pool)

    r_sd, r_gr, r_en = (_ranknorm_np(sd), _ranknorm_np(gr), _ranknorm_np(ent))
    score0 = sch.w_sd * r_sd + sch.w_grad * r_gr + sch.w_entropy * r_en

    m = int(min(n_pool, max(SHORTLIST_MIN, SHORTLIST_FACTOR * batch)))
    short = np.argsort(-score0)[:m]
    short.sort()
    Xs = pool[short]

    gr_s = xp.asarray(_ranknorm_np(gr[short]), dtype=xp.float32)
    en_s = xp.asarray(_ranknorm_np(ent[short]), dtype=xp.float32)

    A: List = []
    V: List = []
    for gp in gps:
        a = gp.basis(Xs)                       # (n_train, m)
        A.append(a)
        V.append(xp.clip(gp.signal_var - (a * a).sum(axis=0), 1e-12, None))
    # rank-1 update vectors, preallocated so the repulsion term is one GEMV
    U = [xp.zeros((batch, m), dtype=xp.float32) for _ in gps]
    w_d = xp.asarray(weights, dtype=xp.float32)

    picked = np.empty(batch, dtype=np.int64)
    taken = xp.zeros(m, dtype=bool)
    idx_all = xp.arange(m)
    for t in range(batch):
        sd_tot = xp.zeros(m, dtype=xp.float32)
        for k in range(len(gps)):
            sd_tot += w_d[k] * xp.sqrt(V[k])
        order = xp.argsort(sd_tot)
        r = xp.empty(m, dtype=xp.float32)
        r[order] = idx_all.astype(xp.float32) / max(m - 1, 1)
        score = sch.w_sd * r + sch.w_grad * gr_s + sch.w_entropy * en_s
        score = xp.where(taken, xp.asarray(-1e30, dtype=xp.float32), score)
        s = int(xp.argmax(score))
        picked[t] = short[s]
        taken[s] = True
        xs = Xs[s:s + 1]
        for k, gp in enumerate(gps):
            cov = gp.kernel_cross(Xs, xs).ravel() - A[k].T @ A[k][:, s]
            if t:
                cov -= U[k][:t].T @ U[k][:t, s]
            u = cov / xp.sqrt(V[k][s])
            V[k] = xp.clip(V[k] - u * u, 1e-12, None)
            U[k][t] = u
    del A, U, V
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()

    diag = {
        "mean_rank_sd": float(r_sd[picked].mean()),
        "mean_rank_grad": float(r_gr[picked].mean()),
        "mean_rank_entropy": float(r_en[picked].mean()),
        "shortlist": m,
        "pool_sd_max": float(sd.max()),
        "pool_entropy_max": float(ent.max()),
    }
    return picked, diag
