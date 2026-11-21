"""Validation: canonical placement (S1), stability ARI (S5), contiguity (S4).

SPEC.md §13.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CANONICAL, Config
from .cluster import build_tree, labels_from_tree, umap_embed


# ------------------------------------------------------- S1: canonical types


@dataclass
class Placement:
    label: str
    name: str
    params: Tuple[float, float, float, float]
    sample_id: int
    letter: str
    cluster_id: int
    probability: float
    predicted_letter: str
    predicted_prob: float
    quarantined: bool


def place_canonical(clusterer, E_active: np.ndarray, sample_ids: Sequence[int],
                    pos_in_active: np.ndarray, labels: np.ndarray,
                    probs: np.ndarray, letters: Dict[int, str],
                    quarantined: np.ndarray, params: np.ndarray) -> List[Placement]:
    """Direct in-sample label plus `approximate_predict` cross-check (§13.1).

    `sample_ids` index the full sample array; `pos_in_active` maps those to rows
    of the clustered (active) subset, or -1 for quarantined / non-finite ones.
    """
    import hdbscan
    sample_ids = [int(s) for s in sample_ids]
    rows = np.array([pos_in_active[s] for s in sample_ids], dtype=np.int64)
    good = rows >= 0
    pl = np.full(len(sample_ids), -1)
    pp = np.zeros(len(sample_ids))
    if good.any():
        pts = np.ascontiguousarray(E_active[rows[good]], dtype=np.float64)
        try:
            a_lab, a_prob = hdbscan.approximate_predict(clusterer, pts)
            pl[good] = a_lab
            pp[good] = a_prob
        except Exception:
            pass
    out: List[Placement] = []
    for k, (name, lab, a, b, c, d) in enumerate(CANONICAL):
        sid = sample_ids[k]
        j = int(rows[k])
        cid = int(labels[j]) if j >= 0 else -2
        if j >= 0:
            letter = letters.get(cid, "noise" if cid < 0 else "?")
            prob = float(probs[j])
        else:
            letter = "Z/X"
            prob = float("nan")
        out.append(Placement(
            label=lab, name=name, params=(a, b, c, d), sample_id=sid,
            letter=letter, cluster_id=cid, probability=prob,
            predicted_letter=letters.get(int(pl[k]), "noise" if pl[k] < 0 else "?"),
            predicted_prob=float(pp[k]),
            quarantined=bool(quarantined[k])))
    return out


# ---------------------------------------------------------- S5: stability


@dataclass
class Stability:
    mean_ari: float
    pairwise_ari: List[float]
    persistence: Dict[int, float]
    n_subsample: int
    n_fits: int
    seconds: float = 0.0


def stability(P: np.ndarray, labels: np.ndarray, cfg: Config, *,
              n_neighbors: int, min_cluster_size: int, min_samples: int,
              method: str, rng: np.random.Generator, log=print) -> Stability:
    """5 independent unseeded UMAP embeddings on a subsample (§13.2)."""
    import time
    from sklearn.metrics import adjusted_rand_score

    cl = cfg.clustering
    n = P.shape[0]
    m = int(min(cl.stability_subsample, n))
    idx = rng.choice(n, size=m, replace=False)
    idx.sort()
    Psub = np.ascontiguousarray(P[idx])
    ref = labels[idx]

    t0 = time.time()

    def one(_seed):
        E, _ = umap_embed(Psub, n_components=cl.umap_components,
                          n_neighbors=n_neighbors, min_dist=cl.umap_min_dist,
                          n_epochs=cl.umap_epochs, metric=cl.umap_metric,
                          random_state=None)
        slt, _mst = build_tree(E, min_samples, core_dist_n_jobs=4)
        lab, _ = labels_from_tree(E, slt, min_cluster_size, method)
        return lab

    with ThreadPoolExecutor(max_workers=cl.stability_fits) as ex:
        labs = list(ex.map(one, range(cl.stability_fits)))

    aris = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            aris.append(float(adjusted_rand_score(labs[i], labs[j])))

    # per-cluster persistence: fraction of embeddings where some cluster
    # reaches >= 0.5 Jaccard overlap with the reference cluster
    persistence: Dict[int, float] = {}
    for cid in np.unique(ref[ref >= 0]):
        mask = ref == cid
        hits = 0
        for lab in labs:
            best = 0.0
            for other in np.unique(lab[lab >= 0]):
                om = lab == other
                inter = np.count_nonzero(mask & om)
                if inter == 0:
                    continue
                union = np.count_nonzero(mask | om)
                best = max(best, inter / union)
            if best >= cl.persistence_jaccard:
                hits += 1
        persistence[int(cid)] = hits / len(labs)

    return Stability(mean_ari=float(np.mean(aris)) if aris else float("nan"),
                     pairwise_ari=aris, persistence=persistence,
                     n_subsample=m, n_fits=len(labs), seconds=time.time() - t0)


# --------------------------------------------------------- S4: contiguity


@dataclass
class Contiguity:
    n_components: int        # components holding >= 5% of the cluster (§13.3)
    sizes: List[int]
    means: List[List[float]]
    threshold: float
    multimodal: bool
    largest_fraction: float = 1.0
    n_components_total: int = 1


def global_knn_threshold(Xn: np.ndarray, k: int, pct: float) -> float:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, Xn.shape[0])).fit(Xn)
    d, _ = nn.kneighbors(Xn)
    return float(np.percentile(d[:, -1], pct))


def contiguity(Xn: np.ndarray, native: np.ndarray, labels: np.ndarray,
               cfg: Config, threshold: float,
               max_points: int = 8000,
               rng: Optional[np.random.Generator] = None) -> Dict[int, Contiguity]:
    """Single-linkage components of each cluster in normalised (a,b,c,d) (§13.3)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    rng = rng or np.random.default_rng(0)
    out: Dict[int, Contiguity] = {}
    for cid in np.unique(labels[labels >= 0]):
        idx = np.where(labels == cid)[0]
        sub = idx
        if idx.size > max_points:
            sub = rng.choice(idx, size=max_points, replace=False)
        pts = Xn[sub]
        if pts.shape[0] < 2:
            out[int(cid)] = Contiguity(1, [int(idx.size)],
                                       [native[idx].mean(axis=0).tolist()],
                                       threshold, False, 1.0, 1)
            continue
        Zl = linkage(pdist(pts), method="single")
        comp = fcluster(Zl, t=threshold, criterion="distance")
        labels_u, counts = np.unique(comp, return_counts=True)
        n_total = int(labels_u.size)
        frac_largest = float(counts.max() / sub.size)
        sizes, means = [], []
        frac = cfg.clustering.contiguity_min_frac
        for cv in labels_u:
            members = sub[comp == cv]
            if members.size / sub.size >= frac:
                sizes.append(int(members.size))
                means.append(native[members].mean(axis=0).tolist())
        order = np.argsort([-s for s in sizes])
        sizes = [sizes[i] for i in order]
        means = [means[i] for i in order]
        # §13.3 counts components holding >= 5% of the cluster.  A cluster that
        # shatters into many pieces none of which reaches 5% has a count of
        # zero -- that is *more* fragmented, not less, so it is flagged too.
        out[int(cid)] = Contiguity(len(sizes), sizes, means, threshold,
                                   len(sizes) != 1, frac_largest, n_total)
    return out
