"""Validation: canonical placement (§8.3), stability (§8.2), contiguity (v1 §13.3)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CANONICAL, CANONICAL_EXEMPT, Config
from .cluster import build_tree, labels_from_tree, umap_embed

# §8.3: the descriptors a canonical type's host cluster must be consistent with
CANON_DESCRIPTORS = ("cv_isi", "burst_index", "spikes_per_burst", "rate")


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
    held_out: bool
    held_out_class: str
    drive: float = float("nan")
    exempt: bool = False
    consistent: Optional[bool] = None
    inlier: Optional[bool] = None
    descriptors: Dict[str, Dict[str, float]] = field(default_factory=dict)


def place_canonical(clusterer, E_active: np.ndarray, sample_ids: Sequence[int],
                    pos_in_active: np.ndarray, labels: np.ndarray,
                    probs: np.ndarray, letters: Dict[int, str],
                    held_out: np.ndarray, held_class: Sequence[str],
                    params: np.ndarray, drive: np.ndarray,
                    desc: Dict[str, np.ndarray],
                    active_idx: np.ndarray) -> List[Placement]:
    """In-sample label, `approximate_predict` cross-check and the §8.3 test.

    §8.3 replaces v1's unfailable "lands in a non-noise cluster" with a
    descriptor-consistency test: each of the type's own measured
    `(CV, burst index, spikes/burst, rate)` must lie inside the host cluster's
    interquartile range.  `TC` and `RZ` are exempt by name, in the spec.
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
            letter = held_class[k]
            prob = float("nan")
        dd: Dict[str, Dict[str, float]] = {}
        consistent: Optional[bool] = None
        if j >= 0 and cid >= 0:
            members = active_idx[labels == cid]
            consistent = True
            inlier = True
            for key in CANON_DESCRIPTORS:
                v = float(desc[key][sid])
                col = desc[key][members]
                col = col[np.isfinite(col)]
                q1 = float(np.percentile(col, 25)) if col.size else float("nan")
                q3 = float(np.percentile(col, 75)) if col.size else float("nan")
                inside = bool(np.isfinite(v) and q1 <= v <= q3)
                # Tukey inlier range, reported alongside: the §8.3 criterion is
                # the IQR itself, which a *typical* member of a tight cluster
                # also fails, so the whisker version separates "gross mismatch"
                # (what §8.3 exists to catch) from ordinary spread.
                w = 1.5 * (q3 - q1)
                within = bool(np.isfinite(v) and q1 - w <= v <= q3 + w)
                dd[key] = {"value": v, "q25": q1, "q75": q3,
                           "inside": inside, "within_whiskers": within}
                consistent = consistent and inside
                inlier = inlier and within
            dd["_inlier"] = {"value": float(inlier)}
        out.append(Placement(
            label=lab, name=name, params=(a, b, c, d), sample_id=sid,
            letter=letter, cluster_id=cid, probability=prob,
            predicted_letter=letters.get(int(pl[k]), "noise" if pl[k] < 0 else "?"),
            predicted_prob=float(pp[k]),
            held_out=bool(held_out[k]), held_out_class=str(held_class[k]),
            drive=float(drive[sid]), exempt=lab in CANONICAL_EXEMPT,
            consistent=consistent,
            inlier=(bool(dd.pop("_inlier")["value"]) if "_inlier" in dd
                    else None),
            descriptors=dd))
    return out


# ---------------------------------------------------------- S5: stability


@dataclass
class StabilityRun:
    name: str
    description: str
    mean_ari: float
    pairwise_ari: List[float]
    threshold: float
    n_subsample: int
    n_fits: int
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.mean_ari) and self.mean_ari >= self.threshold)


@dataclass
class Stability:
    runs: Dict[str, StabilityRun]
    persistence: Dict[int, float]

    @property
    def seed(self) -> StabilityRun:
        return self.runs["seed"]

    def seconds(self) -> float:
        return sum(r.seconds for r in self.runs.values())


def scaled_mcs(min_cluster_size: int, n_fit: int, n_full: int) -> int:
    """`min_cluster_size` at constant *fraction* of the sample set.

    v1 §12.4 sets `min_cluster_size = 60` as "approximately 0.22% of N", so a
    fit on a subsample has to scale it or it resolves a coarser taxonomy than
    the reference and every per-cluster Jaccard comes out low for reasons that
    have nothing to do with stability (assumption V2-A31).
    """
    if n_fit >= n_full:
        return int(min_cluster_size)
    return int(max(5, round(min_cluster_size * n_fit / max(n_full, 1))))


def _label_once(P: np.ndarray, cfg: Config, *, n_neighbors: int,
                min_cluster_size: int, min_samples: int, method: str,
                random_state: Optional[int], n_full: Optional[int] = None
                ) -> np.ndarray:
    cl = cfg.clustering
    mcs = scaled_mcs(min_cluster_size, P.shape[0], n_full or P.shape[0])
    E, _ = umap_embed(P, n_components=cl.umap_components,
                      n_neighbors=n_neighbors, min_dist=cl.umap_min_dist,
                      n_epochs=cl.umap_epochs, metric=cl.umap_metric,
                      random_state=random_state)
    slt, _mst = build_tree(E, min_samples, core_dist_n_jobs=4, mcs_hint=mcs)
    lab, _ = labels_from_tree(E, slt, mcs, method)
    return lab


def stability_tests(P: np.ndarray, labels: np.ndarray, cfg: Config, *,
                    n_neighbors: int, min_cluster_size: int, min_samples: int,
                    method: str, rng: np.random.Generator,
                    P_window: Optional[np.ndarray] = None,
                    P_dt: Optional[np.ndarray] = None,
                    dt_rows: Optional[np.ndarray] = None,
                    log=print) -> Stability:
    """The four §8.2 tests.

    - `seed`: unseeded UMAP refits of the same features (v1 §13.2, retained).
    - `bootstrap`: 80% subsamples, 5 draws, ARI on each pair's intersection.
    - `window shift`: features from the `[600, 2600)` ms window against the
      `[500, 2500)` ms ones, **at matched UMAP seeds** so the comparison
      measures the perturbation and not seed noise (assumption V2-A10).
    - `step size`: the same, for features from a `dt_int/2` re-simulation.
    """
    import time
    from sklearn.metrics import adjusted_rand_score

    cl = cfg.clustering
    n = P.shape[0]
    kw = dict(n_neighbors=n_neighbors, min_cluster_size=min_cluster_size,
              min_samples=min_samples, method=method, n_full=n)
    m = int(min(cl.stability_subsample, n))
    idx = np.sort(rng.choice(n, size=m, replace=False))
    Psub = np.ascontiguousarray(P[idx])
    ref = labels[idx]
    runs: Dict[str, StabilityRun] = {}

    # ---- seed
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cl.stability_fits) as ex:
        labs = list(ex.map(
            lambda s: _label_once(Psub, cfg, random_state=None, **kw),
            range(cl.stability_fits)))
    aris = [float(adjusted_rand_score(labs[i], labs[j]))
            for i in range(len(labs)) for j in range(i + 1, len(labs))]
    runs["seed"] = StabilityRun(
        "seed", f"UMAP random_state, features and sample set fixed "
        f"(v1 §13.2); min_cluster_size "
        f"{scaled_mcs(min_cluster_size, m, n)} at subsample scale",
        float(np.mean(aris)) if aris else float("nan"), aris, cl.ari_seed, m,
        len(labs), time.time() - t0)
    log(f"  stability[seed]: mean ARI {runs['seed'].mean_ari:.3f} "
        f"({runs['seed'].seconds:.0f}s)")

    # ---- per-cluster persistence, from the seed fits (v1 §13.2)
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
                best = max(best, inter / np.count_nonzero(mask | om))
            if best >= cl.persistence_jaccard:
                hits += 1
        persistence[int(cid)] = hits / max(len(labs), 1)

    # ---- bootstrap
    t0 = time.time()
    k = int(round(cl.bootstrap_fraction * m))
    draws = [np.sort(rng.choice(m, size=k, replace=False))
             for _ in range(cl.stability_fits)]
    with ThreadPoolExecutor(max_workers=cl.stability_fits) as ex:
        blabs = list(ex.map(
            lambda dr: _label_once(np.ascontiguousarray(Psub[dr]), cfg,
                                   random_state=cl.random_state, **kw), draws))
    baris: List[float] = []
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            common, ia, ib = np.intersect1d(draws[i], draws[j],
                                            return_indices=True)
            if common.size > 10:
                baris.append(float(adjusted_rand_score(blabs[i][ia],
                                                       blabs[j][ib])))
    runs["bootstrap"] = StabilityRun(
        "bootstrap", f"{cl.bootstrap_fraction:.0%} subsamples, "
        f"{cl.stability_fits} draws, ARI on each pair's intersection",
        float(np.mean(baris)) if baris else float("nan"), baris,
        cl.ari_bootstrap, k, len(draws), time.time() - t0)
    log(f"  stability[bootstrap]: mean ARI {runs['bootstrap'].mean_ari:.3f} "
        f"({runs['bootstrap'].seconds:.0f}s)")

    # ---- feature-perturbation tests, at matched seeds
    seeds = (cl.random_state, cl.random_state + 1, cl.random_state + 2)
    for key, Palt, thr, desc in (
            ("window", P_window, cl.ari_window,
             "analysis window shifted by +100 ms"),
            ("dt", P_dt, cl.ari_dt, "dt_int halved")):
        if Palt is None:
            continue
        t0 = time.time()
        if key == "dt" and dt_rows is not None:
            keep = np.isin(idx, dt_rows)
            sub = idx[keep]
            pos = np.searchsorted(dt_rows, sub)
            Pa = np.ascontiguousarray(P[sub])
            Pb = np.ascontiguousarray(Palt[pos])
        else:
            Pa, Pb = Psub, np.ascontiguousarray(Palt[idx])
        if Pa.shape[0] < 50:
            continue
        with ThreadPoolExecutor(max_workers=min(6, 2 * len(seeds))) as ex:
            res = list(ex.map(
                lambda a: _label_once(a[0], cfg, random_state=a[1], **kw),
                [(Pa, s) for s in seeds] + [(Pb, s) for s in seeds]))
        pa, pb = res[:len(seeds)], res[len(seeds):]
        ar = [float(adjusted_rand_score(x, y)) for x, y in zip(pa, pb)]
        runs[key] = StabilityRun(key, desc,
                                 float(np.mean(ar)) if ar else float("nan"),
                                 ar, thr, Pa.shape[0], 2 * len(seeds),
                                 time.time() - t0)
        log(f"  stability[{key}]: mean ARI {runs[key].mean_ari:.3f} "
            f"({runs[key].seconds:.0f}s, {desc})")

    return Stability(runs=runs, persistence=persistence)


# --------------------------------------------------------- S4: contiguity


@dataclass
class Contiguity:
    n_components: int
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
    """Single-linkage components of each cluster in the normalised design box.

    The box is now 5-D -- `(a, b, c, d, Delta)` -- because `Delta` is a design
    axis (§4.2), so contiguity is judged in the space actually sampled.
    """
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
        out[int(cid)] = Contiguity(len(sizes), sizes, means, threshold,
                                   len(sizes) != 1, frac_largest, n_total)
    return out
