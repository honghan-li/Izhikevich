"""Interpretable per-cluster names (v1 §14.2) and the §8.4 nameability merge."""
from __future__ import annotations

import itertools
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Config

# the descriptors §8.4 arbitrates distinguishability on
MERGE_DESCRIPTORS = ("rate", "cv_isi", "burst_index", "spikes_per_burst",
                     "adapt_index")


def _rate_word(rate: float, nm) -> str:
    if rate < nm.rate_slow:
        return "slow"
    if rate < nm.rate_fast:
        return "medium"
    return "fast"


def _burst_word(spb: float) -> str:
    """Qualitative burst size, readable straight off the return map (§3.2 B)."""
    if spb < 2.5:
        return "doublet"
    if spb < 3.5:
        return "triplet"
    return "multi-spike"


def name_from_medians(med: Dict[str, float], cfg: Config) -> str:
    """Name one cluster from its diagnostic medians.

    Every clustered sample fires and has at least `n_isi_min` ISIs (§3.4), so
    the v1 "silent" / "subthreshold" / "below-rate-floor" cores are gone: those
    populations are now the separately reported `Z`, `S` and `W` classes.
    """
    nm = cfg.naming
    rate = med.get("rate", 0.0)
    bi = med.get("burst_index", 0.0)
    cv = med.get("cv_isi", 0.0)
    ad = med.get("adapt_index", 0.0)
    spb = med.get("spikes_per_burst", 1.0)

    if bi > nm.burst_index:
        ib_rate = rate / max(spb, 1.0)
        core = (f"{_rate_word(rate, nm)} {_burst_word(spb)} bursting "
                f"({spb:.1f} spikes/burst, {ib_rate:.0f} Hz interburst)")
    else:
        core = f"{_rate_word(rate, nm)} tonic"

    # v1 §19 A11: a cluster that mixes bursting and tonic members at the same
    # fundamental is named from its medians and so hides the mixture.
    fb = med.get("frac_bursting", None)
    if fb is not None and 0.2 <= fb <= 0.8:
        core += f" [mixed: {fb:.0%} bursting]"

    tags = []
    if ad > nm.adapt:
        tags.append("+adapting")
    elif ad < -nm.adapt:
        tags.append("+accelerating")
    if cv is not None and np.isfinite(cv) and cv > nm.cv_irregular \
            and bi <= nm.burst_index:
        tags.append("+irregular")
    return ", ".join([core] + tags)


_PAREN = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")


def name_key(name: str) -> str:
    """The qualitative part of a name, with every number stripped.

    This is what §2.1's "distinct interpretable names" counted and what the §9
    criterion is evaluated on: `medium doublet bursting (2.5 spikes/burst,
    83 Hz interburst), +adapting` and the same cluster at 2.7 spikes/burst are
    one name, not two.
    """
    s = _PAREN.sub("", name)
    return " ".join(s.split()).strip().strip(",")


def describe(letter: str, med: Dict[str, float], size: int, cfg: Config) -> str:
    name = name_from_medians(med, cfg)
    cv = med.get("cv_isi", float("nan"))
    base = f"{letter} — {name} (n={size}, rate {med.get('rate', 0.0):.0f} Hz"
    return base + (f", CV {cv:.2f})" if np.isfinite(cv) else ")")


# ------------------------------------------------------- §8.4 nameability merge


def nameability_merge(labels: np.ndarray, desc: Dict[str, np.ndarray],
                      cfg: Config) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Merge clusters whose descriptor distributions are indistinguishable.

    Pairwise two-sided Mann-Whitney U on every descriptor in
    `MERGE_DESCRIPTORS`; a pair is indistinguishable when *every* descriptor's
    p-value exceeds the Bonferroni-corrected threshold
    `alpha / (n_pairs * n_descriptors)`.  Indistinguishable pairs are merged
    transitively (union-find), so a chain A~B~C becomes one cluster.

    Returns `(merged_labels, pair_rows, summary)`; `merged_labels` keeps the
    smallest original id of each merged group.
    """
    from scipy.stats import mannwhitneyu

    ids = np.unique(labels[labels >= 0])
    k = ids.size
    pairs = list(itertools.combinations(range(k), 2))
    n_desc = len(MERGE_DESCRIPTORS)
    if k < 2:
        return labels.copy(), [], {"n_before": int(k), "n_after": int(k),
                                   "alpha": float("nan"), "n_pairs": 0,
                                   "n_merged_pairs": 0}
    alpha = cfg.gates.merge_alpha / max(len(pairs) * n_desc, 1)

    cols = {d: [desc[d][labels == cid] for cid in ids] for d in MERGE_DESCRIPTORS}
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    rows: List[Dict] = []
    for i, j in pairs:
        ps = {}
        for d in MERGE_DESCRIPTORS:
            a = cols[d][i][np.isfinite(cols[d][i])]
            b = cols[d][j][np.isfinite(cols[d][j])]
            if a.size < 3 or b.size < 3:
                ps[d] = 1.0
                continue
            if np.array_equal(np.unique(a), np.unique(b)) and np.unique(a).size == 1:
                ps[d] = 1.0
                continue
            try:
                ps[d] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except ValueError:
                ps[d] = 1.0
        same = all(p > alpha for p in ps.values())
        if same:
            union(i, j)
            rows.append({"a": int(ids[i]), "b": int(ids[j]),
                         "min_p": float(min(ps.values())),
                         **{f"p_{d}": ps[d] for d in MERGE_DESCRIPTORS}})

    merged = labels.copy()
    groups: Dict[int, List[int]] = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    for root, members in groups.items():
        if len(members) == 1:
            continue
        keep = int(ids[min(members)])
        for m in members:
            merged[labels == ids[m]] = keep

    n_after = int(np.unique(merged[merged >= 0]).size)
    summary = {"n_before": int(k), "n_after": n_after, "alpha": float(alpha),
               "n_pairs": len(pairs), "n_merged_pairs": len(rows),
               "groups": [[int(ids[m]) for m in g] for g in groups.values()
                          if len(g) > 1]}
    return merged, rows, summary
