"""C6 — a single Sobol design over the 5-D domain (IMPROVEMENTS.md §7).

The v1 GP active learner is retired: it cost 4x the entire integration budget
while steering on a feature-space signal §2.3 shows to be degenerate.  Every
sample is now design-unbiased, so the "all" vs "LHS-only (authoritative)"
split in v1 §14.1 collapses to one set of statistics.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy.stats import qmc

from .config import Config, canonical_array


def lhs(n: int, seed: int, d: int = 4, optimize: bool = True) -> np.ndarray:
    """Scrambled LHS on [0,1]^d -- kept for the convergence-gate probes."""
    kw = {"d": d, "scramble": True, "seed": seed}
    if optimize:
        kw["optimization"] = "random-cd"
    return qmc.LatinHypercube(**kw).random(n)


def sobol(n: int, seed: int, d: int = 5) -> np.ndarray:
    """Scrambled Sobol points on [0,1]^d, using `random_base2` when it fits."""
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(round(math.log2(n)))
    if 2 ** m == n:
        return eng.random_base2(m)
    return eng.random(n)


def design(cfg: Config) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """The whole design: Sobol with the 7 canonical vectors substituted in.

    Returns `(x, source, seed_rows)` with `source` in {'seed', 'sobol'} and
    `seed_rows[k]` the row holding canonical type `k` in CANONICAL order.  The
    substituted points are the ones nearest each canonical vector in the
    `(a,b,c,d)` sub-space, so the design stays close to space-filling; their
    `Delta` column is kept but ignored, because the canonical seeds are driven
    at v1's `I = 10` (§4.2, assumption V2-A6).
    """
    n = cfg.schedule.n_total
    X = sobol(n, cfg.schedule.seed, d=cfg.space.n_dim)
    can_x = cfg.space.to_unit(canonical_array())[:, :4]
    src = np.array(["sobol"] * n, dtype=object)
    taken: set[int] = set()
    seed_rows: List[int] = []
    for row in can_x:
        d2 = ((X[:, :4] - row) ** 2).sum(axis=1)
        for j in np.argsort(d2):
            if int(j) not in taken:
                taken.add(int(j))
                X[j, :4] = row
                src[j] = "seed"
                seed_rows.append(int(j))
                break
    return X, src, seed_rows
