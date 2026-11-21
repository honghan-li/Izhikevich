"""LHS design, Sobol candidate pool and the canonical seeds (SPEC.md §11.1)."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .config import Config, canonical_array


def lhs(n: int, seed: int, d: int = 4, optimize: bool = True) -> np.ndarray:
    """Scrambled LHS on [0,1]^d with `optimization='random-cd'` (§11.1)."""
    kw = {"d": d, "scramble": True, "seed": seed}
    if optimize:
        kw["optimization"] = "random-cd"
    return qmc.LatinHypercube(**kw).random(n)


def sobol_pool(n: int, seed: int, d: int = 4) -> np.ndarray:
    """Scrambled-Sobol candidate pool (§11.3)."""
    return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def first_iteration(cfg: Config) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Iteration-1 design: LHS with the 7 canonical vectors substituted in.

    Returns (x, source, seed_rows) with source in {'seed', 'lhs'} and
    `seed_rows[k]` the row holding canonical type `k` **in CANONICAL order**
    -- the order matters, because §13.1 reports type -> cluster and the rows
    are not visited in index order.  The substituted LHS points are the ones
    nearest each canonical vector, so the design stays close to space-filling.
    """
    n = cfg.schedule.n_first
    X = lhs(n, cfg.schedule.seed)
    can_x = cfg.space.to_unit(canonical_array())
    src = np.array(["lhs"] * n, dtype=object)
    taken: set[int] = set()
    seed_rows: list[int] = []
    for row in can_x:
        d2 = ((X - row) ** 2).sum(axis=1)
        for j in np.argsort(d2):
            if int(j) not in taken:
                taken.add(int(j))
                X[j] = row
                src[j] = "seed"
                seed_rows.append(int(j))
                break
    return X, src, seed_rows
