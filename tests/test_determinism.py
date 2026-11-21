"""§17.8 -- determinism where the spec pays for it."""
import numpy as np
import pytest

from izhiclust.config import DEFAULT, canonical_array
from izhiclust.features import band_features, make_plan, spectra


@pytest.mark.gpu
def test_identical_seed_gives_bitwise_identical_features(gpu, sim):
    cfg = DEFAULT
    plan = make_plan(cfg, cfg.grid, alpha=0.0)
    P = canonical_array()
    out = []
    for _ in range(2):
        res = sim.run(P, want_device_traces=True)
        F = np.asarray(band_features(spectra(res.traces, plan), plan).get())
        out.append(F)
        del res
    assert np.array_equal(out[0], out[1])


@pytest.mark.gpu
def test_lhs_and_sobol_designs_are_reproducible(gpu):
    from izhiclust import sampling
    assert np.array_equal(sampling.lhs(64, 42), sampling.lhs(64, 42))
    assert np.array_equal(sampling.sobol_pool(256, 7), sampling.sobol_pool(256, 7))
    x1, s1, r1 = sampling.first_iteration(DEFAULT)
    x2, s2, r2 = sampling.first_iteration(DEFAULT)
    assert np.array_equal(x1, x2) and np.array_equal(s1, s2) and r1 == r2
    # the 7 canonical vectors must be present exactly
    can = DEFAULT.space.to_unit(canonical_array())
    seeded = x1[s1 == "seed"]
    assert seeded.shape[0] == 7
    for row in can:
        assert np.isclose(np.abs(seeded - row).sum(axis=1).min(), 0.0, atol=1e-12)


def test_seeded_umap_fits_reproduce_bitwise():
    from izhiclust.cluster import umap_embed
    rng = np.random.default_rng(0)
    cent = rng.normal(0, 5, (6, 12))
    lab = rng.integers(0, 6, 800)
    P = (cent[lab] + rng.normal(0, 1, (800, 12))).astype(np.float32)
    kw = dict(n_neighbors=15, min_dist=0.0, n_epochs=200, random_state=42)
    a, _ = umap_embed(P, n_components=10, **kw)
    b, _ = umap_embed(P, n_components=10, **kw)
    assert np.array_equal(a, b)
    kw2 = dict(n_neighbors=15, min_dist=0.1, n_epochs=200, random_state=42)
    c, _ = umap_embed(P, n_components=2, **kw2)
    d, _ = umap_embed(P, n_components=2, **kw2)
    assert np.array_equal(c, d)


def test_canonical_seed_rows_are_in_canonical_order():
    """§13.1 reports type -> cluster, so the seed rows must stay ordered.

    Regression guard: the substituted LHS rows are not visited in index order,
    so collecting them with `np.where(source == 'seed')` scrambles the
    correspondence between a canonical type and its sample.
    """
    from izhiclust import sampling
    from izhiclust.config import DEFAULT, canonical_array
    x, src, rows = sampling.first_iteration(DEFAULT)
    can = DEFAULT.space.to_unit(canonical_array())
    assert len(rows) == can.shape[0] == 7
    assert len(set(rows)) == 7
    for k in range(7):
        assert np.allclose(x[rows[k]], can[k]), k
    assert int((src == "seed").sum()) == 7
    # and the rows are genuinely not sorted, which is what made the bug possible
    assert rows != sorted(rows)
