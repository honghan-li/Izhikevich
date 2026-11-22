"""v1 §17.8 — determinism where the spec pays for it."""
import numpy as np
import pytest

from izhiclust import isi as isim
from izhiclust.config import DEFAULT, canonical_array


@pytest.mark.gpu
def test_identical_seed_gives_bitwise_identical_spike_times(gpu, sim):
    cfg = DEFAULT
    P = canonical_array()
    out = []
    for _ in range(2):
        res = sim.run(P, cfg.model.I_legacy, store_traces=False,
                      max_spikes=cfg.isi.s_max)
        out.append(res.spikes.times.copy())
        del res
    assert np.array_equal(out[0], out[1])


@pytest.mark.gpu
def test_identical_seed_gives_bitwise_identical_features(gpu, sim):
    cfg = DEFAULT
    P = canonical_array()
    w = isim.Weights.default(cfg)
    out = []
    for _ in range(2):
        res = sim.run(P, cfg.model.I_legacy, store_traces=False,
                      max_spikes=cfg.isi.s_max)
        s = isim.isi_set(res.spikes, cfg.grid.t_transient, cfg.grid.t_steady_end)
        b = isim.compute_blocks(s, cfg)
        sc = isim.fit_scales(b, b, np.zeros((P.shape[0], 2)),
                             np.ones(P.shape[0], bool), cfg)
        out.append(isim.steady_vector(b, sc, w))
        del res
    assert np.array_equal(out[0], out[1])


def test_sobol_design_is_reproducible():
    from izhiclust import sampling
    assert np.array_equal(sampling.sobol(256, 7), sampling.sobol(256, 7))
    x1, s1, r1 = sampling.design(DEFAULT)
    x2, s2, r2 = sampling.design(DEFAULT)
    assert np.array_equal(x1, x2) and np.array_equal(s1, s2) and r1 == r2
    assert x1.shape[1] == DEFAULT.space.n_dim == 5
    can = DEFAULT.space.to_unit(canonical_array())[:, :4]
    seeded = x1[s1 == "seed"][:, :4]
    assert seeded.shape[0] == 7
    for row in can:
        assert np.isclose(np.abs(seeded - row).sum(axis=1).min(), 0.0, atol=1e-12)


def test_canonical_seed_rows_are_in_canonical_order():
    """§8.3 reports type -> cluster, so the seed rows must stay ordered."""
    from izhiclust import sampling
    x, src, rows = sampling.design(DEFAULT)
    can = DEFAULT.space.to_unit(canonical_array())[:, :4]
    assert len(rows) == can.shape[0] == 7
    assert len(set(rows)) == 7
    for k in range(7):
        assert np.allclose(x[rows[k], :4], can[k]), k
    assert int((src == "seed").sum()) == 7


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
