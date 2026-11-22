"""C1 — the ISI encoding (IMPROVEMENTS.md §3.2, §3.3, §3.4, §8.1).

Replaces the v1 spectral feature tests (§3.5 retires that pipeline).
"""
import numpy as np
import pytest
from dataclasses import replace

from izhiclust import isi as isim
from izhiclust.config import DEFAULT
from izhiclust.simulate import SpikeTrains


def trains(list_of_times):
    counts = np.array([len(t) for t in list_of_times], dtype=np.int64)
    off = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    times = np.concatenate([np.asarray(t, dtype=np.float32)
                            for t in list_of_times])
    return SpikeTrains(times, off, np.zeros(counts.size, dtype=bool))


def tonic(rate_hz, t_lo=500.0, t_hi=2500.0, phase=0.0):
    isi = 1000.0 / rate_hz
    return np.arange(t_lo + phase, t_hi, isi)


def doublet(burst_hz, short_isi=2.0, t_lo=500.0, t_hi=2500.0):
    """Two-spike bursts at `burst_hz`."""
    out = []
    t = t_lo
    while t < t_hi:
        out += [t, t + short_isi]
        t += 1000.0 / burst_hz
    return np.array([v for v in out if v < t_hi])


CFG = DEFAULT
G = CFG.grid


def blocks_of(times_list, h=None):
    sp = trains(times_list)
    s = isim.isi_set(sp, G.t_transient, G.t_steady_end)
    return s, isim.compute_blocks(s, CFG, h=h)


def test_block_shapes_and_total_width():
    s, b = blocks_of([tonic(40.0)])
    assert b.A.shape[1] == CFG.isi.n_a == 32
    assert b.B.shape[1] == 144
    assert b.C.shape[1] == 8
    assert b.D.shape[1] == 1
    assert CFG.isi.n_steady == 185
    assert len(CFG.isi.columns()) == 185


def test_tonic_is_a_point_on_the_diagonal():
    """A tonic train is one point at the origin of the return map (§3.2 B)."""
    s, b = blocks_of([tonic(40.0)])
    nb = CFG.isi.n_b
    B = b.B[0].reshape(nb, nb)
    i, j = np.unravel_index(np.argmax(B), B.shape)
    g = CFG.isi.grid_b()
    assert abs(g[i]) < 0.12 and abs(g[j]) < 0.12, (g[i], g[j])
    # mass concentrated on (not merely near) the diagonal: with h = 0.25 and a
    # 0.25-wide grid the kernel spills one cell either way, so the test is the
    # band |i - j| <= 1
    band = sum(B[i, j] for i in range(nb) for j in range(nb) if abs(i - j) <= 1)
    assert band / B.sum() > 0.70      # the h = grid-spacing kernel's own spill
    # block A peaks at lambda = 0 and block C is flat at zero
    assert abs(CFG.isi.grid_a()[int(np.argmax(b.A[0]))]) < 0.12
    assert np.allclose(b.C[0], 0.0, atol=1e-6)


def test_doublet_is_two_mirrored_off_diagonal_points():
    s, b = blocks_of([doublet(20.0)])
    nb = CFG.isi.n_b
    B = b.B[0].reshape(nb, nb)
    g = CFG.isi.grid_b()
    # the two largest cells are off the diagonal and mirror each other
    flat = np.argsort(-B.ravel())[:2]
    pts = [np.unravel_index(k, B.shape) for k in flat]
    (i1, j1), (i2, j2) = pts
    assert i1 != j1 and i2 != j2
    assert abs(g[i1] - g[j2]) < 0.3 and abs(g[j1] - g[i2]) < 0.3
    band = sum(B[i, j] for i in range(nb) for j in range(nb) if abs(i - j) <= 1)
    assert band / B.sum() < 0.50      # vs > 0.70 for the tonic train


def test_encoding_is_rate_invariant_except_block_d():
    """A 3 Hz and a 30 Hz doublet burster share blocks A-C (§3.2)."""
    s, b = blocks_of([doublet(5.0, short_isi=8.0), doublet(50.0, short_isi=0.8)])
    # A and B are rate-invariant up to the finite-count shift of Ibar (the
    # 19-ISI train has one more short interval than long, the 199-ISI one does
    # too, but the ratio differs); C is a coarse 8-segment trend, so it only
    # agrees to the binning of an alternating sequence.
    for M, tol in ((b.A, 0.99), (b.B, 0.99), (b.C, 0.9)):
        n0 = np.linalg.norm(M[0])
        n1 = np.linalg.norm(M[1])
        cos = float(M[0] @ M[1] / max(n0 * n1, 1e-30))
        assert cos > tol, (M.shape, cos)
    assert abs((b.D[0, 0] - b.D[1, 0]) - 1.0) < 0.02      # one decade apart


def test_kde_is_smooth_not_binned():
    """Two nearby trains must be nearby, not maximally different (§3.2)."""
    s, b = blocks_of([tonic(40.0), tonic(40.4), tonic(120.0)])
    A = b.A / np.linalg.norm(b.A, axis=1, keepdims=True)
    near = 1.0 - float(A[0] @ A[1])
    far = 1.0 - float(A[0] @ A[2])
    assert near < 1e-3, near
    assert far <= 1e-6 or far >= near            # same shape, different rate


def test_block_c_detects_drift():
    adapting = np.cumsum(np.concatenate([[500.0], np.linspace(8.0, 25.0, 300)]))
    adapting = adapting[adapting < 2500.0]
    s, b = blocks_of([adapting, tonic(60.0)])
    assert np.linalg.norm(b.C[0]) > 10 * np.linalg.norm(b.C[1])
    assert b.C[0, 0] < b.C[0, -1]                # intervals grow with time


def test_assembled_vector_is_unit_norm_and_weighted():
    times = [tonic(40.0), doublet(20.0), tonic(200.0)]
    sp = trains(times)
    st = isim.isi_set(sp, G.t_transient, G.t_steady_end)
    on = isim.isi_set(sp, 0.0, G.t_transient)
    bs = isim.compute_blocks(st, CFG)
    bo = isim.compute_blocks(on, CFG)
    desc = isim.descriptors(st, G.window_seconds, CFG)
    ex = isim.onset_extras(on, sp, CFG, desc["mean_isi"])
    mask = np.ones(3, dtype=bool)
    sc = isim.fit_scales(bs, bo, ex, mask, CFG)
    w = isim.Weights.default(CFG)
    F, Fs, Fo = isim.assemble(bs, bo, ex, sc, w)
    assert F.shape == (3, CFG.isi.n_features)
    assert Fs.shape == (3, CFG.isi.n_steady)
    assert Fo.shape == (3, CFG.isi.n_onset)
    assert np.allclose(np.linalg.norm(F, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(Fs, axis=1), 1.0, atol=1e-5)
    # w_A = 0 must switch block A off entirely (it is in the sweep grid)
    F0, _, _ = isim.assemble(bs, bo, ex, sc, replace(w, w_a=0.0))
    assert np.allclose(F0[:, :CFG.isi.n_a], 0.0)


def test_descriptors_from_exact_spike_times():
    sp = trains([tonic(50.0), doublet(20.0, short_isi=2.0)])
    s = isim.isi_set(sp, G.t_transient, G.t_steady_end)
    d = isim.descriptors(s, G.window_seconds, CFG)
    assert abs(d["rate"][0] - 50.0) < 0.5
    assert d["cv_isi"][0] < 1e-6
    assert d["burst_index"][0] == 0.0
    # a doublet burster: half the ISIs are short, ~2 spikes per burst
    assert 0.4 < d["burst_index"][1] < 0.6
    assert 1.8 < d["spikes_per_burst"][1] < 2.2
    assert d["cv_isi"][1] > 0.5


def test_classes_split_on_sd_v_and_isi_count():
    n_isi = np.array([0.0, 0.0, 4.0, 50.0])
    diag = {"finite": np.ones(4), "n_spikes_all": np.array([0.0, 0.0, 5.0, 51.0]),
            "sd_v": np.array([0.01, 1.0, 5.0, 20.0])}
    cls = isim.classify(diag, n_isi, 0.1, CFG)
    assert cls.quiescent.tolist() == [True, False, False, False]
    assert cls.subthreshold.tolist() == [False, True, False, False]
    assert cls.sparse.tolist() == [False, False, True, False]
    assert cls.active.tolist() == [False, False, False, True]


def test_health_gates_on_a_tonic_rate_ladder():
    """§8.1: the metric must be monotone in rate separation, octave-ordered.

    This is the v1 §2.3 pathology as a test: a ladder of pure tonic trains over
    three decades.  The v1 encoding fails both gates on this input; the §3.2
    encoding must pass.
    """
    rates = np.geomspace(4.0, 400.0, 60)
    sp = trains([tonic(r) for r in rates])
    s = isim.isi_set(sp, G.t_transient, G.t_steady_end)
    b = isim.compute_blocks(s, CFG)
    d = isim.descriptors(s, G.window_seconds, CFG)
    mask = np.ones(rates.size, dtype=bool)
    sc = isim.fit_scales(b, b, np.zeros((rates.size, 2)), mask, CFG)
    w = isim.Weights.default(CFG)
    F = isim._l2_rows(isim.steady_vector(b, sc, w))
    from izhiclust.cluster import fit_pca_adaptive
    _, _, curve = fit_pca_adaptive(F, 2, 0.99)
    res = isim.health_gates(F, d["rate"], d["cv_isi"], d["burst_index"],
                            replace(CFG, gates=replace(CFG.gates,
                                                       tonic_subset=200)),
                            curve)
    assert res["pca_pass"], res["pca_components_99"]
    assert res["monotonicity_pass"], res["near_over_far"]
    assert res["octave_pass"], (res["octave_mean_distance"],
                                res["ratio15_mean_distance"])


def test_window_gate_keeps_97_percent_of_a_realistic_mix():
    """The §3.4 `n_ISI >= 10` gate is a gate, not a silent drop."""
    rates = np.concatenate([np.full(97, 40.0), np.full(3, 2.0)])
    sp = trains([tonic(r) for r in rates])
    s = isim.isi_set(sp, G.t_transient, G.t_steady_end)
    d = isim.descriptors(s, G.window_seconds, CFG)
    diag = {"finite": np.ones(100), "n_spikes_all": d["n_spikes"],
            "sd_v": np.full(100, 20.0)}
    cls = isim.classify(diag, d["n_isi"], 0.1, CFG)
    assert int(cls.active.sum()) == 97
    assert int(cls.sparse.sum()) == 3
