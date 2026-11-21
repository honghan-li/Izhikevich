"""§17.5 -- feature pipeline behaviour."""
import numpy as np

from izhiclust.config import DEFAULT
from izhiclust.features import band_features, make_plan, spectra, ac_power


def _tone(f_hz, amp=1.0):
    cfg = DEFAULT
    g = cfg.grid
    t = np.arange(g.n_out) * g.dt_out * 1e-3
    return (amp * np.sin(2 * np.pi * f_hz * t)).astype(np.float32)[None, :]


def test_band_plan_shape_and_coverage():
    cfg = DEFAULT
    plan = make_plan(cfg, cfg.grid, alpha=0.0)
    assert plan.n_bands == 128
    assert plan.band_edges[0] == 1
    assert plan.band_edges[-1] == plan.bl_hi + 1
    assert np.all(np.diff(plan.band_edges) >= 1)
    # 40 native bins of width df, then 88 log bands
    assert np.all(np.diff(plan.band_edges[:41]) == 1)
    assert abs(plan.band_centres[0] - 0.5) < 1e-9
    assert plan.band_centres[-1] < 1000.0
    w = plan.band_matrix.sum(axis=0)
    assert np.allclose(w, 1.0, atol=1e-6)


def test_sinusoid_lands_in_the_right_band():
    """>99% of banded energy in the correct band.

    A Hann main lobe is 4 bins wide (2 Hz at df = 0.5 Hz), so the
    one-band form of the criterion can only hold where a band is wider than
    that -- i.e. in the log-spaced region above ~90 Hz.  Inside the 40 native
    0.5 Hz bins the same criterion is applied to the band plus its two
    neighbours, and the peak band must still be the correct one.
    """
    cfg = DEFAULT
    plan = make_plan(cfg, cfg.grid, alpha=0.0, deconvolve=False)
    for f in (7.5, 19.5, 55.0, 220.0, 777.0):
        mag = np.asarray(spectra(_tone(f), plan))[0]
        energy = mag[1:plan.bl_hi + 1] ** 2
        tot = energy.sum()
        j = int(np.argmin(np.abs(plan.band_centres - f)))
        width = int(plan.band_edges[j + 1] - plan.band_edges[j])
        spread = 0 if width >= 8 else 2
        lo = int(plan.band_edges[max(j - spread, 0)])
        hi = int(plan.band_edges[min(j + spread + 1, plan.n_bands)])
        frac = energy[lo - 1:hi - 1].sum() / tot
        assert frac > 0.99, (f, j, width, frac)
        banded = plan.band_matrix.T @ energy
        assert int(np.argmax(banded)) == j, (f, j, int(np.argmax(banded)))


def test_l2_normalised_and_loudness_invariant_shape():
    cfg = DEFAULT
    plan = make_plan(cfg, cfg.grid, alpha=0.0)
    F1 = np.asarray(band_features(spectra(_tone(100.0, 1.0), plan), plan))
    F2 = np.asarray(band_features(spectra(_tone(100.0, 50.0), plan), plan))
    assert abs(np.linalg.norm(F1) - 1.0) < 1e-5
    assert abs(np.linalg.norm(F2) - 1.0) < 1e-5
    # log1p is nonlinear, so the vector is not bit-identical under rescaling;
    # what the L2 step buys is that *shape* survives a 50x loudness change.
    cos = float(F1[0] @ F2[0])
    assert cos > 0.98, cos
    assert int(np.argmax(F1)) == int(np.argmax(F2))


def test_tilt_reweights_log_f_linearly():
    cfg = DEFAULT
    p0 = make_plan(cfg, cfg.grid, alpha=0.0)
    p1 = make_plan(cfg, cfg.grid, alpha=1.0)
    f = p1.band_centres
    r = p1.tilt[1] / p0.tilt[1]
    assert r > 1.0
    k = int(np.argmin(np.abs(p1.freqs - 100.0)))
    assert abs(p1.tilt[k] - (cfg.features.f_ref / p1.freqs[k])) < 1e-9
    assert p0.tilt.max() == 1.0 and p0.tilt.min() == 1.0


def test_ac_power_excludes_dc():
    cfg = DEFAULT
    plan = make_plan(cfg, cfg.grid, alpha=0.0)
    x = np.full((1, cfg.grid.n_out), 42.0, dtype=np.float32)
    assert float(np.asarray(ac_power(spectra(x, plan), plan))[0]) < 1e-6
