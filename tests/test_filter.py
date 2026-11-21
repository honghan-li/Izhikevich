"""§17.3 output filter and §17.4 alias suppression."""
import numpy as np
import pytest

from filter_reference import cascaded_boxcar_decimate, filter_only, triangular_fir
from izhiclust.config import DEFAULT, canonical_array
from izhiclust.features import _boxcar_response, make_plan, spectra
from izhiclust.simulate import grid_for_dt
from dataclasses import replace


def test_analytic_response_matches_measured():
    """|H(f)|^2 to < 0.1 dB over 0-10 kHz (away from the nulls)."""
    L, dt_ms = 10, 0.01
    fs = 1.0 / (dt_ms * 1e-3)                    # 100 kHz
    freqs = np.concatenate([np.arange(10.0, 1000.0, 10.0),
                            np.arange(1000.0, 9000.0, 50.0)])
    n = 200000
    t = np.arange(n) / fs
    h = triangular_fir(L)
    worst = 0.0
    for f in freqs:
        x = np.sin(2 * np.pi * f * t)
        y = np.convolve(x, h, mode="valid")
        amp = np.abs(y[1000:]).max()             # steady state
        pred = _boxcar_response(np.array([f]), L, dt_ms, 2)[0]
        db = 20 * np.log10(max(amp, 1e-18) / max(pred, 1e-18))
        worst = max(worst, abs(db))
    assert worst < 0.1, f"worst deviation {worst:.3f} dB"


def test_nulls_at_multiples_of_output_rate():
    for f in (10000.0, 20000.0, 30000.0):
        assert _boxcar_response(np.array([f]), 10, 0.01, 2)[0] < 1e-12


def test_in_band_tilt_is_small():
    h1 = _boxcar_response(np.array([1000.0]), 10, 0.01, 1)[0]
    h2 = _boxcar_response(np.array([1000.0]), 10, 0.01, 2)[0]
    assert abs(20 * np.log10(h1) + 0.14) < 0.02
    assert abs(20 * np.log10(h2) + 0.28) < 0.03
    assert 20 * np.log10(_boxcar_response(np.array([9000.0]), 10, 0.01, 2)[0]) < -37.0


def test_deconvolution_restores_amplitude():
    """§9 step 5 restores a known sinusoid's amplitude to < 0.5%."""
    cfg = DEFAULT
    g = cfg.grid
    plan = make_plan(cfg, g, alpha=0.0)
    fs_fine = 1.0 / (g.dt_int * 1e-3)
    n_fine = g.n_out * g.L
    t = np.arange(n_fine) / fs_fine
    for f in (37.0, 300.0, 900.0):
        x = np.sin(2 * np.pi * f * t)
        y = cascaded_boxcar_decimate(x, g.L).astype(np.float32)[None, :]
        mag = np.asarray(spectra(y, plan))[0]
        k = int(round(f / (plan.freqs[1] - plan.freqs[0])))
        # Hann window coherent gain 0.5, one-sided amplitude = 2|X|/N
        amp = 2.0 * mag[k - 2:k + 3].max() / (0.5 * g.n_out)
        assert abs(amp - 1.0) < 5e-3, (f, amp)


def test_alias_suppression():
    """§17.4: >= 50 dB alias rejection in 0-1 kHz with the filter, ~20 dB without.

    The probe is a 100 Hz step-discontinuity train (1/f harmonics) whose
    content in the fold band 9-11 kHz is placed off the 100 Hz comb, so the
    folded energy is separable from the legitimate in-band harmonics.
    """
    L, dt_ms = 10, 0.01
    fs = 1.0 / (dt_ms * 1e-3)
    n = 200000                                     # 2000 ms at 100 kHz
    t = np.arange(n) / fs
    x = np.zeros(n)
    for k in range(1, 51):                         # legitimate 0-5 kHz comb
        x += np.sin(2 * np.pi * 100.0 * k * t) / k
    # fold-band content placed so its aliases (|f - 10 kHz|) land 50 Hz off the
    # 100 Hz comb, which keeps folded energy separable from legitimate content
    offs = 150.0 + 100.0 * np.arange(9)
    fold = np.concatenate([10000.0 + offs, 10000.0 - offs])
    for f in fold:
        x += np.sin(2 * np.pi * f * t) / (f / 100.0)

    ref_amp = 1.0 / 10.0                           # the legitimate 1 kHz harmonic
    out = {}
    for name, y in (("off", x[L - 1::L]), ("on", cascaded_boxcar_decimate(x, L))):
        m = y.size
        X = np.abs(np.fft.rfft(y * np.hanning(m)))
        fr = np.fft.rfftfreq(m, d=L * dt_ms * 1e-3)
        amp = 2.0 * X / (0.5 * m)
        worst = 0.0
        for f in fold:
            fa = abs(f - 10000.0)
            j = int(round(fa / (fr[1] - fr[0])))
            worst = max(worst, amp[max(j - 2, 0):j + 3].max())
        out[name] = 20 * np.log10(ref_amp / worst)
    assert out["on"] >= 50.0, out
    assert 15.0 <= out["off"] <= 25.0, out


@pytest.mark.gpu
def test_kernel_filter_matches_host_reference(gpu, sim):
    """The kernel's register boxcar equals the host triangular FIR exactly."""
    cfg = DEFAULT
    short = replace(cfg.grid, t_total=50.0)
    g_fine = grid_for_dt(short, cfg.grid.dt_int, dt_out=cfg.grid.dt_int)   # L = 1
    g_out = grid_for_dt(short, cfg.grid.dt_int, dt_out=cfg.grid.dt_out)    # L = 10
    P = canonical_array()
    raw = sim.run(P, grid=g_fine, out_start=0, n_out=g_fine.n_out_total,
                  refine=False).traces
    got = sim.run(P, grid=g_out, out_start=0, n_out=g_out.n_out_total,
                  refine=False).traces
    for i in range(P.shape[0]):
        want = cascaded_boxcar_decimate(raw[i].astype(np.float64), g_out.L)
        assert np.allclose(got[i], want, atol=2e-3, rtol=1e-5), i
