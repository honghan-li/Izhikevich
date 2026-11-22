"""C4 — the convergence gate (IMPROVEMENTS.md §5, fractional form §5.1)."""
import numpy as np
import pytest

from izhiclust.config import DEFAULT
from izhiclust.pipeline import _isi_error, _spike_time_error


def test_isi_error_ignores_a_common_time_shift():
    """Criterion 2 is about intervals: a shifted train has zero ISI error."""
    t = np.arange(500.0, 2500.0, 7.3)
    shifted = t + 0.3
    assert _isi_error(t, shifted) == pytest.approx(0.0, abs=1e-9)
    # while the absolute spike-time error sees the whole shift
    assert _spike_time_error(t, shifted) == pytest.approx(0.3, abs=1e-6)


def test_isi_error_measures_interval_distortion():
    t = np.arange(500.0, 2500.0, 10.0)
    jittered = t + np.arange(t.size) * 0.01      # 0.01 ms stretch per interval
    assert _isi_error(t, jittered) == pytest.approx(0.01, abs=1e-6)


def test_isi_error_uses_the_common_prefix_when_counts_differ():
    """§5 criterion 1 tolerates one extra spike, so criterion 2 must still
    be evaluable for such a probe."""
    t = np.arange(500.0, 2500.0, 10.0)
    got = np.concatenate([t, [2499.5]])
    assert np.isfinite(_isi_error(t, got))
    assert _isi_error(t, got) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(_spike_time_error(t, got))


@pytest.mark.gpu
@pytest.mark.slow
def test_gate_selects_a_step_and_reports_both_readings(gpu, sim):
    from izhiclust.pipeline import convergence_gate

    g = convergence_gate(DEFAULT, sim, log=lambda *a: None)
    num = DEFAULT.numerics
    assert g["passed"]
    assert g["dt_selected"] in num.gate_dts
    sel = g["stats_by_dt"][f"{g['dt_selected']:g}"]
    # the §5.1 fractional criteria, at the selected step
    assert sel["count_fraction"] >= num.gate_pass_fraction
    assert sel["max_count_diff"] <= num.gate_max_count_diff
    assert sel["isi_fraction"] >= num.gate_pass_fraction
    assert sel["max_isi_error_ms"] <= num.gate_isi_error_cap_ms
    assert sel["feature_fraction"] >= num.gate_pass_fraction
    # coarser steps on the ladder must not also qualify
    coarser = [dt for dt in num.gate_dts if dt > g["dt_selected"]]
    for dt in coarser:
        s = g["stats_by_dt"][f"{dt:g}"]
        assert not (s["count_fraction"] >= num.gate_pass_fraction
                    and s["max_count_diff"] <= num.gate_max_count_diff
                    and s["isi_fraction"] >= num.gate_pass_fraction
                    and s["max_isi_error_ms"] <= num.gate_isi_error_cap_ms
                    and s["feature_fraction"] >= num.gate_pass_fraction), dt
    # chaotic probes are excluded before the fractional rule is applied
    assert g["n_used"] + g["n_sensitivity_limited"] <= g["probes"]
    assert g["v_floor"] >= num.v_floor_rms_mv
