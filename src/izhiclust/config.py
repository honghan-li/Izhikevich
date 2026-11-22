"""Every constant in SPEC.md as amended by IMPROVEMENTS.md, as frozen dataclasses.

Section references prefixed `v1 §` point at SPEC.md; bare `§` at
IMPROVEMENTS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------- model (§4)


@dataclass(frozen=True)
class Model:
    """The hybrid system.  `I` is no longer a constant (§4): it is set per
    sample as `I*(theta) + Delta`, so only the initial condition and the peak
    live here."""
    v0: float = -70.0          # initial membrane potential
    v_peak: float = 30.0       # reset threshold / equalised peak
    I_legacy: float = 10.0     # v1's fixed drive, kept for the canonical seeds
    # u0 = b * v0 (the paper's own initialisation) -- derived, not stored.


def i_saddle_node(b: np.ndarray) -> np.ndarray:
    """§4.1: `I_SN = (5-b)^2 / 0.16 - 140`, the fold of the continuous system."""
    b = np.asarray(b, dtype=np.float64)
    return (5.0 - b) ** 2 / 0.16 - 140.0


def i_andronov_hopf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """§4.1: `I_AH = [(5-b)^2 - (b-a)^2] / 0.16 - 140`, NaN where `b <= a`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = ((5.0 - b) ** 2 - (b - a) ** 2) / 0.16 - 140.0
    return np.where(b > a, out, np.nan)


def excitability_class(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """§4.1: Class 1 where `b <= a` (no Hopf), Class 2 where `b > a`."""
    return np.where(np.asarray(b) > np.asarray(a), 2, 1).astype(np.int8)


# ------------------------------------------------- coefficient space (v1 §3)


@dataclass(frozen=True)
class Space:
    """`(a, b, c, d)` as in v1 §3, plus the overdrive `Delta` as a fifth
    log-spaced design axis (§4.2, assumption V2-A1)."""
    a_lo: float = 0.005
    a_hi: float = 0.20        # log-uniform
    b_lo: float = -0.10
    b_hi: float = 0.35
    c_lo: float = -80.0
    c_hi: float = -40.0
    d_lo: float = 0.0
    d_hi: float = 12.0
    delta_lo: float = 1.0     # overdrive above rheobase
    delta_hi: float = 15.0    # log-uniform

    names: Tuple[str, str, str, str, str] = ("a", "b", "c", "d", "delta")
    n_dim: int = 5

    def to_native(self, x: np.ndarray) -> np.ndarray:
        """x in [0,1]^5 -> (a, b, c, d, Delta).  Accepts 4 columns too."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        k = x.shape[1]
        out = np.empty((x.shape[0], k), dtype=np.float64)
        out[:, 0] = self.a_lo * (self.a_hi / self.a_lo) ** x[:, 0]   # log-uniform
        out[:, 1] = self.b_lo + (self.b_hi - self.b_lo) * x[:, 1]
        out[:, 2] = self.c_lo + (self.c_hi - self.c_lo) * x[:, 2]
        out[:, 3] = self.d_lo + (self.d_hi - self.d_lo) * x[:, 3]
        if k > 4:
            out[:, 4] = self.delta_lo * (self.delta_hi / self.delta_lo) ** x[:, 4]
        return out

    def to_unit(self, p: np.ndarray) -> np.ndarray:
        """(a, b, c, d[, Delta]) -> x in [0,1]^k."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        k = p.shape[1]
        out = np.empty((p.shape[0], k), dtype=np.float64)
        out[:, 0] = (np.log(p[:, 0]) - math.log(self.a_lo)) / (
            math.log(self.a_hi) - math.log(self.a_lo))
        out[:, 1] = (p[:, 1] - self.b_lo) / (self.b_hi - self.b_lo)
        out[:, 2] = (p[:, 2] - self.c_lo) / (self.c_hi - self.c_lo)
        out[:, 3] = (p[:, 3] - self.d_lo) / (self.d_hi - self.d_lo)
        if k > 4:
            out[:, 4] = (np.log(p[:, 4]) - math.log(self.delta_lo)) / (
                math.log(self.delta_hi) - math.log(self.delta_lo))
        return out


# ------------------------------------------------------------ time grid (§6)


@dataclass(frozen=True)
class Grid:
    """Two analysis windows now (§6): onset `[0, t_transient)` and steady
    `[t_transient, t_steady_end)`.  `t_total` carries an extra
    `window_shift` ms so the §8.2 window-shift test can compare two windows of
    *equal length* (assumption V2-A3)."""
    t_total: float = 2600.0     # ms simulated
    t_transient: float = 500.0  # ms; end of the onset window
    t_steady_end: float = 2500.0  # ms; end of the steady-state window
    window_shift: float = 100.0   # ms; the §8.2 shifted window is [600, 2600)
    dt_int: float = 0.01        # ms
    L: int = 10                 # integration steps per output sample

    @property
    def dt_out(self) -> float:
        return self.dt_int * self.L

    @property
    def n_steps(self) -> int:
        return int(round(self.t_total / self.dt_int))

    @property
    def n_out_total(self) -> int:
        return self.n_steps // self.L

    @property
    def out_start(self) -> int:
        """0-based index of the first steady-window output sample."""
        return int(round(self.t_transient / self.dt_out)) - 1

    @property
    def n_out(self) -> int:
        """Output samples from `t_transient` to `t_steady_end`."""
        return int(round((self.t_steady_end - self.t_transient) / self.dt_out))

    @property
    def win_start_step(self) -> int:
        """1-based integration step whose end is t = t_transient."""
        return int(round(self.t_transient / self.dt_int))

    @property
    def win_end_step(self) -> int:
        return int(round(self.t_steady_end / self.dt_int))

    @property
    def t_window(self) -> np.ndarray:
        return self.t_transient + self.dt_out * np.arange(self.n_out)

    @property
    def t_full(self) -> np.ndarray:
        return self.dt_out * (1.0 + np.arange(self.n_out_total))

    @property
    def window_seconds(self) -> float:
        return (self.t_steady_end - self.t_transient) / 1000.0

    @property
    def onset_seconds(self) -> float:
        return self.t_transient / 1000.0


# ------------------------------------------------- ISI encoding (§3.2, §3.3)


@dataclass(frozen=True)
class Isi:
    """C1: the clustering input is an encoding of the ISI sequence."""
    clip_lo: float = -1.75      # log10(I/Ibar) grid, §3.2
    clip_hi: float = 1.0
    n_a: int = 32               # block A: 1-D KDE points
    n_b: int = 12               # block B: n_b x n_b return-map KDE grid
    n_c: int = 8                # block C: drift resample points
    h: float = 0.25             # KDE bandwidth in log10 units
    h_choices: Tuple[float, ...] = (0.15, 0.25, 0.40)
    w_a: float = 0.5
    w_b: float = 1.0
    w_c: float = 0.5
    w_d: float = 0.3
    w_onset: float = 0.3        # §6; 0 must stay reachable in the sweep
    c_scale_floor: float = 0.01   # decades per point; floors block C's scale
    d_scale_floor: float = 0.05   # decades; floors block D's z-score scale
    n_isi_min: int = 10         # §3.4 `W` sparse-class gate
    n_isi_min_onset: int = 2    # below this the onset blocks are left at zero
    onset_count_ms: float = 100.0   # §6: spikes in the first 100 ms
    s_max: int = 8192           # §3.1 capacity; 2x, see V2-A13

    @property
    def n_block_b(self) -> int:
        return self.n_b * self.n_b

    @property
    def n_steady(self) -> int:
        return self.n_a + self.n_block_b + self.n_c + 1

    @property
    def n_onset(self) -> int:
        return self.n_steady + 2

    @property
    def n_features(self) -> int:
        return self.n_steady + self.n_onset

    def grid_a(self) -> np.ndarray:
        return np.linspace(self.clip_lo, self.clip_hi, self.n_a)

    def grid_b(self) -> np.ndarray:
        return np.linspace(self.clip_lo, self.clip_hi, self.n_b)

    def columns(self) -> list[str]:
        """Column names for `features_isi.csv` (steady-state block)."""
        cols = [f"A_lam={v:+.4f}" for v in self.grid_a()]
        gb = self.grid_b()
        cols += [f"B_lam={gb[i]:+.3f}_next={gb[j]:+.3f}"
                 for i in range(self.n_b) for j in range(self.n_b)]
        cols += [f"C_t={k / max(self.n_c - 1, 1):.3f}" for k in range(self.n_c)]
        cols += ["D_log10_mean_isi"]
        return cols

    def onset_columns(self) -> list[str]:
        return ["onset_" + c for c in self.columns()] + [
            "onset_n_spikes_first_100ms", "onset_first_isi_over_steady_mean"]


# ----------------------------------------------- drive / rheobase (§4)


@dataclass(frozen=True)
class Drive:
    """C3: the numerical rheobase search of the hybrid system."""
    i_max: float = 30.0         # upper end of the bisection bracket
    bracket_pad: float = 1.0    # start at min(I_SN, I_AH) - 1
    tol: float = 0.1            # bisection tolerance in I
    t_total: float = 500.0      # ms per bisection evaluation
    t_measure: float = 200.0    # ms; spikes in the last 200 ms decide
    dt_int: float = 0.05        # ms; coarse step for the search (V2-A2)
    max_iter: int = 10
    canonical_drive: float = 10.0   # the published seeds keep v1's I = 10


# ---------------------------------------------- sampling schedule (§7)


@dataclass(frozen=True)
class Schedule:
    """C6: one Sobol sequence over the 5-D design domain, no active learning."""
    n_total: int = 65536        # 2^16, for Sobol balance (assumption V2-A4)
    chunk: int = 2048           # GPU chunk size (v1 §7.5)
    seed: int = 42


# ---------------------------------- dimension reduction / clustering (v1 §12)


@dataclass(frozen=True)
class Clustering:
    pca_components: int = 20        # §8.1 gate: 99% of variance within 20
    pca_min_evr: float = 0.99
    pca_gate_components: int = 20
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    umap_components: int = 10
    umap_epochs: int = 500
    umap_metric: str = "euclidean"
    vis_min_dist: float = 0.1
    vis_components: int = 2
    min_cluster_size: int = 60
    min_samples: int = 10
    selection_method: str = "leaf"
    random_state: int = 42
    noise_penalty_threshold: float = 0.25
    sweep_neighbors: Tuple[int, ...] = (15, 30, 50)
    sweep_min_cluster_size: Tuple[int, ...] = (30, 60, 120)
    sweep_min_samples: Tuple[int, ...] = (5, 10, 25)
    sweep_methods: Tuple[str, ...] = ("leaf", "eom")
    sweep_fast_min_samples: Tuple[int, ...] = (5, 10)
    sweep_subsample: int = 20000    # configs are compared on a subsample
    stability_fits: int = 5
    stability_subsample: int = 32000   # ~50% of the clustered set
    bootstrap_fraction: float = 0.8
    persistence_jaccard: float = 0.5
    persistence_floor: float = 0.6
    contiguity_knn: int = 5
    contiguity_percentile: float = 95.0
    contiguity_min_frac: float = 0.05
    n_representatives: int = 3
    # §8.2 ARI thresholds
    ari_seed: float = 0.6
    ari_bootstrap: float = 0.6
    ari_window: float = 0.8
    ari_dt: float = 0.9


# ----------------------------------------------------- validation (§8.1, §8.3)


@dataclass(frozen=True)
class Gates:
    """§8.1 representation-health gates and the §9 run-level thresholds."""
    pca_max_components: int = 20
    tonic_cv_max: float = 0.02      # §2.3 pure-tonic subset definition
    tonic_bi_max: float = 0.02
    tonic_subset: int = 4000
    near_window: Tuple[float, float] = (0.05, 0.10)   # |d log f0|
    far_threshold: float = 1.0
    monotonicity_ratio: float = 0.50
    octave_tol: float = 0.02        # |d log f0 - log10 r| tolerance
    noise_fraction_max: float = 0.25
    gated_cluster_fraction: float = 0.60
    distinct_names_min: int = 8
    burst_capture_fraction: float = 0.50
    burst_index_threshold: float = 0.2
    merge_alpha: float = 0.01       # §8.4 Bonferroni-corrected Mann-Whitney
    roundtrip_tol: float = 1e-6


CANONICAL = (
    # name, label, a, b, c, d
    ("regular spiking", "RS", 0.02, 0.20, -65.0, 8.0),
    ("intrinsically bursting", "IB", 0.02, 0.20, -55.0, 4.0),
    ("chattering", "CH", 0.02, 0.20, -50.0, 2.0),
    ("fast spiking", "FS", 0.10, 0.20, -65.0, 2.0),
    ("low-threshold spiking", "LTS", 0.02, 0.25, -65.0, 2.0),
    ("thalamo-cortical", "TC", 0.02, 0.25, -65.0, 0.05),
    ("resonator", "RZ", 0.10, 0.26, -65.0, 2.0),
)

# §8.3: types the constant-drive protocol cannot reach are exempted by name.
CANONICAL_EXEMPT = ("TC", "RZ")


def canonical_array() -> np.ndarray:
    return np.array([[r[2], r[3], r[4], r[5]] for r in CANONICAL], dtype=np.float64)


def canonical_labels() -> list[str]:
    return [r[1] for r in CANONICAL]


# ------------------------------------------------ numerics / gates (§5)


@dataclass(frozen=True)
class Numerics:
    local_error_tol: float = 1e-6     # v1 §7.3 refinement trigger
    refine_factor: int = 4            # dt_int / 4
    gate_dts: Tuple[float, ...] = (0.04, 0.02, 0.01, 0.005, 0.0025, 0.001)
    gate_dt_accept: float = 0.01
    gate_dt_ref: float = 0.001
    gate_rel_l2: float = 0.01         # §5 criterion 3: 1% relative L2
    gate_spike_time_ms: float = 0.05  # §5 criterion 2: half of dt_out
    # §5.1: the criteria are fractional, with a bounded worst case, so that
    # the selected step does not get stricter merely because the probe set grew
    gate_pass_fraction: float = 0.90
    gate_max_count_diff: int = 1      # §5 criterion 1 worst-case guard
    gate_isi_error_cap_ms: float = 0.25   # §5 criterion 2 worst-case guard
    gate_n_lhs: int = 17              # + 7 canonical = 24 probes
    v_floor_multiplier: float = 10.0  # §3.4 numerical term
    v_floor_rms_mv: float = 0.1       # §3.4 physical term, 0.1 mV RMS
    threads_per_block: int = 256


# -------------------------------------------- characterisation (v1 §14.2)


@dataclass(frozen=True)
class Naming:
    rate_slow: float = 20.0
    rate_fast: float = 80.0
    burst_index: float = 0.3
    burst_isi_frac: float = 0.25      # v1 §7.6
    adapt: float = 0.15
    cv_irregular: float = 0.5


@dataclass(frozen=True)
class Diagnostics:
    n_isi_bins: int = 32              # v1 §7.6 log-ISI histogram (retained)
    isi_lo: float = 0.1               # ms
    isi_hi: float = 2000.0            # ms


@dataclass(frozen=True)
class Config:
    model: Model = field(default_factory=Model)
    space: Space = field(default_factory=Space)
    grid: Grid = field(default_factory=Grid)
    isi: Isi = field(default_factory=Isi)
    drive: Drive = field(default_factory=Drive)
    schedule: Schedule = field(default_factory=Schedule)
    clustering: Clustering = field(default_factory=Clustering)
    gates: Gates = field(default_factory=Gates)
    numerics: Numerics = field(default_factory=Numerics)
    naming: Naming = field(default_factory=Naming)
    diag: Diagnostics = field(default_factory=Diagnostics)

    # run-time switches (CLI)
    quick: bool = False
    sweep: bool = True
    sweep_fast: bool = False
    stage2: bool = True
    tree_reuse: bool = True
    compact_csv: bool = False
    gzip_csv: bool = False
    csv_full_trace: bool = False
    health_gate: bool = True
    outdir: str = "output"
    datadir: str = "data"
    show: bool = True

    def quickened(self) -> "Config":
        """--quick: a ~5 min smoke run."""
        return replace(
            self,
            schedule=replace(self.schedule, n_total=4096),
            clustering=replace(self.clustering, min_cluster_size=20,
                               min_samples=5, stability_subsample=2048,
                               pca_components=20),
            gates=replace(self.gates, tonic_subset=1500),
            sweep=False,
            health_gate=False,
            quick=True,
        )


DEFAULT = Config()
