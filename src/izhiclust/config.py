"""Every constant in SPEC.md, as frozen dataclasses.

Section references in comments point at SPEC.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------- model (§2)


@dataclass(frozen=True)
class Model:
    I: float = 10.0            # fixed drive, Fig. 2 of the paper
    v0: float = -70.0          # initial membrane potential
    v_peak: float = 30.0       # reset threshold / equalised peak
    # u0 = b * v0 (the paper's own initialisation) -- derived, not stored.

    @property
    def b_bifurcation(self) -> float:
        """`b` at which the rest state disappears: (5-b)^2 = 0.16 (140+I)."""
        return 5.0 - math.sqrt(0.16 * (140.0 + self.I))


# ------------------------------------------------- coefficient space (§3)


@dataclass(frozen=True)
class Space:
    a_lo: float = 0.005
    a_hi: float = 0.20        # log-uniform
    b_lo: float = -0.10
    b_hi: float = 0.35
    c_lo: float = -80.0
    c_hi: float = -40.0
    d_lo: float = 0.0
    d_hi: float = 12.0

    names: Tuple[str, str, str, str] = ("a", "b", "c", "d")

    def to_native(self, x: np.ndarray) -> np.ndarray:
        """x in [0,1]^4 (normalised domain, §3) -> (a, b, c, d)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        out = np.empty_like(x)
        out[:, 0] = self.a_lo * (self.a_hi / self.a_lo) ** x[:, 0]   # log-uniform
        out[:, 1] = self.b_lo + (self.b_hi - self.b_lo) * x[:, 1]
        out[:, 2] = self.c_lo + (self.c_hi - self.c_lo) * x[:, 2]
        out[:, 3] = self.d_lo + (self.d_hi - self.d_lo) * x[:, 3]
        return out

    def to_unit(self, p: np.ndarray) -> np.ndarray:
        """(a, b, c, d) -> x in [0,1]^4."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        out = np.empty_like(p)
        out[:, 0] = (np.log(p[:, 0]) - math.log(self.a_lo)) / (
            math.log(self.a_hi) - math.log(self.a_lo))
        out[:, 1] = (p[:, 1] - self.b_lo) / (self.b_hi - self.b_lo)
        out[:, 2] = (p[:, 2] - self.c_lo) / (self.c_hi - self.c_lo)
        out[:, 3] = (p[:, 3] - self.d_lo) / (self.d_hi - self.d_lo)
        return out


# ------------------------------------------------------------ time grid (§6)


@dataclass(frozen=True)
class Grid:
    t_total: float = 2500.0    # ms
    t_transient: float = 500.0  # ms discarded
    dt_int: float = 0.01       # ms
    L: int = 10                # integration steps per output sample
    boxcar_stages: int = 2     # §7.2 cascaded boxcar

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
        """0-based index of the first analysis-window output sample (t=500.0)."""
        return int(round(self.t_transient / self.dt_out)) - 1

    @property
    def n_out(self) -> int:
        return self.n_out_total - self.out_start - 1

    @property
    def win_start_step(self) -> int:
        """1-based integration step whose end is t = t_transient."""
        return int(round(self.t_transient / self.dt_int))

    @property
    def t_window(self) -> np.ndarray:
        return self.t_transient + self.dt_out * np.arange(self.n_out)

    @property
    def t_full(self) -> np.ndarray:
        return self.dt_out * (1.0 + np.arange(self.n_out_total))

    @property
    def window_seconds(self) -> float:
        return (self.t_total - self.t_transient) / 1000.0

    @property
    def df_hz(self) -> float:
        return 1.0 / self.window_seconds


# ------------------------------------------------- feature pipeline (§9)


@dataclass(frozen=True)
class Features:
    f_band_hi: float = 1000.0   # band-limit, Hz
    n_native: int = 40          # native Df bins, 0.5-20 Hz
    f_native_hi: float = 20.0
    n_log: int = 88             # log bands, 20-1000 Hz
    alpha: float = 0.0          # spectral tilt exponent
    f_ref: float = 10.0         # Hz, tilt reference
    # §9 step 9 (amendment A1 in §19): relative noise floor subtracted before
    # log1p, as a fraction of the sample's largest band (0.01 = -40 dB).
    noise_floor_frac: float = 0.01
    tilt_choices: Tuple[float, ...] = (0.0, 0.5, 1.0)

    @property
    def n_bands(self) -> int:
        return self.n_native + self.n_log


# ---------------------------------------------- sampling schedule (§11)


@dataclass(frozen=True)
class Schedule:
    n_first: int = 4096
    n_iter: int = 2048
    m: int = 12                 # total iterations
    gp_frac: float = 0.70
    pool: int = 20000           # Sobol candidate pool
    n_gp: int = 3000            # GP training cap
    k_pc: int = 6               # GP outputs = first 6 PCs
    w_sd: float = 0.40
    w_grad: float = 0.30
    w_entropy: float = 0.30
    knn_entropy: int = 25
    interim_subsample: int = 8000
    chunk: int = 2048           # GPU chunk size (§7.5)
    seed: int = 42

    @property
    def n_total(self) -> int:
        return self.n_first + (self.m - 1) * self.n_iter

    def split(self, n: int) -> Tuple[int, int]:
        """(n_gp_acquired, n_lhs) for a follow-on iteration."""
        n_acq = int(round(self.gp_frac * n))
        return n_acq, n - n_acq


# ---------------------------------- dimension reduction / clustering (§12)


@dataclass(frozen=True)
class Clustering:
    pca_components: int = 50
    pca_min_evr: float = 0.99
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
    stability_fits: int = 5
    stability_subsample: int = 12000
    persistence_jaccard: float = 0.5
    persistence_floor: float = 0.6
    contiguity_knn: int = 5
    contiguity_percentile: float = 95.0
    contiguity_min_frac: float = 0.05
    n_representatives: int = 3


# ----------------------------------------------------- validation (§13.1)

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


def canonical_array() -> np.ndarray:
    return np.array([[r[2], r[3], r[4], r[5]] for r in CANONICAL], dtype=np.float64)


def canonical_labels() -> list[str]:
    return [r[1] for r in CANONICAL]


# ------------------------------------------------ numerics / gates (§7.3-7.4)


@dataclass(frozen=True)
class Numerics:
    local_error_tol: float = 1e-6     # §7.3 refinement trigger
    refine_factor: int = 4            # dt_int / 4
    gate_dts: Tuple[float, ...] = (0.04, 0.02, 0.01, 0.005, 0.0025, 0.001)
    gate_dt_accept: float = 0.01
    gate_dt_ref: float = 0.001
    gate_rel_l2: float = 0.01         # 1% relative L2
    gate_margin: float = 0.25         # select dt at 4x margin (§19 A2)
    gate_alias_ratio: float = 5.0     # filtered must beat naive by >= 5x (§19 A3)
    gate_fine_out_divisor: int = 8    # reference output grid for the alias test
    gate_n_lhs: int = 17              # + 7 canonical = 24 probes
    p_floor_multiplier: float = 10.0  # §10
    # §19 A5: P_floor also has a physical term -- the ac_power of an in-band
    # fluctuation of this RMS amplitude (mV).  10x the numerical floor is only
    # ~1 part in 1e10 of a spiking sample's ac_power, which let pure roundoff
    # be retained and named "subthreshold/resonant".
    p_floor_rms_mv: float = 0.1
    threads_per_block: int = 256


# -------------------------------------------- characterisation (§14.2)


@dataclass(frozen=True)
class Naming:
    rate_floor: float = 4.0           # Hz, spectral-estimate floor (§6)
    rate_slow: float = 20.0
    rate_fast: float = 80.0
    burst_index: float = 0.3
    burst_isi_frac: float = 0.25      # §7.6
    adapt: float = 0.15
    cv_irregular: float = 0.5
    corr_disagreement: float = 0.2    # §14.1 |dr| flag
    min_lhs_for_unbiased: int = 30
    near_floor_factor: float = 100.0  # §19 A5: NEAR-FLOOR diagnostic flag


@dataclass(frozen=True)
class Diagnostics:
    n_isi_bins: int = 32              # §7.6 log-ISI histogram
    isi_lo: float = 0.1               # ms
    isi_hi: float = 2000.0            # ms


@dataclass(frozen=True)
class Config:
    model: Model = field(default_factory=Model)
    space: Space = field(default_factory=Space)
    grid: Grid = field(default_factory=Grid)
    features: Features = field(default_factory=Features)
    schedule: Schedule = field(default_factory=Schedule)
    clustering: Clustering = field(default_factory=Clustering)
    numerics: Numerics = field(default_factory=Numerics)
    naming: Naming = field(default_factory=Naming)
    diag: Diagnostics = field(default_factory=Diagnostics)

    # run-time switches (CLI, §15/§16)
    quick: bool = False
    sweep: bool = True
    sweep_fast: bool = False
    stage2: bool = True
    tree_reuse: bool = True
    compact_csv: bool = False
    gzip_csv: bool = False
    csv_full_trace: bool = False
    ipw_stats: bool = False
    outdir: str = "output"
    datadir: str = "data"
    show: bool = True

    def quickened(self) -> "Config":
        """--quick: n1=1024, n=512, m=6, --no-sweep (§15)."""
        return replace(
            self,
            schedule=replace(self.schedule, n_first=1024, n_iter=512, m=6,
                             interim_subsample=4000, n_gp=1500, pool=8000),
            clustering=replace(self.clustering, min_cluster_size=20, min_samples=5,
                               stability_subsample=2048),
            sweep=False,
            quick=True,
        )


DEFAULT = Config()
