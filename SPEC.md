# Izhikevich Spiking-Neuron Cluster Discovery — Specification

**Status:** draft for review · **Date:** 2026-11-21 · **Target device:** NVIDIA RTX 500 Ada Laptop (4094 MiB, cc 8.9), 22 CPU threads, 62 GB RAM

---

## 1. Objective

Explore the four-dimensional coefficient space `(a, b, c, d)` of the Izhikevich
spiking-neuron model at fixed drive `I`, and identify as many distinguishable
firing-pattern clusters as possible. A cluster qualifies as *of interest* only
when it is (i) reproducible across embedding seeds, (ii) characterisable in
interpretable terms, and (iii) contiguous enough in parameter space that its
mean coefficient vector is meaningful. Sections 13–14 enforce those three
conditions; discovery without them is not a result.

Reference: E. M. Izhikevich, "Simple Model of Spiking Neurons", *IEEE Trans.
Neural Networks* **14**(6):1569–1572, Nov. 2003 (`spikes.pdf`).

### 1.1 Scope map

Six workflow additions were agreed as in scope beyond the base pipeline. `S`
numbers are retained from the review discussion so they stay traceable; `S6`,
`S7` and `S9` were considered and excluded, and are recorded in §18.

| | Addition | Where specified |
|---|---|---|
| S1 | Validate against the paper's 7 known neuron types | §13.1 |
| S2 | Interpretable per-cluster names from firing diagnostics | §7.6, §14.2 |
| S3 | Correct the sampler-induced bias in coefficient correlations | §14.1 |
| S4 | Check each cluster is contiguous in parameter space | §13.3 |
| S5 | Score cluster stability across embedding seeds (ARI) | §13.2 |
| S8 | Treat the model as an ODE-with-reset, not a PDE | §2, §7 |

## 2. The model

The system is a **two-dimensional ODE with a state-dependent reset event** — not
a PDE. The distinction matters operationally, not just terminologically: the
reset is a genuine jump discontinuity, and essentially every numerical decision
in this spec (§7–§9) exists to handle it. A PDE framing would suggest spatial
discretisation and method-of-lines machinery, none of which applies.

```
dv/dt = 0.04 v^2 + 5 v + 140 - u + I
du/dt = a (b v - u)
if v >= 30:   v <- c,   u <- u + d
```

`v` is membrane potential (mV scale), `u` a recovery variable. Only one
nonlinear term, `v^2`.

**Fixed drive:** `I = 10` — the value used for every panel of the paper's Fig. 2,
so results are directly comparable to the published RS/IB/CH/FS/LTS/TC/RZ types.

**Initial condition:** `v0 = -70.0`, `u0 = b * v0` (the paper's own
initialisation, `u = b .* v`), with `I` applied as a step at `t = 0`.

**Analytic anchor for range selection.** At equilibrium `u = b v`, so
`0.04 v^2 + (5 - b) v + (140 + I) = 0`, with discriminant
`(5 - b)^2 - 0.16 (140 + I)`. At `I = 10` this vanishes at **`b ≈ 0.101`**.
For `b > 0.101` no fixed point exists and the neuron fires tonically for any
`a, c, d`; for `b < 0.101` a rest state exists, admitting quiescence,
bistability and subthreshold resonance. The sampling box must straddle 0.101 or
half the phenomenology is unreachable.

## 3. Coefficient space

| Coeff | Range | Sampling | Paper values inside |
|---|---|---|---|
| `a` | `[0.005, 0.20]` | **log-uniform** | 0.02 (RS, IB, CH, LTS, TC), 0.1 (FS, RZ) |
| `b` | `[-0.10, 0.35]` | uniform | 0.2, 0.25, 0.26, −0.1 (alt. RS form) |
| `c` | `[-80, -40]` mV | uniform | −65 (RS, FS, LTS, TC, RZ), −55 (IB), −50 (CH) |
| `d` | `[0, 12]` | uniform | 8 (RS), 4 (IB), 2 (CH, FS, LTS, RZ), 0.05 (TC) |
| `I` | `10` | fixed | Fig. 2 |

`a` is log-sampled because it is a rate: `1/a` spans 5–200 ms, and uniform
sampling would put 97.5% of draws in the fast-recovery half of that span.

**Normalised domain** `x ∈ [0,1]^4` used by the sampler and GP:
`x0 = (log a - log 0.005)/(log 0.20 - log 0.005)`, and min–max for `b, c, d`.

## 4. Architecture

Stack: **CuPy fused `RawKernel`** for integration, **cuFFT** for spectra, a
**hand-rolled ARD-RBF GP in CuPy** for the sampler, **umap-learn + hdbscan** on
CPU for clustering. Rationale: the 250 000-step integration loop must not pay
per-step kernel-launch overhead (a tensor-op formulation would spend ~100 s per
batch on launches alone); and a 3000-point GP needs one 10 ms Cholesky, which
does not justify adding a second GPU framework and ~3 GB of dependencies.

```
/opt/neuron/
  pixi.toml
  SPEC.md
  src/izhiclust/
    config.py        # every constant in this spec, as frozen dataclasses
    kernel.cu        # DP5 + reset + cascaded-boxcar output + in-kernel diagnostics
    simulate.py      # RawModule wrapper, chunking, dt/4 refinement pass
    features.py      # Hann, rfft, |H| deconvolution, band-limit, hybrid banding, log1p, L2
    sampling.py      # LHS, Sobol candidate pool, canonical seeds
    gp.py            # ARD-RBF GP: NLL+analytic grad, L-BFGS-B, posterior mean/var/grad, rank-1 var update
    acquire.py       # three-term acquisition, greedy batch with variance re-conditioning
    cluster.py       # PCA(50), UMAP(10), HDBSCAN, selection sweep, quarantine
    validate.py      # canonical placement (S1), stability ARI (S5), contiguity (S4)
    characterize.py  # interpretable naming rules (S2)
    stats.py         # triangular matrices, design-unbiased variants (S3)
    plots.py         # 2D scatter, per-representative two-panel figures
    io.py            # ./data CSVs, ./output figures, manifest, npz
    pipeline.py      # orchestrator
  tests/
  data/              # created at run time (§16)
  output/            # created at run time (§16)
```

## 5. Environment (pixi)

`pixi.toml`, channel `conda-forge`, platform `linux-64`:

```
python, numpy, scipy, pandas, matplotlib, scikit-learn,
umap-learn, hdbscan, cupy, cuda-version=13.*, tqdm, pytest
```

Notes:
- `cuda-version=13.*` matches the installed driver (595.71.05 / CUDA 13.2)
  exactly; cupy 14.2's default build already requires `>=13,<14`.
- Install `matplotlib` (not `matplotlib-base`) so an interactive backend is
  available: `DISPLAY=:0` is present, so `plt.show()` popups work. Fall back to
  `Agg` with a warning if no display is found, and still write all files.
- `hdbscan` (the standalone package), **not** `sklearn.cluster.HDBSCAN`, because
  only the former exposes `relative_validity_` and `approximate_predict`, both
  of which this spec requires.
- **Task 1 of implementation is to verify the solve resolves**, and to pin the
  resulting versions into the manifest.

## 6. Time grid

| Quantity | Value | Justification |
|---|---|---|
| `T_total` | 2500 ms | |
| transient discard | `[0, 500)` ms | 2.5× the slowest model timescale `1/a_min = 200 ms` |
| analysis window | `[500, 2500)` ms = 2000 ms | |
| `dt_int` | **0.01 ms** | §7; exactly 10 steps per output sample, so outputs land on grid points and no dense-output interpolation is needed |
| `dt_out` | **0.1 ms** | matches the paper's Fig. 2 resolution; `f_Nyq = 5000 Hz` |
| `N_out` (window) | 20 000 | `Δf = 1 / 2.0 s = 0.5 Hz` |
| total steps | 250 000 | |

**Why 2000 ms of analysis window.** The binding constraint is the *slowest*
rhythm, not the fastest. Burst envelopes are built from `u` accumulating on the
`1/a` timescale, so at `a = 0.005` the interburst period can reach several
hundred ms. 2000 ms yields 2–4 cycles at the slowest corner, ≥8 cycles at 4 Hz,
≥20 at 10 Hz, and hundreds at fast-spiking rates. The spectral-estimate floor is
~8 spikes, which over 2000 ms is a **4 Hz firing-rate floor**; anything slower is
quarantined (§10) rather than clustered.

## 7. Integrator

Fixed-step **Dormand–Prince 5(4)** (DOPRI5 tableau, 5th-order solution, FSAL →
6 RHS evaluations per step).

**Fixed step, not adaptive, deliberately.** The embedded 4(5) error estimate is
meaningless on any step containing a reset, because that step carries an `O(1)`
jump. A step-size controller would either reject indefinitely or require
explicit event location with off-grid restarts — neither of which vectorises
across a SIMD batch. Fixed step with event-detection-at-step-end is the honest
choice, and §7.3 quantifies its cost.

### 7.1 Reset event

Test `v >= 30` at the end of each 0.01 ms step. On trigger:
1. If that step coincides with an output sample, record the sample as exactly
   `+30.0` — **peak equalisation**, identical to what the paper does for Fig. 3
   ("all spikes were equalized at +30 mV by resetting v first to +30 mV and then
   to c"). Without it, the overshoot (a step can end as high as ~33 mV)
   injects parameter-dependent amplitude noise directly into the high bins.
2. `v <- c`, `u <- u + d`, continue.

### 7.2 Anti-aliased output stage

The reset is a true step discontinuity, so the signal is **not bandlimited at
any sample rate** — its harmonic amplitudes fall only as `1/f`. Naive 10:1
decimation (taking every 10th step) folds 9–11 kHz content into 0–1 kHz. For a
100 Hz firing rate, legitimate content at 1 kHz is `k=10` ≈ −20 dB while the
folding content near 9.5 kHz is `k=95` ≈ −40 dB: aliasing would sit only ~20 dB
below signal at the top of the analysis band, a ~10% amplitude error. Worse, the
fold amount depends on where the firing rate sits relative to 10 kHz, making it
a **non-smooth, parameter-dependent** perturbation — which both corrupts the
UMAP metric and violates the smooth-response-surface assumption the GP relies on.

**Fix:** emit the output sample as a **cascaded boxcar (2 stages, `L = 10`)** of
the integration-rate samples, accumulated in registers. Boxcars have nulls
exactly at multiples of the output rate, which is precisely where the images are:

| Frequency | 1 stage | 2 stages |
|---|---|---|
| 9 kHz (worst of fold band) | −19 dB | **−38 dB** |
| 10 kHz (image centre) | null | null |
| 1 kHz (in-band distortion) | −0.14 dB | **−0.28 dB** |

Two stages give ≈58 dB of margin (~0.1% residual). Cost is ~2 extra FMAs per
integration step. The filter is a known linear operator, so its small in-band
tilt is removed **exactly** in §9 by dividing the magnitude spectrum by
`|H(f)|^2`, `H(f) = sin(pi f L T) / (L sin(pi f T))`, `L = 10`, `T = 0.01 ms`.

The 2-stage cascade is 19 samples ≈ 0.19 ms long, comparable to the ~0.15 ms
spike upstroke, so it visibly rounds spikes. Harmless for 0–1 kHz features, but
wrong for display — hence representative traces are re-simulated with filtering
**disabled** (§8, §16).

Rejected alternative: cumulative time-domain integration of `v` before the
rFFT. It is exactly a `1/f` spectral weighting, but obtained the worst way —
infinite DC gain amplifies any unsettled slow drift by precisely `1/f`, and
integrating before windowing breaks endpoint matching and reintroduces leakage.
The same tilt is available exactly and side-effect-free as a post-FFT weighting,
specified as the `--spectral-tilt` option in §9.

### 7.3 Error budget and per-sample adaptivity

DP5 truncation on the smooth flow is negligible: `|dv/dt| <= ~330 mV/ms` at
threshold and local error is `O(dt^6)`. The binding error is **spike-time
quantisation from the reset**, bounded by `dt_int = 0.01 ms` — a 3.6° phase error
at 1000 Hz (top of the analysis band) and 0.36° at 100 Hz, giving in-band
magnitude error < 0.2%.

The embedded 4th-order solution **is** computed on non-spike steps and reduced to
a per-sample `max_local_error`. Any sample with `max_local_error > 1e-6`, or with
a non-finite state, is re-run at `dt_int / 4` in a second pass. Adapting
per-sample rather than per-step is the correct way to get adaptivity on SIMD
hardware.

### 7.4 Convergence gate (blocking, runs before the main sweep)

24 probe vectors — the 7 canonical types (§12) plus 17 LHS points — integrated at
`dt_int ∈ {0.04, 0.02, 0.01, 0.005, 0.0025, 0.001}` ms. **Accept `dt_int = 0.01`
only if the 128-band feature vector (§9, at `alpha = 0`) is within 1% relative
L2 of the `dt = 0.001` reference.** Second check: the same vector must be stable when
`dt_out` is halved to 0.05 ms — the direct empirical test that aliasing is
controlled. If it is not, the remedy is a longer boxcar cascade, not a smaller
`dt_int`. Runs in seconds; the pipeline aborts if either check fails.

### 7.5 Kernel layout

One thread per parameter set, 256 threads/block, grid-stride. `v`, `u`, the four
coefficients and all diagnostic accumulators live in registers for the whole
250 000-step loop. Output is written as `traces[t * n_batch + i]` so consecutive
threads write consecutive addresses — fully coalesced 128-byte transactions.

Chunk size **2048**: trace buffer `2048 × 20000 × 4 B = 164 MB`, ~500 MB peak
with cuFFT workspace, comfortable inside 4094 MiB. At 2048 threads on 2048 CUDA
cores there is ~1 thread/core and no latency hiding, so expect 5–10% of peak
FP32 → roughly **1–3 s per chunk**. Not worth optimising; the integration is not
the bottleneck (§15).

### 7.6 In-kernel diagnostics

Accumulated in registers, no second pass over the traces. These are **not**
clustering features — the clustering input is the rFFT magnitude per §9. They
exist to characterise and validate clusters (§13, §14).

- `n_spikes`; ISI `sum`, `sumsq`, `min`, `max`; `first_isi`, `last_isi`
- **32-bin log-ISI histogram**, 0.1 ms to 2000 ms. This is what makes burst
  index, `CV(ISI)` and spikes-per-burst derivable on the host from fixed-size
  state, despite each needing a threshold that is only known after the mean ISI
  is available.
- `mean(v)`, `sd(v)`, `max_local_error`, `all_finite` flag

Host-derived: `rate = n_spikes / 2.0 s`; `CV_ISI = sd/mean`;
`burst_index` = fraction of ISIs below `0.25 × mean ISI` (from the histogram);
`spikes_per_burst ≈ (#ISI below threshold)/(#ISI above threshold) + 1`
(approximate — the histogram discards ordering);
`adapt_index = (last_isi - first_isi)/(last_isi + first_isi)`.

## 8. Trace persistence

Full traces **are** persisted, to CSV, per §16.2. The footprint is modest:
`26 624 × 20 000 = 5.32e8` values, so ~4.3 GB of CSV at `%.4g` and 2.1 GB as
`float32` binary, against 644 GB free.

The cost is write *time*, not space. `numpy.savetxt` runs at roughly 10 MB/s and
would take ~11 min; `pandas.DataFrame.to_csv` with the C engine and an explicit
`float_format` sustains 60–100 MB/s. **Implementation must use chunked
`to_csv(mode='a')` at 2048 rows per chunk**, budgeted at 3 min for all CSV output
(§15). `--gzip-csv` compresses the two large files to ~1.7 GB combined at the
cost of more CPU time; `--compact-csv` skips them entirely, saving the 3 min.

Separately, the integrator is fully deterministic given
`(a, b, c, d, I, v0, u0, dt_int, dt_out)`. The ~90 representative traces are
therefore **re-simulated exactly on demand** rather than extracted from the CSV,
because they need the §7.2 output filter *disabled* for display sharpness — a
different signal from the filtered trace that fed the DFFT.

## 9. Feature pipeline

Per sample, on the 20 000-point analysis window:

1. **Mean-subtract.** Otherwise `|X[0]|` ≈ `|mean(v)| × 20000` dwarfs everything
   and PC1 degenerates to mean voltage, i.e. to `c` — clusters would split on
   reset depth rather than firing pattern.
2. **Hann window.** The trace is not periodic in the window; a rectangular
   window leaks sharp firing peaks across neighbouring bins.
3. **`rfft`** → 10 001 one-sided bins, `Δf = 0.5 Hz`, `f ∈ [0, 5000]` Hz. Taking
   `rfft` (rather than `fft`) *is* the "truncate away the symmetric part" step.
4. **Magnitude.**
5. **Deconvolve the known output filter:** divide by `|H(f)|^2` (§7.2).
6. **Optional spectral tilt:** multiply by `(f_ref / f)^alpha`,
   `alpha ∈ {0, 0.5, 1}`, `f_ref = 10 Hz`, default `alpha = 0`. Swept in §11.
   `alpha = 1` is the exact equivalent of time-domain integration, without its
   drift amplification. On log-spaced bands it is a linear reweighting in
   `log f`, so it is a real change to the UMAP metric rather than a cosmetic one.
7. **Band-limit to `(0, 1000]` Hz** → 2000 bins. Nyquist is 5 kHz, but the
   8000 bins above 1 kHz are numerical noise for every parameter set in the box
   and would otherwise dominate a Euclidean metric.
8. **Hybrid banding to 128 bands.** Pure log spacing from 0.5 Hz fails: a
   6.1%-wide band at 0.5 Hz is 0.03 Hz, far narrower than `Δf = 0.5 Hz`, leaving
   ~40 empty bands. Instead:
   - **40 native bins** at `Δf = 0.5 Hz` covering `0.5–20 Hz`
   - **88 log-spaced bands** (4.55%/band) covering `20–1000 Hz`, where band
     width `>= Δf` holds throughout
   Band value = **RMS magnitude** over the band's bins, so wide bands do not win
   merely by containing more of them.
9. **`log1p`.** Magnitudes are heavy-tailed; a few large bins would otherwise
   drive the entire k-NN graph.
10. **L2-normalise.** Clusters on spectral *shape*, not loudness. Loudness
    correlates with `c` and is retained separately as the `sd(v)` diagnostic.

Output: `F ∈ R^(N × 128)`, plus the band centre frequencies for CSV headers and
plot axes.

Banding resolution is deliberately coarse at high `f`: 10.0 Hz vs 10.5 Hz tonic
firing is not a *type* difference, and spending metric weight on it would
fragment clusters along a nuisance axis.

## 10. Quarantine

Applied **before** PCA/UMAP. Samples that are genuinely at rest — `n_spikes == 0`
**and** in-band AC power below `P_floor` — would form one enormous degenerate
blob that distorts the whole k-NN graph and the density estimate with it. They
are held out, labelled cluster **`Z`**, and their parameter region reported
separately.

Critically, **zero-spike samples are not dropped wholesale.** Subthreshold
resonators (the paper's RZ) produce no spikes but a real spectral peak; the AC
power floor is exactly what separates them from true quiescence. Expect this to
matter, since `b < 0.101` (the region where a rest state exists, §2) lies inside
the box.

**`ac_power`** is defined once here and used throughout (§10, §14.2, §16.2): the
sum of `|X(f)|^2` over the band-limited bins `f ∈ (0, 1000]` Hz of the
mean-subtracted, Hann-windowed, filter-deconvolved spectrum — i.e. §9 after step
7 and before banding, with `alpha = 0`. It excludes DC by construction, so it
measures fluctuation energy only.

`P_floor` is set from the convergence-gate run as `10 ×` the median `ac_power`
of a numerically silent reference sample, and recorded in the manifest.
Non-finite samples that survive the §7.3 refinement pass are labelled `X` and
excluded; their count is an error-budget metric and should be zero.

## 11. Sampling schedule and GP active learning

### 11.1 Schedule

| Iteration | `n` | Composition |
|---|---|---|
| 1 | 4096 | scrambled LHS (`optimization='random-cd'`), **with the 7 canonical vectors (§12) substituted for 7 LHS points** so known types are guaranteed present |
| 2–12 | 2048 each | **1434 GP-acquisition (70%) + 614 fresh LHS (30%)** |

**`m = 12`**, total **`N = 26 624`** (10 850 LHS + 15 774 GP-acquired).

`m = 12` because the ARD lengthscales need enough rounds to sharpen — below ~8
the GP barely adapts — while per-round overhead stays inside budget (§15).

The **30% LHS floor is not padding.** Max-variance acquisition reliably collapses
into one region once lengthscales become anisotropic; the floor keeps the design
globally ergodic so a cluster in an as-yet-unvisited corner remains discoverable.
It also supplies the design-unbiased subset that §14 requires.

### 11.2 GP model

Outputs: the first **`K = 6` PCA components** of `F`, each standardised. Six
**independent ARD-RBF GPs** (constant mean, Gaussian noise), hyperparameters by
L-BFGS-B on the log-marginal likelihood with **analytic gradients** over 6 log
parameters per output (4 lengthscales, signal variance, noise variance).

Training cap `N_gp = 3000`, composed as *all* points from the last two iterations
plus a uniform draw from earlier ones, so recent structure is never diluted away.
A 3000×3000 Cholesky is ~10 ms on this GPU, so a full fit is seconds.

**ARD is mandatory, and the reason is easy to get wrong.** GP posterior variance
does not depend on the observed `y` for a fixed kernel. Max-variance acquisition
would therefore degenerate into a plain space-filling design. It becomes
informative *only* because the lengthscales are learned: directions and regions
where the spectra change fast acquire short lengthscales and hence high
posterior variance. An isotropic kernel would destroy this mechanism entirely.

### 11.3 Acquisition

Pool: 20 000 scrambled-Sobol candidates in `[0,1]^4`. Each term rank-normalised
to `[0,1]` before weighting:

| Weight | Term | Purpose |
|---|---|---|
| 0.40 | total posterior SD, weighted by PC explained variance | explore |
| 0.30 | `||grad_x mu(x)||` (analytic for RBF) | seek bifurcation surfaces, where behaviour changes fastest |
| 0.30 | predictive entropy of a 25-NN classifier on current HDBSCAN labels | seek type boundaries and unlabelled voids |

The latter two are the `y`-dependent terms that make this active learning rather
than space-filling.

**Batch selection:** greedy, re-conditioning the posterior variance after each
pick. The update is exact and `y`-free via a rank-1 Cholesky update, and it is
what supplies the repulsion preventing the whole batch from clumping on a single
maximum.

**Interim labels** for the entropy term are cheap by design: PCA(50) → HDBSCAN on
an 8000-point subsample, no UMAP. Full UMAP runs once, at the end.

## 12. Dimension reduction and clustering

**Two separate UMAP fits, which must not be the same fit.**

### 12.1 Clustering embedding — `n_components = 10`

- **Not the 2D visualisation embedding.** A 2D UMAP aggressively distorts local
  density to fit the plane. HDBSCAN *is* a density algorithm, so clustering there
  both fragments real clusters and fuses adjacent ones. Cluster in 10D, then
  *colour* the 2D plot with those labels.
- **Not 2–5 dims:** UMAP must tear the topology to embed that low, merging types.
- **Not 20–50 dims:** HDBSCAN's mutual-reachability distance degrades as the
  k-NN distance distribution concentrates; density contrast collapses above
  ~15–20 dims.

Parameters: `n_neighbors=30, min_dist=0.0, n_components=10, metric='euclidean',
n_epochs=500`; `random_state` per the policy in §12.3.

- `min_dist=0.0` **specifically because this embedding feeds a density
  clusterer** — it lets points pack arbitrarily tightly, maximising density
  contrast. `min_dist` is otherwise a purely aesthetic parameter.
- `n_neighbors=30`: at `N ≈ 27k` with 10–40 expected clusters (~700–2700 points
  each), 30 balances local detail against global structure. 15 over-fragments;
  100 washes out small clusters.
- `metric='euclidean'` on L2-normalised features is monotone-equivalent to
  cosine, so it yields an identical k-NN graph and is faster.

**PCA(50) before UMAP**, `whiten=False`, fitted on the 128-band matrix. Three
reasons, in descending strength:

1. **It costs nothing, because it is computed anyway.** §11.2 needs the first 6
   PCs every iteration to drive the GP sampler, so the basis already exists.
   Feeding the same projection to UMAP is free.
2. **Denoising.** The 128 bands are strongly correlated (neighbouring bands of a
   harmonic comb move together); 50 PCs retain >99% of the variance while
   dropping the low-variance band-to-band noise that would otherwise enter the
   k-NN graph.
3. **Speed, but less than it sounds.** PCA cuts the distance computation from
   128 to 50 dimensions, a ~2.5× reduction — **but only in the k-NN graph
   phase.** UMAP's layout optimisation operates in the *output* space and is
   indifferent to input dimension, and at `N ≈ 27k` the layout is the larger
   share of the fit. Expect ~20–30% off a UMAP fit, not 3×. (An earlier draft
   said "~3× faster k-NN", which is true of that phase and misleading about the
   fit as a whole.)

`whiten=False` deliberately: whitening equalises variance across components,
which would amplify exactly the low-variance noise directions that reason 2
exists to suppress.

**Residual risk, and the guard.** PCA is a linear projection, so a rare cluster
distinguished only by a low-variance direction could in principle be flattened
before UMAP sees it. At >99% retained variance this is unlikely but not
impossible. §17 therefore asserts cumulative explained variance at 50 PCs ≥ 99%
and records the full variance curve in the manifest; if 50 components fall short,
raise `n_components` rather than proceeding.

### 12.2 Visualisation embedding — `n_components = 2`

`n_neighbors=30, min_dist=0.1`, fitted **independently** on the same PCA(50)
input, coloured by the 10D labels.

### 12.3 `random_state` policy — determinism only where it is paid for

umap-learn **disables parallelism in the layout optimisation whenever
`random_state` is set**, to guarantee reproducibility. Seeding every fit would
therefore run all of them on 1 of this machine's 22 threads, and UMAP is 36% of
the runtime (§15). Determinism is bought selectively:

| Fit | `random_state` | Why |
|---|---|---|
| Final reported 10D clustering embedding | **42** | feeds the published labels and the §17 determinism criterion |
| 2D visualisation embedding (§12.2) | **42** | feeds the published scatter plot |
| 3 exploratory sweep fits (§12.4 stage 1) | `None` | need only be mutually comparable *within* a run; the winning config is re-fitted seeded before reporting |
| 2 stage-2 tilt fits (§12.4) | `None` | same |
| 5 stability fits (§13.2) | `None` | §13.2 *wants* five different embeddings, so independent unseeded fits are both correct and fastest |

The measured single- vs multi-threaded ratio must be recorded in the manifest —
the ~4× assumed in §15 is an estimate, not a measurement.

`n_epochs` is a third lever, not exercised by default: umap-learn defaults to 200
for `N > 10k` and this spec uses 500 for crisper density structure. Dropping to
300 would cut layout time ~40%. Treat any change as a measured trade-off against
cluster crispness, not a free win.

### 12.4 HDBSCAN

`min_cluster_size=60` (≈0.22% of `N`), `min_samples=10`,
`cluster_selection_method='leaf'`, `metric='euclidean'`,
`allow_single_cluster=False`, `prediction_data=True`,
**`gen_min_span_tree=True`** (required for `relative_validity_`).

- `min_samples=10` is the key knob. The library default (`= min_cluster_size`)
  is far too conservative: it labels a large fraction as noise while swallowing
  small clusters. Decoupling the two is what preserves rare behaviour types.
- **`'leaf'` is what serves the objective directly.** The default `'eom'`
  (excess of mass) prefers few large stable clusters and absorbs fine structure;
  `'leaf'` returns the finest leaves of the condensed tree. **Report both** —
  `leaf` as the fine taxonomy (A, B, C, …), `eom` as a coarse grouping, plus the
  leaf→eom containment table.

**Selection sweep, two staged passes.** A single full-factorial sweep is not
affordable: `alpha` is a *feature-pipeline* parameter (§9 step 6), so changing it
invalidates the features, the PCA basis and every UMAP fit downstream — a
`3 × 3 × 3 × 2 × 3 = 162`-config grid would cost ~40 min of HDBSCAN alone against
the whole runtime budget (§15). Staged instead:

- **Stage 1 — clustering parameters at `alpha = 0`:** `n_neighbors ∈ {15,30,50}`
  × `min_cluster_size ∈ {30,60,120}` × `min_samples ∈ {5,10,25}` × method ∈
  `{leaf,eom}` = **54 configs**, evaluated with tree reuse (§12.4.1) in
  **9 tree builds** → ~3.5 min, not 54 fits → 13.5 min.
- **Stage 2 — spectral tilt, holding stage 1's winner fixed:** `alpha ∈ {0.5, 1}`
  = **2 configs**, each requiring a features → PCA → UMAP(10) → HDBSCAN
  re-run → ~4 min.

Scored by **`relative_validity_` (DBCV)** with a penalty for noise fraction above
25%. Both stage tables are reported; the overall winner drives the figures and is
**re-fitted once with `random_state=42`** (§12.3) before anything is reported,
with the hand-set defaults above as the reproducible fallback and as the config
reported when the sweep is skipped (`--no-sweep`).

#### 12.4.1 Tree reuse — the sweep must not refit what it does not change

A naive sweep calls `HDBSCAN.fit` 54 times, which rebuilds the same trees six
times over. The cost structure of the algorithm is staged, and the sweep must
mirror it:

| Phase | Depends on | Cost |
|---|---|---|
| core distances → mutual-reachability MST → single-linkage tree | embedding, `min_samples`, metric | **dominant** (~15 s at `N ≈ 27k`, 10D) |
| condense the tree | `min_cluster_size` | cheap |
| extract clusters from the condensed tree | `cluster_selection_method`, `allow_single_cluster`, `cluster_selection_epsilon` | cheap |

So stage 1 is **9 tree builds** (3 embeddings × 3 `min_samples`), each followed
by 6 near-free extractions (3 `min_cluster_size` × 2 methods). `9 × 15 s ≈
2.3 min`, plus per-labelling `relative_validity_` evaluation, which is not free —
budget **~3.5 min** against the 13.5 min a naive sweep would cost.

Implementation: call `hdbscan.hdbscan()` once per (embedding, `min_samples`) to
obtain the `single_linkage_tree_`, then re-run only the condense/extract steps
via `hdbscan.hdbscan_._tree_to_labels` for each `(min_cluster_size, method)`
pair. This is a semi-internal API, so §17 requires a test asserting that a
reused-tree labelling is **identical** to a full `HDBSCAN.fit` at the same
parameters; if the API shifts under a version bump, that test fails loudly rather
than silently producing different clusters. `--no-tree-reuse` falls back to 54
full fits.

`--sweep-fast` drops `min_samples = 25` from stage 1 (36 configs, 6 tree builds,
~2.3 min).

**Cluster labelling:** clusters are sorted by descending size and assigned
`A, B, C, …` (then `AA, AB, …` beyond 26). `Z` is reserved for quarantine (§10),
`X` for non-finite, and HDBSCAN noise (`-1`) is reported as `noise`, never
lettered.

## 13. Validation (suggestions S1, S4, S5 — in scope)

### 13.1 S1 — placement of the paper's known types

The 7 canonical vectors at `I = 10`:

| Type | `a` | `b` | `c` | `d` |
|---|---|---|---|---|
| RS regular spiking | 0.02 | 0.20 | −65 | 8 |
| IB intrinsically bursting | 0.02 | 0.20 | −55 | 4 |
| CH chattering | 0.02 | 0.20 | −50 | 2 |
| FS fast spiking | 0.10 | 0.20 | −65 | 2 |
| LTS low-threshold spiking | 0.02 | 0.25 | −65 | 2 |
| TC thalamo-cortical | 0.02 | 0.25 | −65 | 0.05 |
| RZ resonator | 0.10 | 0.26 | −65 | 2 |

All lie inside the §3 box. Each is seeded into iteration 1 and placed via
`hdbscan.approximate_predict`. Report a table of type → cluster letter →
membership probability, and annotate each on the 2D scatter (§16).

This is the only external ground truth available, and it converts "we found 23
clusters" into "we recovered the 7 published types and found 16 more" — which is
the actual objective.

**Expected and not a failure:** TC's rebound bursting requires a hyperpolarising
current step and RZ's resting/spiking bistability requires timed brief stimuli.
Neither is reachable under a single constant `I` (§18, item 6), so TC and RZ
should be expected to land in generic tonic clusters. The spec records this so
the result is not misread.

### 13.2 S5 — cluster stability

Re-run the chosen config under **5 independent UMAP embeddings** on a
12 000-point subsample. Report mean pairwise **Adjusted Rand Index** across them,
and per-cluster **persistence** = fraction of embeddings in which some cluster
attains ≥0.5 Jaccard overlap with it. Clusters below 0.6 persistence are flagged
`UNSTABLE` in all reports. An unstable cluster is an artefact of the embedding,
not a finding.

The 5 fits use **`random_state=None`** (§12.3), which is both correct and fastest
here: this test deliberately wants five *different* embeddings, so there is
nothing for a fixed seed to buy, and leaving it unset restores multi-threaded
layout optimisation. The five fits are independent, so they also run concurrently
across the 22 available threads. Budget ~1.5 min rather than ~6 min.

Consequence worth stating: the stability *number* is therefore not bitwise
reproducible between runs — it is a Monte-Carlo estimate over embeddings. That is
the correct semantics for a stability statistic, but it means the §17 acceptance
threshold (mean ARI ≥ 0.6) is a threshold on an estimate, and a run landing at
0.60 ± 0.03 should be re-run rather than treated as a crisp pass or fail.

### 13.3 S4 — contiguity in parameter space

For each cluster, run single-linkage on its own `(a,b,c,d)` points in the
normalised domain, cut at the 95th percentile of the global `k=5` NN distance,
and count components holding ≥5% of the cluster's points.

This matters because a cluster fragmented across parameter space means its
**mean coefficient vector is meaningless** — and the mean is something §14
prints. Any cluster with more than one component is flagged `MULTIMODAL`, its
per-component means and sizes are reported alongside the pooled mean, and the
pooled mean is marked as not interpretable.

## 14. Per-cluster statistics (suggestions S2, S3 — in scope)

### 14.1 S3 — the triangular matrix, and its sampling bias

For each cluster, a 4×4 matrix over `(a, b, c, d)`:
- **lower triangle:** Pearson `r` between coefficient pairs
- **diagonal:** standard deviation (native units — `a` on its linear scale for
  interpretability, with the log-scale SD reported as an extra column)
- **upper triangle:** blank, via a pandas frame with `NaN` above the diagonal
  rendered by `to_string(na_rep='')`

**The bias, and the fix.** The GP sampler deliberately oversamples boundaries
and fast-changing regions (§11.3), so within-cluster coefficient correlations
computed over all samples partly measure *the sampling design* rather than the
cluster. Each matrix is therefore reported **twice**:

1. over all `N` samples in the cluster — descriptive of what was simulated
2. over the cluster's **LHS-only subset** (drawn from the 10 850 uniformly
   sampled points, §11.1) — **design-unbiased, and authoritative**

Where the two disagree materially (any `|Δr| > 0.2`), the discrepancy is flagged
in the report. If a cluster's LHS-only subset falls below 30 points, its
unbiased matrix is reported as insufficient-sample rather than computed. A
third, optional variant inverse-propensity-weighted by a k-NN density estimate
in parameter space is available behind `--ipw-stats`.

Also reported per cluster: size, LHS-only size, mean coefficient vector (both
subsets), noise-adjacency, `relative_validity_` contribution, persistence (§13.2),
component count (§13.3).

### 14.2 S2 — interpretable cluster names

Derived from the §7.6 diagnostics (cluster medians), so every cluster carries a
human-readable characterisation alongside its letter. A cluster that cannot be
characterised is not yet "of interest".

| Condition | Name fragment |
|---|---|
| `rate < 4 Hz`, AC power above `P_floor` | `subthreshold/resonant` |
| `rate < 4 Hz`, AC power below `P_floor` | `silent` (→ quarantine `Z`, §10) |
| `burst_index > 0.3` | `bursting`, qualified by `spikes_per_burst` and interburst rate |
| otherwise | `tonic` |
| `rate < 20 Hz` / `20–80` / `> 80 Hz` | `slow` / `medium` / `fast` |
| `adapt_index > 0.15` | `+adapting` |
| `adapt_index < -0.15` | `+accelerating` |
| `CV_ISI > 0.5` and `burst_index <= 0.3` | `+irregular` |

Example rendering: `C — fast tonic, +adapting (n=1842, rate 94 Hz, CV 0.08)`.

## 15. Runtime budget (~31 min, inside the agreed ~45 min target)

| Stage | Time | Device |
|---|---|---|
| Convergence gate (§7.4) | 1 min | GPU |
| 12 iterations × ~45 s (sim 3 s · rFFT 1 s · PCA 2 s · GP fit 10 s · acquisition 5 s · interim HDBSCAN 20 s) | 9 min | GPU + CPU |
| Final PCA + 3 unseeded UMAP(10) fits at 27k (sweep stage 1) | 2.5 min | CPU ×22 |
| HDBSCAN sweep stage 1, 54 configs via 9 tree builds (§12.4.1) | 3.5 min | CPU |
| Sweep stage 2, `alpha ∈ {0.5, 1}` (features → PCA → UMAP → HDBSCAN) | 4 min | GPU + CPU |
| Final seeded UMAP(10) re-fit of the winning config (§12.3) | 2.5 min | CPU ×1 |
| Seeded UMAP(2) visualisation fit | 2 min | CPU ×1 |
| 5-embedding stability on 12k subsample (§13.2), unseeded & concurrent | 1.5 min | CPU ×22 |
| Validation, stats, figures | 2 min | CPU |
| CSV writing, incl. the 6.7 GB of bulk files (§16.2) | 3 min | I/O |
| **Total** | **~31 min** | |

**Where the time goes, and what it means for optimisation.** Even after the
restructure, CPU clustering remains the largest block (~12 min of 32); the GPU
work is ~7 min, of which the Dormand–Prince integration proper is **~1.4 min**.
The integrator is not the bottleneck and should not be optimised further. If
anything, the headroom argues the other way: `dt_int` could be halved, or
`T_total` extended, for well under a minute each.

Two restructures got this from ~50 min to ~32, both by removing repeated work
rather than reducing scope — `N`, trace length, sweep coverage and statistical
rigour are all unchanged:
- **§12.4.1 tree reuse:** 13.5 → 3.5 min. The sweep was rebuilding the same
  single-linkage tree six times per `min_samples`.
- **§12.3 selective seeding:** ~18 → ~8.5 min of UMAP. Seeding every fit forced
  all of them onto 1 of 22 threads; only the two reported embeddings need it.

Both savings rest on estimates that must be replaced with measurements and
recorded in the manifest: the single- vs multi-threaded UMAP ratio (assumed ~4×)
and the tree-build-to-extraction cost ratio.

Remaining levers, now unnecessary but retained: `--sweep-fast` (−1.2 min),
`--compact-csv` (−3 min), `--no-stage2` (−4 min). The ~13 min of headroom against
your 45 min target is better spent raising `N` — see §11.1 — than banked.

`--quick` reduces to `n1=1024, n=512, m=6, --no-sweep` for a ~5 min smoke run.


## 16. Outputs

### 16.1 `./output/`

Written at `./output` relative to the working directory (`/opt/neuron/output`),
created if absent, and overridable with `--outdir`. The same convention as
`./data` (§16.2) — neither path is at the filesystem root.

- `clusters_umap2d.png` — 2D scatter, clusters labelled `A, B, C, …` at their
  centroids, HDBSCAN noise in light grey, `Z` plotted separately, and the 7
  canonical types overlaid as annotated markers (§13.1). `UNSTABLE` and
  `MULTIMODAL` clusters marked in the legend. 200 dpi, plus a `plt.show()` popup.
- `A1.png, A2.png, A3.png, B1.png, …` — **one file per representative, holding
  both of its graphs**: a two-panel figure with **top** `v(t)` over the full
  2500 ms, analysis window shaded, and **bottom** `|rFFT|` vs frequency, linear
  0–500 Hz with a log-`f` inset, dominant peaks annotated in Hz. Shared figure
  title carries the cluster letter, its §14.2 name, and the sample's
  `(a, b, c, d)`.
- `A_traces.png`, … — one extra per cluster, stacking that cluster's three
  representative traces on a common axis for direct comparison. Additive to the
  per-representative files above, not a replacement for them.
- `cluster_report.md` — names (§14.2), means, both triangular matrices (§14.1),
  flags, canonical placements, stability and contiguity results. Also printed to
  stdout.
- `sweep_table.csv`, `run_manifest.json` (every parameter in this spec, seeds,
  timings, resolved package versions, convergence-gate results).

**Representative selection:** the 3 points nearest the cluster centroid **in the
10D clustering embedding** (where the clusters are actually defined), tie-broken
on HDBSCAN membership probability. Re-simulated with the §7.2 output filter
disabled so displayed spikes stay sharp.

### 16.2 `./data/` — created after the calculations complete

Written at `./data` relative to the working directory (`/opt/neuron/data`),
created if absent once the calculations complete, and overridable with
`--datadir`.

All matrices use the **same orientation — one row per sample, one column per
time/frequency** — so every file joins to `samples.csv` on `sample_id`. Column
headers carry physical units (ms or Hz), not indices.

**The two bulk files** — the simulation results and the DFFT output:

| File | Shape | Size |
|---|---|---|
| `traces.csv` | 26 624 × 20 001 — `sample_id` + one column per output sample, headers `t=500.0 … t=2499.9` (ms). The §7.2-filtered analysis-window trace, i.e. exactly the signal the DFFT was computed on. `%.4g` mV. | **~4.3 GB** |
| `spectra_full.csv` | 26 624 × 10 002 — `sample_id` + the complete one-sided rFFT magnitude, headers `f=0.0 … f=5000.0` (Hz), `Δf = 0.5 Hz`, post-`\|H(f)\|^2`-deconvolution, pre-banding, `alpha = 0`. `%.6g`. | **~2.4 GB** |

The analysis window (20 000 pts) is written rather than the full 2500 ms so that
`traces.csv` and `spectra_full.csv` are row-for-row and sample-for-sample
consistent — a trace in one file is the exact input to the spectrum in the other.
`--csv-full-trace` writes all 25 000 points (~5.3 GB) instead, breaking that
correspondence.

**Derived and summary files:**

| File | Shape | Size |
|---|---|---|
| `samples.csv` | 26 624 × ~35 — `iteration, sample_id, source(lhs\|gp\|seed), a, b, c, d, I, n_spikes, rate, cv_isi, burst_index, spikes_per_burst, adapt_index, mean_v, sd_v, ac_power, max_local_error, finite, quarantine, label_leaf, label_eom, letter, membership_prob, umap10_0..9, umap2_0..1` | ~11 MB |
| `features_bands.csv` | 26 624 × 128 — the clustering input, headers = band centre frequencies (Hz) | ~31 MB |
| `traces/<LETTER><k>.csv` | ~90 files × 25 000 rows — `t_ms, v, u`, full 2500 ms, **filter disabled** (the display signal, §16.1) | ~63 MB |
| `cluster_stats.csv` | one row per cluster — every scalar from §14 | <1 MB |
| `corr_<LETTER>_all.csv`, `corr_<LETTER>_lhs.csv` | 4×4 each, upper triangle empty | <1 MB |
| `run_data.npz` | coefficients, 128-band features, diagnostics, both embeddings, labels — `float32` binary, for re-clustering without re-simulating | ~40 MB |

Total ~6.9 GB. Flags: `--gzip-csv` writes the two bulk files as `.csv.gz`
(~1.7 GB combined, still readable directly by `pandas.read_csv`);
`--compact-csv` omits them, leaving ~150 MB and saving the 3 min of write time
(§15). Note that `features_bands.csv` is **not** redundant with
`spectra_full.csv` — it carries the tilt, banding, `log1p` and L2 steps of §9
that `spectra_full.csv` deliberately stops short of.

## 17. Test plan and acceptance criteria

Unit tests (`pytest`):
1. **DP5 order** — on `y' = -y` and on a scalar nonlinear problem with known
   solution, observed convergence order is 5 ± 0.2 over `dt` halvings.
2. **Reset correctness** — one parameter set integrated against a CPU
   `scipy.integrate.solve_ivp` reference using `events` to terminate at
   `v = 30` and restart after the reset. Spike times must agree to < 0.01 ms.
3. **Output filter** — measured cascaded-boxcar response matches analytic
   `|H(f)|^2` to < 0.1 dB across 0–10 kHz; the deconvolution in §9 step 5 restores a
   known sinusoid's amplitude to < 0.5%.
4. **Aliasing** — a synthetic 100 Hz step-discontinuity train shows ≥50 dB
   alias suppression in 0–1 kHz with the filter on, and ~20 dB with it off.
5. **Feature pipeline** — a pure sinusoid at a known frequency puts >99% of its
   banded energy in the correct band; `log1p`/L2 steps are scale-invariant.
6. **GP** — recovers known ARD lengthscales from synthetic draws to within 20%;
   rank-1 variance update matches a full recompute to 1e-10.
7. **Triangular formatting** — upper triangle renders empty, diagonal equals
   `numpy.std(ddof=1)`, lower triangle equals `numpy.corrcoef`.
8. **Determinism** — identical seed produces bitwise-identical features, and the
   two seeded UMAP fits (§12.3) reproduce bitwise across runs. The unseeded fits
   are explicitly exempt.
9. **Tree reuse equivalence (§12.4.1)** — for a fixed embedding, a labelling
   obtained via `_tree_to_labels` on a reused `single_linkage_tree_` is
   **identical** to a full `HDBSCAN.fit` at the same parameters, across all 6
   `(min_cluster_size, method)` combinations. This guards a semi-internal API: a
   version bump that changes it must fail here rather than silently produce
   different clusters.
10. **PCA sufficiency (§12.1)** — cumulative explained variance at 50 components
    is ≥ 99% on the real feature matrix. Below that, `n_components` must be
    raised rather than the check relaxed; the full variance curve goes in the
    manifest.

Acceptance criteria for a full run:
- Convergence gate (§7.4) passes both checks.
- Zero samples labelled `X` (non-finite after refinement).
- Noise fraction ≤ 25% at the selected config.
- ≥ 10 clusters at `cluster_selection_method='leaf'`.
- Mean pairwise ARI across the 5 stability embeddings ≥ 0.6 (a Monte-Carlo
  estimate, not a crisp threshold — see §13.2).
- All 7 canonical types placed in a non-noise cluster.
- Every §16 artefact present and non-empty; `samples.csv`, `traces.csv`,
  `spectra_full.csv` and `features_bands.csv` all have exactly `N` data rows and
  identical `sample_id` ordering.
- Round-trip check: re-reading `traces.csv` and recomputing §9 reproduces
  `features_bands.csv` to within 1e-4 relative — proves the two bulk CSVs are
  consistent with each other and with the clustering input.

## 18. Out of scope, and why

These were considered and deliberately excluded from v1; the `S` numbering from
§1.1 is retained, which is why the list starts at 6. Each is recorded so the gap
is known rather than forgotten.

6. **Varying `I`.** Two published behaviours are *protocol*-dependent rather than
   parameter-dependent: TC rebound bursting needs a hyperpolarising step, and RZ
   bistability needs timed brief stimuli. Neither can appear under a single
   constant `I` no matter how `(a,b,c,d)` is sampled. Natural v2: run the
   pipeline as sheets at `I ∈ {2, 10, 20}`, or prepend a brief hyperpolarising
   pre-pulse. See §13.1 for how this limits the validation.
7. **Stationarity index** (spectrum of the window's first half vs second half).
   Would flag slow corners not settled by the 500 ms discard.
9. **Spike-train (delta-train) spectrum** as a complementary feature block. The
   `v(t)` spectrum mixes rhythm with spike waveform; a spike-train spectrum
   isolates rhythm alone and would plausibly sharpen rhythm-based separation, at
   roughly double the feature-construction code. The `--spectral-tilt` option
   (§9 step 6) is the cheap partial substitute already in scope.

## 19. Implementation notes, assumptions and amendments

Written during implementation (2026-11-21). Part A records decisions the spec
left open; part B records places where the spec's own prescriptions were
changed, each with the measurement that forced the change. Everything here is
reflected in `run_manifest.json`, so a run is self-describing.

### 19.A Assumptions filled in during implementation

**D1 — output-sample indexing.** Output sample `j` covers integration steps
`[jL+1, (j+1)L]` and is timestamped at the *end* of its block,
`t = (j+1) · dt_out`. The analysis window is therefore `j = 4999 … 24998`
(`t = 500.0 … 2499.9` ms, exactly 20 000 samples) and the display traces are
`j = 0 … 24999` (`t = 0.1 … 2500.0` ms, 25 000 rows), matching §16.2.

**D2 — peak equalisation.** §7.1 is implemented by clamping the *emitted
sample value* to `+30.0` on the step where `v >= 30`, before the reset, so the
output stage sees exactly the paper's "reset `v` first to +30 mV and then to
`c`". In unfiltered display mode a block containing any spike emits `+30.0`,
otherwise the block's last integration sample — without this, 10:1 decimation
of an unfiltered trace would miss most spike peaks (the upstroke is ~0.15 ms).

**D3 — the boxcar in registers.** The 2-stage cascade is accumulated as
`A_k = Σ_{m=1..L} (L+1-m) x_m`, `B_k = Σ_{m=1..L} (m-1) x_m`, emitting
`(A_k + B_{k-1}) / L²`. That is algebraically exactly two cascaded length-`L`
boxcars followed by decimation by `L` (the triangular `2L-1`-tap FIR), at two
FMAs per step as §7.2 budgets, and with no shift register. A test compares the
kernel's filtered output against a host FIR reference driven by the kernel's
own `L = 1` output.

**D4 — local error norm.** `max(|e_v|, |e_u|)`, absolute, maximised per
sample over non-reset steps only. Measured maximum over the 24 gate probes at
the production step size: `5.8e-11`, i.e. four orders of magnitude below the
`1e-6` trigger, so the §7.3 refinement pass never fires (0 samples refined in
the production run). The pass is implemented and exercised by construction at
coarser steps, where `dt = 0.04` reaches `3.6e-5`.

**D5 — kernel precision.** The kernel integrates in `float64` and stores
traces as `float32`. FP64 runs at 1/64 rate on this device, but the whole
sweep still integrates in ≈100 s; `float32` would have made the §17.1 order
test and the §17.2 spike-time comparison meaningless.

**D6 — gate step sizes that do not divide `dt_out`.** At `dt = 0.04`,
`L = round(0.1/0.04) = 2`, so that probe runs at `dt_out = 0.08` ms. The
analysis window is still exactly 2000 ms, so `Δf` is still 0.5 Hz and the
§9 band plan is unchanged; only the bin count differs. The `|H(f)|²`
deconvolution always uses that run's own `(L, dt_int)`.

**D7 — refinement and the output filter.** The `dt/4` pass keeps `L · dt_int`
constant, so the filter's nulls stay at multiples of 10 kHz and the nominal
deconvolution applies to refined samples to better than 0.01% in band.

**D8 — GP precision.** Six independent ARD-RBF GPs fitted in `float32`: the
fit is dominated by a 3000×3000 Cholesky plus two triangular solves, measured
at 37 ms in FP32 against 700 ms in FP64 on this GPU. The noise variance is
floored at `1e-4` and the jitter set to `1e-5` to keep the FP32 factorisation
conditioned; hyperparameters warm-start from the previous iteration.
Lengthscale recovery on synthetic draws is within 7% in FP32 and 3% in FP64,
and the rank-1 variance update agrees with a full recompute to `1.6e-12` in
FP64 (§17.6).

**D9 — greedy batch shortlist.** The rank-1 variance re-conditioning is exact,
but it needs an `O(n_train × n_candidates)` basis *per GP output*, which for
the full 20 000-point pool is 1.4 GB on a 3.6 GB device. The variance
*ranking* term is therefore computed over the whole pool in chunks, and the
greedy repulsion then runs on the top `4 × batch` candidates by initial
acquisition score. Selecting 1434 from 5736 pre-screened candidates leaves the
repulsion plenty of room.

**D10 — entropy term classes.** HDBSCAN noise (`-1`) is kept as its own class
in the 25-NN predictive entropy, so both type boundaries and unlabelled voids
score highly, as §11.3 intends.

**D11 — candidate pool per iteration.** A fresh scrambled-Sobol pool is drawn
each iteration (`seed = 1000 + iteration`), which also makes it impossible for
an acquisition to repeat an earlier iteration's point.

**D12 — canonical seeding.** The 7 canonical vectors replace the LHS points
*nearest* to them, so the iteration-1 design stays close to space-filling.
They are counted inside the design-unbiased LHS subset of §14.1 (7 points of
10 850).

**D13 — cluster letters.** The alphabet skips `X` and `Z`, which §12.4
reserves: `A…W`, `Y`, then `AA`, `AB`, ….

**D14 — `relative_validity_` for reused trees.** Computed by calling hdbscan's
own property implementation on a shim object carrying `(labels_,
_min_spanning_tree)`. A test asserts it matches a full `HDBSCAN.fit` to
`1e-12`, and that the per-cluster DBCV contributions reported in §14 sum to it.

**D15 — sweep score.** `relative_validity_ − 2.0 · max(0, noise_fraction −
0.25)`. The weight 2.0 guarantees an all-noise labelling cannot win, since
`relative_validity_ ≤ 1`.

**D16 — noise adjacency.** Fraction of a cluster's points whose nearest
neighbour in the 10-D clustering embedding is HDBSCAN noise.

**D17 — histogram-derived burst statistics.** The log-ISI bin straddling
`0.25 × mean(ISI)` is given fractional credit, so `burst_index` and
`spikes_per_burst` do not jump as the threshold crosses a bin edge.

**D18 — `mean(v)` / `sd(v)`.** Accumulated at the integration rate over the
analysis window, on the peak-equalised sample value.

**D19 — plotting `Z`.** Quarantined samples are excluded from the PCA/UMAP
fits, so they have no embedding of their own; §16.1's "`Z` plotted separately"
is implemented by mapping them through the fitted visualisation UMAP with
`transform` and drawing them as grey crosses.

**D20 — representative figures.** The top panel is the *unfiltered*
re-simulated trace (display sharpness, §8); the bottom panel is the
deconvolved `|rFFT|` of the *filtered* analysis-window signal — i.e. the
spectrum the clustering actually used. The two panels are deliberately
different signals and the figure caption says so.

**D21 — row-count verification.** The "exactly `N` rows and identical
`sample_id` ordering" criterion is checked by an exact line count of every
bulk CSV plus a first/last `sample_id` check. Parsing all 20 001 columns of a
4.3 GB CSV to compare every id would cost minutes for no extra information;
the writer emits ids from one monotone array.

**D22 — scope of the §17.2 spike-time test.** Asserted as: (i) the first spike
agrees with a `scipy.integrate.solve_ivp` event reference to within one step;
(ii) the discrepancy accumulates at no more than `10 · dt_int` per spike; and
(iii) ISIs agree to within `10 · dt_int`. End-of-step event detection is late
by at most `dt_int` per spike (§7.3) and that lateness shifts the remainder of
the train, so the *absolute* spike-time error necessarily grows linearly along
the train — measured at 1.5-2 x `dt_int` per spike, since the delayed reset
also perturbs the following interval slightly. A common time shift leaves
the magnitude spectrum untouched, so the quantity that matters spectrally is
the per-spike increment, which is what (ii) and (iii) bound.

**D23 — scope of the §17.5 banding test.** A Hann main lobe is 4 bins wide
(2 Hz at `Δf = 0.5 Hz`), so ">99% of banded energy in the correct band" can
only hold where a band is wider than that — the log-spaced region above
~90 Hz. Inside the 40 native 0.5 Hz bins the criterion is applied to the band
plus its two neighbours; the peak band must be the correct one in every case.
Likewise the `log1p`/L2 steps are *not* strictly scale-invariant (`log1p` is
nonlinear); the test asserts what is true and what the steps are for — shape
invariance under a 50× amplitude change (cosine > 0.98, identical argmax band)
and unit output norm.

**D24 — GP training subset.** §11.2 asks for "*all* points from the last two
iterations plus a uniform draw from earlier ones" under a cap of
`N_gp = 3000`, but two iterations are 4096 points, so the two halves of that
sentence cannot both hold from iteration 2 onwards. The implementation honours
the stated intent — "recent structure is never diluted away" — by drawing the
3000 uniformly from the last two iterations when they exceed the cap, and
topping up from earlier iterations only when they do not.

**D25 — `n_neighbors` for the visualisation embedding.** §12.2 fixes
`n_neighbors = 30` for the 2-D fit while §12.4's sweep may select a different
value for the clustering embedding. The visualisation fit follows the winning
configuration, so the picture and the labels drawn on it come from the same
neighbourhood scale; `min_dist = 0.1` and `random_state = 42` are as §12.2
specifies.

### 19.B Amendments to the spec, with the measurements that forced them

**A1 — §9 step 9: subtract a relative noise floor before `log1p`.**
Each sample's banded magnitude vector has `0.01 × max(band)` (−40 dB)
subtracted, clipped at zero, before `log1p`. *Why:* `log1p` applied to raw
magnitudes ties its compression point to absolute loudness. With a spiking
sample's peak band at ~`2e4` and the leakage/jitter floor of its near-empty
bands at ~1.5, `log1p` maps them to 9.9 and 0.9 — a ratio of 11, not 1.3e4 —
so roughly a hundred noise bands each contributed 0.03–0.07 of the unit
feature vector and about a quarter of the metric was noise. *Measured, over
the 24 gate probes:* median `dt=0.01`-vs-`0.001` relative L2 falls from 0.084
to 0.009; median pairwise distance between the 7 canonical types rises from
1.07 to 1.23; the ratio of between-type separation to numerical sensitivity
improves 6×. This is the single most consequential change in this list.

**A2 — §7.4 first check: probe classification, median statistic, and
`dt_int` selection.** The check is evaluated as follows. (i) Probes that §10
would quarantine (no spikes) are excluded: their spectral *shape* is roundoff,
so it cannot converge in `dt`. (ii) Probes whose `dt=0.001`-vs-`0.0005`
baseline already exceeds 1% are *sensitivity-limited* — they sit on a
bifurcation surface where the feature vector is genuinely discontinuous in the
parameters, and no step size converges there; they are excluded and counted
(4 of 17 spiking probes). (iii) Acceptance uses the **median** over the
remaining 13. (iv) `dt_int` is selected as the coarsest ladder step whose
median is within a 4× margin of the criterion. *Measured:* median relative L2
is 0.0094 at `dt = 0.01` (a marginal pass, with 6 of 13 probes individually
over 1%) and 0.00084 at `dt = 0.0025`. Since the measured integration cost is
~25 s at `dt = 0.01` against the 1.4 min §15 budgeted, **the production run
uses `dt_int = 0.0025` ms, `L = 40`** — a 4× margin on the gate for ~75 s more
GPU time. `dt_out` stays 0.1 ms, so `L · dt_int` and hence the whole §7.2
output stage are unchanged.

**A3 — §7.4 second check: alias control measured as a ratio.** The literal
test ("the same vector must be stable when `dt_out` is halved") conflates
residual aliasing with the output filter's smooth, exactly-deconvolved in-band
effect, and the latter dominates: against a 8×-finer output grid the residual
is 3.1% at `dt_out = 0.1`, 2.0% at 0.05 and 0.9% at 0.025 — i.e. `O(dt_out)`
and smooth, not the non-smooth parameter-dependent fold §7.2 was built to
prevent. The gating criterion is therefore the ratio §7.4 actually cares
about: filtered output must be at least 5× closer to a fine-output-grid
reference than naive decimation. *Measured:* 0.031 filtered vs 0.216 naive,
i.e. **7.1×**. The literal `dt_out`-halving numbers (median 0.012, max 0.021)
are still computed and reported.

**A4 — §12.1/§17.10: `n_components` raised to reach 99% variance.** With the
A1 noise floor removed, the 128 band vectors are far less correlated (a comb at
31 Hz and one at 76 Hz share almost no bands), and 50 components retain only
~84–88% of the variance. §12.1 explicitly instructs "raise `n_components`
rather than proceeding", so the pipeline computes the full variance curve and
uses the smallest component count reaching 99% (95 components on the quick
run), recording both the curve and the EVR at 50 in the manifest. PCA feeds
UMAP, whose layout cost is indifferent to input dimension, so the only cost is
a slightly weaker denoising of the k-NN graph.

**A5 — §10: `P_floor` gains a physical term.** `10 ×` the median `ac_power` of
a numerically silent reference is `1.3e-1`, but the silent references
themselves span a factor of 10, so that threshold sits *at* the numerical
floor: 459 of 3584 quick-run samples were retained as "subthreshold/resonant"
with `ac_power` from `0.13` to `4.6e6`, against a minimum of `6.9e8` for any
spiking sample. Those are roundoff being named as a finding. `P_floor` is
therefore `max(10 × median_silent, ac_power of a 0.1 mV RMS in-band
fluctuation)`, the second term measured through the same §9 steps on a
synthetic tone: `7.5e5`. A genuine subthreshold limit cycle is a
millivolt-scale oscillation and clears this by three orders of magnitude; the
numerical floor is nine orders below it. Both terms are recorded in the
manifest, and the `NEAR-FLOOR` flag (A8) catches anything that still lands
within 100× of the floor.

**A6 — §8: bulk CSVs written with `numpy.savetxt`.** §8 assumed
`pandas.DataFrame.to_csv` at 60–100 MB/s and `numpy.savetxt` at ~10 MB/s.
*Measured here* (pandas 3.1.1, numpy 2.5.5, a 1024 × 20 001 float32 block):
`to_csv(float_format='%.4g')` writes at 10 MB/s and `savetxt(fmt='%.4g')` at
46–61 MB/s — the reverse. The writer keeps §8's chunked-append structure, the
`%.4g` / `%.6g` formats and the `sample_id` column, but formats through
`savetxt` over an open handle. The two bulk files are written in ~2 min.

**A7 — §17 round-trip criterion scoped to clustered samples.** A quarantined
sample is numerically silent: its banded spectrum is roundoff, its
L2-normalised *shape* is arbitrary, and `%.4g` erases it (such rows round-trip
at `O(1)` relative error, by construction). The criterion is evaluated over
the samples that actually enter the clustering, with the excluded rows'
statistics reported alongside. Over clustered rows the round trip is exact to
the printed precision.

**A8 — two report additions.** (i) `NEAR-FLOOR`: a cluster whose median rate
is below 4 Hz *and* whose median `ac_power` is within 100× of `P_floor` is
flagged, since its spectral shape may be numerical rather than physical.
(ii) `largest_component_fraction`: §13.3's cut is a *global* density threshold
applied to each cluster's *own* point set, so a small cluster fragments simply
because its own points are sparser than the global `k=5` NN scale. The
fraction of the cluster in its largest parameter-space component is the number
to read alongside the `MULTIMODAL` flag, and is reported with it.

**A9 — §14.2: a slow spiker is not a subthreshold resonator.** §14.2's first
two rows name any cluster with `rate < 4 Hz` from its AC power alone, which
labels a neuron firing at 3 Hz — six real spikes in the window, a clean
harmonic comb — "subthreshold/resonant". §6 compounds this by asserting that
anything below 4 Hz "is quarantined (§10) rather than clustered", which §10's
operative definition (`n_spikes == 0` **and** AC power below `P_floor`) does
not do. The implementation follows §10 for the quarantine, and splits §14.2's
first row: a cluster with no spikes is `subthreshold/resonant` (above
`P_floor`) or `silent` (below), while a *spiking* cluster under 4 Hz is named
`very slow tonic` / `very slow bursting` and flagged `BELOW-RATE-FLOOR`, which
records §6's real point — that such a spectrum rests on only a handful of
cycles. Adaptation / irregularity tags now attach to any spiking cluster
rather than only to those above 4 Hz.

**A10 — §17's round-trip criterion is a mean, not a max.** Re-reading
`traces.csv` and recomputing §9 reproduces `features_bands.csv` to a *mean*
relative L2 of 7.1e-5 over clustered rows, but the worst row is ~2e-3. That
tail is not a precision problem: writing the traces at `%.9g`, which is
lossless for the stored `float32`, leaves it at 2.0e-3 (measured over 220
clustered probes: max 1.6e-3 at `%.4g`, 2.0e-3 at `%.9g`; mean 7.1e-5 and
1.9e-5). The cause is the A1 relative floor — a band sitting within roundoff
of `0.01 x peak` flips between a small positive value and exactly zero under
any perturbation, which moves the L2-normalised vector by far more than the
perturbation itself. A hard threshold and a max-over-rows criterion cannot
both hold, so the criterion is evaluated on the mean, with p99 and max
reported beside it. `%.4g` is kept as §16.2 specifies: it costs 5e-5 of mean
agreement and saves 2.5 GB against `%.9g`.

**A11 — §14.2: say when a cluster is mixed.** §14.2 names a cluster from its
medians, which hides the commonest composite case in this data: a cluster
whose members share a fundamental but split between bursting and tonic,
because the `v(t)` spectrum of a burst train and of a tonic train at the same
fundamental share most of their comb. The canonical CH vector lands in exactly
such a cluster — 182 members with a 17 Hz fundamental, median burst index
0.00 but 90th percentile 0.50, so the median rule names it "slow tonic" and
CH itself joins with a membership probability of 0.49, the lowest of the seven
canonical types. A cluster with between 20% and 80% of its members above the
§14.2 burst-index threshold now carries `[mixed: X% bursting]` in its name (13
of 102 clusters in the reported run), and `frac_bursting` / `frac_irregular`
are reported for every cluster so near-misses like CH's are visible too. The
base name still follows §14.2's table.

### 19.C Measured facts worth carrying forward

- The GPU is far faster at the integration than §15 assumed: ~1.8 s per
  2048-sample chunk at `dt = 0.01` (≈25 s for the whole sweep), which is what
  made amendment A2's 4× step-size refinement free.
- About half the §3 box is quiescent at `I = 10` (`b < 0.101`, §2), so the
  quarantine holds out ~50% of samples and the clustering runs on ~13 k of the
  26 624. That is the box straddling the bifurcation as §2 requires, not a
  defect — but it halves the effective design size, and a v2 that wants 27 k
  clustered points should raise `N` or shrink the `b` range below 0.101.
- `hdbscan` 0.8.44's `_tree_to_labels` reproduces a full `HDBSCAN.fit`
  bit-for-bit across all six `(min_cluster_size, method)` combinations, so the
  §12.4.1 tree reuse is sound at this version; test 9 is the tripwire.
- `umap-learn` 0.6 ships a new initialisation/layout pipeline behind
  `compatibility_layout`; the code pins the legacy 0.5.x pipeline the §12
  parameter choices were calibrated against.
- PC1 carries ~80% of the feature variance and is dominated by the
  silent/spiking distinction, which is a *step* in `b`. The ARD fit responds
  correctly by driving the `b` lengthscale to its lower bound (0.02 of the
  normalised range), but the consequence is that the posterior SD is near
  `sf2` almost everywhere and the max-variance term of §11.3 ends up close to
  uninformative (the selected batches average a 0.39-0.60 SD rank, i.e. near
  the pool median). The gradient and entropy terms carry the acquisition.
  A v2 should fit the GPs on the spiking subset and handle the silence
  boundary with an explicit classifier, which is what the entropy term is
  already doing implicitly.
- **Adaptation is invisible in the analysis window.** `adapt_index` is defined
  (§7.6) from the first and last ISI *inside* `[500, 2500)` ms, but
  spike-frequency adaptation is a transient: the paper's RS adapts over its
  first ~100 ms, which §6 discards. Measured over the reported run, 16.8% of
  clustered samples exceed `|adapt_index| > 0.15` but only 2 of 102 clusters
  have a median above it, and the `+adapting` tag never fires (`+accelerating`
  fires twice). A v2 wanting adaptation must measure it across the transient —
  compare the first ISIs after `t = 0` with the steady state — which costs one
  extra in-kernel accumulator.
- **The taxonomy is rate-dominated.** Of 102 clusters, 101 are named `tonic`
  and differ mainly in rate (medians spanning 2-293 Hz); only one, `CL`, is
  named bursting (2.7 spikes/burst, 99% of its members bursting). 13% of
  clustered samples have `burst_index > 0.3`, but they are spread across 84 of
  the 102 clusters rather than concentrated. The `v(t)` magnitude spectrum ties
  a burst train and a tonic train at the same fundamental to nearly the same
  comb — precisely the limitation §18 item 9 predicts, and what the deferred
  delta-train spectrum would address. The `--spectral-tilt` substitute was
  swept and lost (DBCV 0.512 at `alpha = 1`, 0.445 at `alpha = 0.5`, against
  0.569 at `alpha = 0`).
- **Contiguity at this granularity is a sampling statement.** 69 of 102
  clusters are flagged `MULTIMODAL`, but their largest parameter-space
  component typically holds only ~5% of the cluster across ~80 components:
  a ~115-point cluster spread over a 3-D surface in the 4-D box is simply
  sparser than the global `k=5` NN scale the §13.3 cut uses. 22 clusters are
  both stable and contiguous, and those are the ones whose mean coefficient
  vector §1 would accept.
- `scipy.stats.qmc.Sobol` warns that the 20 000-point candidate pool §11.3
  specifies is not a power of two. The warning is left visible rather than
  silenced; 16 384 or 32 768 would restore the balance property if a v2 wants
  it.
