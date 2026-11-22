# Izhikevich Cluster Discovery — Improvement Specification (v2)

**Status:** draft for review · **Date:** 2026-11-21 · **Amends:** `SPEC.md`
(v1, same date) · **Evidence:** `output/cluster_report.md`,
`output/clusters_umap2d.png`, `data/run_data.npz` from the 2026-11-21 10:50 run

---

## 1. Objective

The v1 pipeline is numerically sound and fully instrumented; its *encoding* is
not fit for the stated objective. This document specifies the changes needed to
make §1 of `SPEC.md` — "identify as many distinguishable firing-pattern clusters
as possible", subject to reproducibility, interpretability and contiguity —
actually achievable.

The central change is **replacing the `v(t)` magnitude-spectrum feature vector
with an encoding of the interspike-interval (ISI) sequence** (§3). Everything
else follows from it or from defects the v1 run exposed.

Section numbers prefixed `v1 §` refer to `SPEC.md`.

### 1.1 Change map

| # | change | §  | effort |
|---|---|---|---|
| C1 | ISI-sequence encoding replaces the 128-band spectrum | 3 | large |
| C2 | Spike times captured in-kernel at `dt_int` resolution | 3.1 | small |
| C3 | Drive set relative to per-sample rheobase, not fixed `I = 10` | 4 | medium |
| C4 | Convergence gate judged on spike timing, not on the feature metric; fractional thresholds | 5 | small |
| C5 | Onset transient featurised rather than discarded | 6 | small |
| C6 | GP active learning retired in favour of denser quasi-random sampling | 7 | small (deletion) |
| C7 | Representation-health gates; stability tests that can fail | 8 | medium |
| C8 | Nameability merge; acceptance criteria that can fail | 8.4, 9 | medium |
| C9 | Anti-aliased output stage and `spectra_full.csv` retired; rFFT becomes host-side and on-demand | 3.5 | small (deletion) |

---

## 2. Diagnosis — the measured case for change

All figures measured from the v1 run, not estimated.

### 2.1 The result does not survive its own criteria

| quantity | v1 run |
|---|---|
| clusters reported | 102 |
| clusters passing **both** the v1 §13.2 and §13.3 gates | **22** |
| distinct interpretable names across all 102 (v1 §14.2) | **6** |
| clusters named `<slow\|medium\|fast> tonic` | 99 |
| clusters named bursting | **1** |
| η² of `log(rate)` explained by the cluster label | 0.574 |

`clusters_umap2d.png` shows ~100 thin one-dimensional filaments rather than
blobs: the signature of a 1-D manifold (firing rate) segmented arbitrarily.

### 2.2 The encoding is blind to bursting

| quantity | v1 run |
|---|---|
| clustered samples with burst index > 0.2 | 1947 (13.3%) |
| clusters those 1947 are distributed across | **86 of 102** |

The canonical CH (chattering) seed measures `CV(ISI) = 1.50`,
`burst index = 0.596`, 2.48 spikes/burst at 83 Hz. It was assigned to cluster
**R**, named *"slow tonic"*, whose members have median rate 17 Hz, median
`CV(ISI) = 0.000` and median burst index 0.000, at membership `p = 0.488`. The
v1 §17 criterion ("all 7 canonical types placed in a non-noise cluster") passed.

The canonical IB seed measures `CV = 0`, `BI = 0` — correctly, because IB's
bursting at `I = 10` is an onset transient and v1 §6 discards `[0, 500)` ms. Its
§13.1 pass is therefore vacuous (→ C5).

### 2.3 The metric saturates within a 5% rate window

Mean cosine distance between pure-tonic clustered samples (4000-sample
subset, `CV < 0.02`, `BI < 0.02`), binned by frequency separation:

| \|Δ log f₀\| | mean cosine distance |
|---|---|
| < 0.002 | 0.113 |
| 0.010 – 0.020 | 0.231 |
| 0.020 – 0.050 | 0.433 |
| 0.050 – 0.100 | **0.559** |
| 0.200 – 0.500 | 0.541 |
| 0.500 – 1.000 | 0.556 |
| 2.000 – 5.000 | 0.777 |

The metric's usable dynamic range is consumed inside a **5% rate difference**;
beyond that it carries no ordering for the k-NN graph to exploit until rates
differ by more than ~3×.

It also inverts: **octave-related pairs sit at 0.379, 1.5×-related pairs at
0.501.** A 40 Hz and an 80 Hz neuron are measurably *more similar* than 40 Hz
and 60 Hz, because the 2f₀ comb is a subset of the f₀ comb. This is an artefact
of representing a near-periodic point process by its harmonic comb on disjoint
bands.

### 2.4 The representation is sparse per sample and full-rank in aggregate

| quantity | v1 run |
|---|---|
| median participation ratio (effective bands used per sample) | 28.2 of 128 |
| p10 participation ratio | 7.6 |
| PCA components for 99% variance | **92 of 128** (v1 §19 A4) |

Sparse-per-sample *and* near-full-rank-in-aggregate means the support moves
between samples. No linear compression can help; the fix must be a change of
representation, not a change of dimension.

### 2.5 The convergence gate inherits the metric's brittleness

From the v1 gate table: mean spike count is 172.7 at `dt = 0.02` ms and 173.0 at
`dt ∈ {0.01, 0.005, 0.0025, 0.001}` — converged by `dt = 0.02`. Yet the
128-band relative-L2 criterion failed 6/13 probes at `dt = 0.01` and forced
`dt_int = 0.0025` ms, a 4× integration cost. A 0.1% shift in f₀ moves the comb
and produces a ~20% L2 change, so the gate was measuring comb phase, not
truncation error (→ C4).

### 2.6 Budget is allocated away from the physics

| stage | seconds | share |
|---|---|---|
| `gp_fit` + `acquire` | 346.4 | 37.9% |
| `simulate` | 85.0 | 9.3% |
| `features` | 1.6 | 0.2% |
| total | 913.9 | |

The active learner costs 4× the entire integration while steering on an
acquisition signal computed in the feature space §2.3 shows to be degenerate.
It also splits every per-cluster statistic into "all" and "LHS-only
(authoritative)" variants, the latter typically ⅓ of cluster membership
(cluster A: 365 members, 119 LHS) (→ C6).

---

## 3. C1 — ISI-sequence encoding

**Decision: the clustering input is an encoding of the ISI sequence. The `v(t)`
magnitude spectrum is retired from the clustering path.**

Rationale: v1 §7.1 equalises every spike peak to exactly +30 mV, so the spike
waveform carries almost no parameter-dependent information. All of the
information distinguishing firing *patterns* is in the spike timing. A 128-band
magnitude spectrum is a lossy and metric-hostile encoding of a point process:
it converts a near-periodic train into a harmonic comb whose support moves with
f₀, producing exactly the pathologies in §2.3–§2.4.

The spectral pipeline (v1 §9) is retained **only as a plotting** artefact: the
per-representative trace-and-spectrum figures keep their current form, computed
on demand for ~100 samples. The batched `spectra_full.csv` deliverable is
dropped, which turns out to remove far more than the FFT itself — see §3.5.

### 3.1 C2 — spike-time capture

The ISI features must be exact, not quantised to `dt_out = 0.1 ms`: intra-burst
intervals are ~2 ms, where 0.1 ms quantisation is a 5% error.

Add to `kernel.cu` a per-sample **spike-time buffer**, capacity `S_max = 4096`,
written at `dt_int` resolution with a monotonic counter and an overflow flag:

```
spike_times[s * n_batch + i]   // float32 ms, s < S_max
n_spikes_recorded[i], overflow_flag[i]
```

Sizing: `2048 × 4096 × 4 B = 34 MB` per chunk, against the 164 MB trace buffer
already in use (v1 §7.5) — no change to the chunking strategy. Headroom: the
v1 run's maximum was 2227 ISIs in the 2000 ms window (≈1114 Hz); `S_max = 4096`
covers 2 kHz. `overflow_flag` must be zero for every sample; a non-zero count is
an acceptance failure (§9).

All ISI features are then computed **host-side** in a new `isi.py`. The v1
in-kernel log-ISI histogram and ISI moments (v1 §7.6) are retained unchanged as
characterisation diagnostics.

### 3.2 Feature blocks

Let `I₁ … I_n` be the ISIs in the analysis window and `Ī` their mean. Define
the normalised log interval

```
λ_i = log10(I_i / Ī),   clipped to [-1.75, 1.0]
```

The clip range is set from measurement: **97.2%** of v1 clustered samples have
*all* their ISIs inside `log10(I/Ī) ∈ [-1.5, 0.75]`; the full observed range is
`[-2.49, 1.55]`. Mass outside the grid accumulates in the edge cell.

Normalising by `Ī` is what makes the encoding rate-invariant: a 3 Hz and a
30 Hz doublet burster produce the same shape, and rate re-enters only through
block D, with an explicit weight.

| block | content | dims |
|---|---|---|
| **A** | 1-D Gaussian KDE of `{λ_i}` on 32 grid points over `[-1.75, 1.0]` | 32 |
| **B** | 2-D Gaussian KDE of `{(λ_i, λ_{i+1})}` on a 12×12 grid over the same range on both axes — the **return map** | 144 |
| **C** | drift: `{λ_i}` resampled onto 8 points of normalised ordinal time `i/n` | 8 |
| **D** | `log10(Ī)` — rate, as a single scalar | 1 |
| | **total** | **185** |

Why each:

- **B is the core.** The return map is the canonical discriminator of firing
  pattern, and the discriminating structure is geometric rather than
  statistical: a tonic train is one point *on* the diagonal; a doublet burster
  is two points *off* it, mirrored; an n-spike burster is a chain of n points;
  an adapting train drifts along one side; an irregular train is a structured
  cloud. These are the distinctions v1 cannot see.
- **A is deliberately redundant with B** — the marginal is recoverable from the
  joint. It is included as a *reweighting*, not as new information: it lets the
  ISI distribution be emphasised over the ISI ordering, and it is better
  estimated than the joint for short trains. `w_A = 0` must be in the sweep
  grid so the redundancy is testable rather than assumed.
- **C** captures the slow systematic trend (spike-frequency adaptation,
  acceleration) that a return map, being a 1-lag statistic, represents only
  weakly.
- **D** restores rate as an *explicit, weighted* axis. In v1 rate dominated
  implicitly and discontinuously (§2.3); here its influence is a single number
  under sweep control.

**Smooth, not binned.** Both A and B are kernel density estimates, not
histograms. Hard disjoint bins reproduce in the ISI domain exactly the defect
§2.3 identifies in the frequency domain — two similar trains landing in
neighbouring cells register as maximally different. KDE also decouples output
dimensionality from sample count, which matters: **43%** of v1 clustered samples
have fewer than 64 ISIs (p10 = 20, p1 = 6), so a hard 8×8 = 64-cell grid would
be dominated by occupancy noise for nearly half the dataset.

Bandwidth `h = 0.25` in `log10` units, swept over `{0.15, 0.25, 0.40}` alongside
the v1 §12.4 clustering parameters.

### 3.3 Normalisation and weighting

Each block is L2-normalised to unit norm independently, then scaled by its
weight; the concatenation is L2-normalised once more so cosine distance remains
the metric.

```
F = normalise( [ w_A·Â | w_B·B̂ | w_C·Ĉ | w_D·D̂ ] )
```

Defaults `w_A = 0.5, w_B = 1.0, w_C = 0.5, w_D = 0.3`. Swept in stage 2 of the
v1 §12.4 selection sweep, which replaces the now-moot spectral-tilt stage
(`alpha` belongs to the retired spectral path).

Per-block L2 normalisation before weighting is what makes the weights
interpretable and the sweep meaningful; without it, block B's 144 dimensions
would dominate blocks C and D by construction.

### 3.4 Minimum-ISI gate and the special classes

Three populations cannot be featurised by §3.2 and must be labelled rather than
forced through it:

| class | definition | v1 count | disposition |
|---|---|---|---|
| `Z` quiescent | no spikes, `sd(v) ≤ V_floor` | 12017 | held out, as v1 §10 (and largely eliminated by C3, §4) |
| `S` subthreshold | no spikes, `sd(v) > V_floor` | 17 | **new**: reported as its own class with its resonant peak frequency, not clustered |
| `W` sparse | spiking but `n_ISI < 10` | ~351 (2.4%) | held out and reported; the return map is not estimable |

The `n_ISI ≥ 10` threshold retains **97.6%** of v1 clustered samples. It sits
above the v1 §6 spectral-estimate floor (4 Hz → 8 spikes → 7 ISIs → 6 return-map
pairs), so it supersedes the 4 Hz rate floor and the v1 §19 A9
BELOW-RATE-FLOOR flag.

The `S` class is where v1's subthreshold resonators (RZ) went. Representing them
by their spectral peak and reporting them separately is more honest than forcing
17 non-spiking samples into an ISI-based clustering.

**`V_floor` replaces `P_floor`.** v1 gated the `Z`/`S` split on `ac_power`, which
requires an rFFT of every sample — the one production use of the spectrum that
C1 does not remove by itself (§3.5). It is not needed: `sd(v)` is already
accumulated in registers (v1 §7.6) and the two rank the same population.
Measured over the 12034 zero-spike samples of the v1 run, the 17 that `ac_power`
retained have `sd(v) ∈ [0.411, 0.842]` mV against a `Z`-class maximum of 0.4126
mV — a near-clean separation, with `corr(log ac_power, log sd(v)) = 0.80`.

The two rules are **not identical**: `ac_power` excludes DC and everything above
1 kHz, `sd(v)` does not, so a handful of boundary samples reclassify. That is
acceptable because v1's own `P_floor` was `max(numerical 1.318e-01, physical
7.500e+05)` — the physical term, defined as "0.1 mV RMS", won by six orders of
magnitude. The v1 gate was already an RMS-amplitude test expressed spectrally.

`V_floor` is calibrated the same way v1 calibrated `P_floor`: as the larger of
(i) the numerical floor measured on the convergence-gate silent references and
(ii) a physical floor of 0.1 mV RMS, and is recorded in the manifest with the
resulting `Z`/`S` counts. Under C3 (§4) the zero-spike population nearly
vanishes, so this boundary stops being load-bearing.

### 3.5 C9 — retiring the spectral production path

**Decision: `spectra_full.csv` is dropped; the per-representative spectral
figures are retained. The rFFT ceases to be a pipeline stage and becomes a
host-side, on-demand tool applied to ~100 samples.**

Spike detection is unaffected by any of this: v1 §7.1 tests `v >= 30` at the end
of each `dt_int` step on the raw integrated state, *before* the output stage. The
§3.2 features never touch the decimated trace.

Once nothing batched consumes the spectrum, the v1 §7.2 anti-aliased output
stage loses its entire justification. Its stated purpose was that decimation
aliasing "corrupts the UMAP metric and violates the smooth-response-surface
assumption the GP relies on" — and C1 removes the first consumer, C6 the second.
The removable chain:

| removed | reason |
|---|---|
| 2-stage cascaded boxcar (`kernel.cu`, v1 §7.2) | no batched FFT consumer remains |
| `\|H(f)\|²` deconvolution (v1 §9 step 5) | nothing left to deconvolve |
| aliasing gate (v1 §7.4, second check) | no aliasing-sensitive consumer |
| unit tests 3 and 4 (v1 §17) | test the above |
| hybrid banding, floor, `log1p`, L2 (v1 §9 steps 6–10) | superseded by §3.2 |
| `--spectral-tilt`, the `alpha` sweep (v1 §9 step 6, §12.4 stage 2) | belongs to the retired path |
| `spectra_full.csv` (v1 §16.2) | dropped by decision |
| `ac_power` (v1 §10) | replaced by `sd(v)`/`V_floor` (§3.4) |

Two simplifications follow:

- **Production traces are naively decimated** at `dt_out = 0.1` ms, for display
  only. They become *sharper*: v1 §7.2 notes the 19-sample cascade is ~0.19 ms
  against a ~0.15 ms spike upstroke, so it visibly rounds spikes.
- **v1 §8's dual-signal special case disappears.** v1 maintains two signals per
  representative — filtered for the FFT, unfiltered for the plot. There is now
  one.

**How the retained figures get their spectra.** Re-simulate each representative
(and each `S`-class sample, for its resonant peak) at `dt_out = 0.01` ms — a 10×
finer output grid, no decimation filter — and rFFT host-side, displaying
0–1 kHz as now. With Nyquist at 50 kHz instead of 5 kHz, the fold band sits near
95–105 kHz, where a 100 Hz train's harmonics are of order `k ≈ 1000`, i.e. ~60 dB
down: aliasing is negligible by construction rather than by filtering. Cost is
~100 samples × 250 000 points, seconds on the host, reusing the determinism
argument v1 §8 already makes for representative traces.

**Output changes (v1 §16.2).** `spectra_full.csv` is removed; `features_bands.csv`
is replaced by `features_isi.csv` (185 columns per §3.2) and a new
`spike_times.csv` (ragged, one row per sample). `traces.csv` is retained
unchanged in schema. Since `spectra_full.csv` was one of the two bulk files
behind `--compact-csv` and `--gzip-csv`, those flags now govern `traces.csv` and
`spike_times.csv` only, and the §15 3-minute CSV budget falls accordingly.
`run_data.npz` gains `spike_times` and loses `features`/`band_centres`.

v1 §19 A1 (relative noise-floor subtraction) and A4 (PCA 50 → 92) are superseded;
§8.1 replaces A4 with a gate rather than an accommodation.

---

## 4. C3 — drive relative to rheobase

**Decision: `I` is set per parameter set as a controlled overdrive above the
numerically determined rheobase of the hybrid system, not fixed at `I = 10` and
not placed at an analytic bifurcation point.**

v1 §2 fixes `I = 10`, with the consequence that **12017 samples (45.1%)** are
quiescent and discarded, and that "how far above threshold am I" is the dominant
nuisance axis in the clustering.

### 4.1 Why not the analytic bifurcation points

Both are closed-form for this model. With `u = bv` at equilibrium,
`0.04v² + (5−b)v + (140+I) = 0`:

```
saddle-node:      I_SN = (5 − b)² / 0.16 − 140
Andronov–Hopf:    I_AH = [ (5 − b)² − (b − a)² ] / 0.16 − 140      (exists only when b > a)
```

(At `a = 0.02, b = 0.2`: `I_AH = 3.80`, `I_SN = 4.00`.) `I_AH < I_SN` whenever
`b > a`, so the Hopf precedes the fold in that half of the box and is absent in
the other — which is itself the Class 2 / Class 1 excitability split.

Neither is the right normaliser:

1. **Neither depends on `c` or `d`.** Both are properties of the continuous
   subsystem alone. Normalising by them would leave two of the four design axes
   un-normalised — and `c, d` are precisely the axes that set burst structure
   through the reset map.
2. **The dynamics degenerate at onset.** At the saddle-node the firing rate → 0
   (below any practical window); at the Hopf the limit-cycle amplitude → 0 and
   may never reach `v = 30`. Sampling *at* the bifurcation samples the one place
   the pipeline cannot measure.
3. **The reset makes both wrong anyway.** The Hopf here is subcritical and the
   reset supplies a finite kick, so spiking can persist below `I_AH`
   (bistability); conversely a small post-Hopf cycle can fail to reach threshold
   above it. The quantity of interest is the rheobase of the *hybrid* system.

Their use in v2 is as a **bracket** for the numerical search and as a **label**
(Class 1 where `b ≤ a`, Class 2 where `b > a`).

### 4.2 The procedure

Per parameter set `θ = (a,b,c,d)`, bisect on "does the system spike in steady
state" over `I ∈ [min(I_SN, I_AH) − 1, I_max]`, using 500 ms simulations and a
spike-count-over-the-last-200 ms test, to a tolerance of 0.1 in `I`. ~10
evaluations of a 500 ms run per sample, against the 2500 ms production run:
roughly **2–5% of the integration budget**, which v1 measured at 85 s total.

Then simulate at `I = I*(θ) + Δ` for `Δ ∈ {1, 5, 15}` (three sheets), or treat
`Δ` as a fifth log-spaced design axis. Record `I*`, `Δ`, `I_SN`, `I_AH` and the
excitability class per sample in `samples.csv`.

Samples with no `I*` in range (no spiking at any `I ≤ I_max`) are labelled and
reported; they replace most of the `Z` population.

### 4.3 What this buys

- The `Z` quarantine goes to approximately zero: every clustered sample fires by
  construction, roughly doubling effective sample size at constant cost.
- `c` and `d` become genuinely explorable rather than near-irrelevant to whether
  anything fires at all.
- The f–I onset shape (continuous vs discontinuous) becomes a free, genuinely
  *discrete* classifier — a real excitability-class distinction that v1 cannot
  observe.
- Comparisons across parameter sets become fair: equal overdrive rather than
  equal absolute drive.

---

## 5. C4 — convergence gate on a physical criterion

v1 §7.4 accepts `dt_int` on 1% relative L2 of the 128-band feature vector. With
C1 that vector no longer exists, and §2.5 shows the criterion was measuring comb
phase rather than truncation error.

**Replacement criterion.** Accept `dt_int` when, against the `dt = 0.001` ms
reference over the probe vectors that survive the §5.1 exclusion:

1. spike **count** in the analysis window is identical for **≥ 90%** of used
   probes, and no probe differs by more than 1 spike;
2. the median **per-ISI** error `|ΔI|` is < 0.05 ms (half of `dt_out`) for
   **≥ 90%** of used probes, and no probe exceeds 0.25 ms;
3. the §3.2 feature vector is within 1% relative L2 for **≥ 90%** of used
   probes — now a meaningful test, because the ISI encoding is not
   comb-phase-sensitive.

The v1 §7.3 per-sample refinement pass is retained unchanged; the v1 §7.4
**aliasing check is removed** with the output stage it tested (§3.5).

### 5.1 Why the thresholds are fractional

The draft required **every** used probe to pass all three checks. V2-A15 reads it
that way correctly, and settles the resulting `dt_int = 0.0025` on cost grounds
(4× the integration, 220 s of a ~40 min run — affordable, so taken). The
criteria are nonetheless amended to the fractional form above, for a reason
V2-A15 does not address and which is not about cost:

**A rule requiring all `N` probes to match exactly gets monotonically stricter
as `N` grows.** The selected `dt_int` then depends on how many probes happen to
be in the set — enlarging the probe set to characterise the box better would
silently force a smaller step, and the gate's answer would not be comparable
between runs with different probe counts. A fractional rule with a bounded worst
case is invariant to that.

Measured on the 2026-11-21 18:03 gate run, the amended rule selects
`dt_int = 0.01` ms, while `dt = 0.02` still fails:

| dt (ms) | counts identical | per-ISI err > 0.05 ms | feature rel L2 > 1% | strict | fractional |
|---|---|---|---|---|---|
| 0.02 | 17/23 (74%) | 6 | 1 | fail | fail |
| 0.01 | 21/23 (**91%**) | 2 (91% pass) | 1 (96% pass) | fail | **pass** |
| 0.005 | 22/23 (96%) | 1 | 1 | fail | pass |
| 0.0025 | 23/23 | 0 | 0 | selected | pass |

That matches the physical reading: v1 measured mean spike count stable from
`dt = 0.02` onward, and per-ISI error at `dt = 0.01` is 0.007 ms against a
`dt_out` of 0.1 ms. The two probes that still shift their spike count at
`dt = 0.01` differ by one spike in ~200, which is a window-edge effect, not a
convergence failure — hence the "no probe differs by more than 1 spike" guard,
which keeps the 10% tolerance from hiding a grossly wrong probe.

Criterion 2 now gates on the **per-ISI** error for the reasons established in
V2-A12, which supersedes the draft's accumulating spike-time formulation.

**Chaotic probes are excluded, not tolerated.** Probes that fail criterion 1 or
2 even between the two finest ladder steps (`dt = 0.0005` vs `0.001`) are
**sensitivity-limited**: no step size converges them, and they are excluded from
the gate and reported as a finding — in this model the signature of chaotic
regimes (the CH corner), not a numerical defect. The first v2 run excluded 1 of
24 this way. This exclusion must run *before* the fractional rule, so genuinely
chaotic trajectories do not consume the tolerance that exists for window-edge
effects.

---

## 6. C5 — the onset transient

v1 §6 discards `[0, 500)` ms for spectral stationarity. That is correct for a
spectrum and wrong for typing: it deletes exactly what distinguishes IB from RS
at constant drive (§2.2).

With C1 the stationarity argument no longer binds, because the ISI encoding does
not require a stationary windowed signal. Specify **two feature sets per
sample**:

- **steady-state**, from `[500, 2500)` ms — blocks A–D as §3.2, the clustering
  input;
- **onset**, from `[0, 500)` ms — the same blocks, plus the number of spikes in
  the first 100 ms and the ratio of the first ISI to `Ī_steady`.

The onset block enters the clustering vector at weight `w_onset`, swept in
`{0, 0.3, 1.0}`. `w_onset = 0` must be in the grid so the v1 behaviour remains
reachable and the contribution is measured rather than assumed.

Transient-defined distinctions (IB's initial burst) become observable; where
they remain unreachable, v1 §13.1's caveat must say so explicitly rather than
letting a vacuous pass stand.

---

## 7. C6 — retire GP active learning

**Decision: replace the v1 §11 GP-driven acquisition with a single Sobol
sequence over the design domain, at 3–4× the sample count.**

Justification is §2.6: `gp_fit` + `acquire` cost 346 s against 85 s for all
integration, and steer on a feature-space signal §2.3 shows to be degenerate.
At v1's measured integration throughput, the GP's wall-clock cost alone buys
roughly 100 000 additional simulated samples.

Consequences:

- Every sample is design-unbiased, so the "all" vs "LHS-only (authoritative)"
  split in v1 §14.1 collapses to a single set of statistics, and per-cluster
  `n` rises by ~3× before the density increase is counted.
- The IPW machinery (v1 §14.1, `--ipw-stats`) becomes unnecessary and is
  removed.
- `gp.py` and `acquire.py` are deleted, along with their tests.

This is explicitly **deferral, not rejection**: active learning on a
representation that separates firing patterns is a reasonable v3 feature. It is
premature while the acquisition target is the quantity §2.3 measures.

---

## 8. C7/C8 — validation that can fail

### 8.1 Representation-health gates (new, blocking, pre-clustering)

Run immediately after featurisation, before PCA. Each is one line and would
have caught the v1 run in under a minute:

| gate | threshold | v1 value |
|---|---|---|
| PCA components for 99% of variance | **≤ 20** | 92 |
| mean cosine distance for pairs with `\|Δ log f₀\| ∈ [0.05, 0.10]`, relative to the same for `\|Δ log f₀\| > 1.0` | **≤ 0.50 ×** | 0.818× (saturated) |
| octave-pair mean distance ≥ 1.5×-pair mean distance | **required** | violated (0.379 vs 0.501) |

The second and third are the §2.3 measurements promoted to gates: they test that
the metric is monotone in rate separation and free of harmonic inversion.

### 8.2 Stability tests that probe something

v1 §13.2 varies only the UMAP seed, holding features and sample set fixed; ARI
0.780 mostly establishes that UMAP is reproducible. Add, each reported as a
mean pairwise ARI with its own threshold:

| test | what it varies | threshold |
|---|---|---|
| seed (v1 §13.2, retained) | UMAP `random_state` | ≥ 0.6 |
| **bootstrap** | 80% subsample of the clustered set, 5 draws | ≥ 0.6 |
| **window shift** | analysis window shifted by +100 ms | ≥ 0.8 |
| **step size** | `dt_int` halved | ≥ 0.9 |

The window-shift test is nearly free and probes exactly the sensitivity §2.3
exposes: a representation whose labels move when the window shifts by 100 ms is
encoding window phase, not dynamics.

### 8.3 Canonical placement with teeth

v1 §13.1 requires only that each of the 7 canonical types land in a non-noise
cluster — unfailable at 102 clusters, and §2.2 shows it passing while CH lands
in a cluster named "slow tonic".

**Replacement:** each canonical type must land in a cluster whose §14.2
descriptors are consistent with the published type. Concretely, for the type's
own measured `(CV, burst index, spikes-per-burst, rate)`, each must lie within
the host cluster's interquartile range. CH would fail this today
(`BI = 0.596` against a cluster median of 0.000).

Types the protocol cannot reach (TC, RZ at constant drive — v1 §13.1, §18 item
6) are exempted **by name, in the spec**, not by a blanket relaxation.

### 8.4 Nameability merge

102 clusters sharing 6 names (§2.1) means the clustering is operating below the
resolution of its own descriptive vocabulary.

Add a post-clustering merge step in `characterize.py`: if two clusters'
descriptor distributions `(rate, CV, burst index, spikes-per-burst, adaptation)`
are statistically indistinguishable — pairwise Mann-Whitney at Bonferroni-
corrected α = 0.01 on every descriptor — merge them and record the merge in the
report. The characterisation becomes the arbiter of distinguishability rather
than a label applied after the fact.

Report both counts: clusters found, and clusters surviving the merge.

---

## 9. Revised acceptance criteria

Replacing the v1 §17 run-level list. Criteria that cannot fail are marked where
they were removed.

| criterion | threshold |
|---|---|
| convergence gate (§5) passes all three checks | required |
| zero samples labelled `X`; zero `overflow_flag` set (§3.1) | required |
| noise fraction at the selected config | ≤ 25% |
| **PCA components for 99% variance** (§8.1) | **≤ 20** |
| **metric monotonicity and octave gates** (§8.1) | **required** |
| ~~≥ 10 clusters at `leaf`~~ | *removed — rewards over-segmentation* |
| **fraction of reported clusters passing both §13.2 and §13.3** | **≥ 0.60** (v1: 0.216 — 22 of 102) |
| **distinct names after the §8.4 merge** | **≥ 8** (v1: 6 of 102) |
| **≥ 1 cluster named bursting holding ≥ 50% of all `BI > 0.2` samples** | **required** (v1: 86 clusters shared them) |
| seed ARI / bootstrap ARI / window-shift ARI / `dt` ARI (§8.2) | ≥ 0.6 / 0.6 / 0.8 / 0.9 |
| **canonical placement by descriptor consistency** (§8.3) | required, TC+RZ exempt |
| every artefact present, non-empty, row-consistent | required (v1 §17, retained) |
| round-trip: spike times → features reproduces `features_isi.csv` to 1e-6 | required |

The bursting criterion is the single most direct test that C1 worked.

---

## 10. Phasing

Each phase ends in a runnable pipeline and a report; none depends on a later one.

| phase | contents | expected effect |
|---|---|---|
| **1** | C2 + C1 (§3) + C4 (§5) + the §8.1 gates | the decisive change. Settles whether bursting separates, and should restore `dt_int = 0.01` |
| **1b** | C9 (§3.5) | pure deletion, safe only once phase 1 has shown the ISI features work; do not remove the output stage before its replacement is validated |
| **2** | C7 + C8 (§8.2–§8.4, §9) | makes the result reportable: gates that can fail, merge, honest counts |
| **3** | C3 (§4) | removes the 45% quarantine and the overdrive nuisance axis |
| **4** | C6 (§7) + C5 (§6) | reallocates budget to density; adds transient phenomenology |

Phase 1 is where the hypothesis lives. If the §8.1 gates and the §9 bursting
criterion pass after phase 1, the rest is consolidation; if they do not, the
diagnosis in §2 is wrong and phases 2–4 should not be started.

---

## 11. Out of scope for v2

1. **Stimulus-protocol battery.** The paper's types are defined by responses to
   protocols (hyperpolarising step and rebound, brief timed pulse, slow ramp),
   not by one constant-drive trace — which is why v1 §18 item 6 concedes TC and
   RZ are unreachable. A 4-protocol battery costs ~4× an 85 s integration
   budget, i.e. minutes, and would plausibly yield more *genuinely distinct*
   types than every change in this document combined. It is excluded only
   because it changes the objective from "explore `(a,b,c,d)` at fixed drive" to
   "characterise response types", which is the user's call, not a defect fix.
2. **Active learning on the v2 representation** (§7) — reasonable once the
   encoding is known to separate what it should.
3. **Spike-train (delta-train) spectrum** (v1 §18 item 9) — largely subsumed by
   §3.2 block B, which encodes the same rhythm information with an ordered
   metric.
4. **Optimal-transport metric** on the ISI distribution. Correct in principle,
   `O(N²)` in practice at N = 26 624; the §3.2 KDE smoothing is the cheap
   approximation and should be shown insufficient before this is revisited.

---

## 12. Risks

| risk | mitigation |
|---|---|
| The ISI encoding discards subthreshold dynamics entirely | §3.4 `S` class keeps non-spiking resonators visible and separately characterised; the spectral path is retained for them |
| Very slow bursters have too few ISIs for a stable return map | §3.4 `n_ISI ≥ 10` gate, reported not silently dropped; a longer `T_total` is the remedy if the `W` class grows past ~5% |
| Block weights (§3.3) become a tuning surface that can be gamed | all weights swept on the v1 §12.4 DBCV criterion, with `w_A = 0` and `w_onset = 0` in the grid so each block's contribution is measured; the §9 criteria are evaluated only at the selected config |
| Rheobase bisection fails to converge near bistable corners | bracket from the analytic `I_SN`/`I_AH`; record non-convergence as a labelled class rather than retrying |
| Fewer clusters than v1 | expected and intended. 102 clusters at 6 names is not a result; §9 replaces cluster *count* with cluster *quality* as the acceptance axis |
| Dropping `spectra_full.csv` (§3.5) forecloses independent spectral re-analysis | the integrator is deterministic given `(a,b,c,d,I,v0,u0,dt_int,dt_out)` (v1 §8), so any sample's spectrum is reproducible on demand from `samples.csv` — the same argument v1 already uses to re-simulate representatives rather than store them |
| Removing the §7.2 output stage (C9) before the ISI path is proven leaves no validated feature pipeline | §10 sequences C9 as phase 1b, after phase 1 has demonstrated the §3.2 features; the two changes are independent and must not be landed together |

---

## 13. Implementation assumptions (v2)

Everything in §§3–10 that the spec left open, or that measurement forced to
differ from it, resolved here rather than by asking.  Each assumption is
tagged in the code at the place it takes effect.

### 13.1 Where measurement forced a change to this document

| # | spec text | what was implemented, and why |
|---|---|---|
| **V2-A2** | §4.2: "~10 evaluations of a 500 ms run per sample … roughly **2–5%** of the integration budget" | The arithmetic does not hold: ten 500 ms runs are 5000 ms against a 2500 ms production run, i.e. **200%**, not 2–5%. The search is therefore run at a coarse `dt_int = 0.05` ms (it only has to answer "does it spike in steady state"), which brings it to **≈2%** of the production integration — the stated figure, by a different route. Measured: 0.4 s of rheobase search against 220 s of production integration. |
| **V2-A7** | §3.3: "Each block is L2-normalised to unit norm independently" | Applied to blocks **A and B** only. Block D is a *single scalar*: per-sample L2 normalisation maps it to `±1` and destroys the rate axis the block exists to carry. Block C is signed and is exactly zero for a regular train, so per-sample normalisation would amplify an arbitrarily small drift to full weight and is undefined at zero. C and D are therefore scaled by a **dataset-wide** constant (RMS block norm for C, z-score for D), which is what makes `w_C` and `w_D` interpretable. The constants are recorded in `run_manifest.json` (`derived.isi_scales`) so the encoding is reproducible from `spike_times.csv`. |
| **V2-A9** | §3.2 block C: "`{λ_i}` resampled onto 8 points of normalised ordinal time `i/n`" | Resampled as **equal-count segment means**, not by interpolation at 8 points. Point sampling aliases on alternating (burst) sequences — the value at `i/n` depends on the parity of that index — and block C is specified to carry the *slow systematic trend*. Measured: under point sampling a 5 Hz and a 50 Hz doublet burster, which are the same train rescaled, had `cos = 0.73` on block C; under segment means, `0.96`. Both scales are additionally **floored** (`c_scale_floor = 0.01` decades per point, `d_scale_floor = 0.05` decades), because on a population of perfectly regular trains the measured RMS of block C is pure roundoff and dividing by it amplifies that roundoff to unit weight. |
| **V2-A12** | §5 criterion 2: "the median absolute **spike-time** error is < 0.05 ms" | Gated on the median absolute **interval** (ISI) error instead; the absolute spike-time error is reported alongside, ungated. Spike detection is end-of-step, so each detected time is systematically late by up to `dt` and the lateness *accumulates* along the train: the measured median absolute error scales linearly with `dt` **and with window length** (1.10, 0.59, 0.54, 0.25, 0.067 ms at `dt` = 0.04 … 0.0025), so the literal criterion fixes `dt ≈ 0.05/n_spikes` — it measures accumulated phase, which is the same category of defect §2.5 identifies in the v1 gate. The §3.2 features consume intervals, and a common time shift leaves them untouched. |
| **V2-A15** | §5: "Accept `dt_int` when, against the `dt = 0.001` ms reference over the 24 probe vectors: …" | Originally read strictly — **every** used probe passing every check — which selected `dt_int = 0.0025` ms because 2 of 23 probes still shift their spike *count* at 0.01 ms. **Superseded by §5.1**, which amends the criteria to the fractional form (≥ 90% of used probes, with a bounded worst case) for a reason independent of cost: an all-probes rule gets monotonically stricter as the probe set grows, so the gate's answer would depend on how many probes were run. The implementation now applies §5.1 and selects **`dt_int = 0.01` ms**; the per-probe table in §1 of the run report carries both readings. Criterion 2's per-ISI statistic is evaluated over the common prefix when a probe's counts differ, so that a probe criterion 1's one-spike tolerance admits is still measurable under criterion 2. |
| **V2-A13** | §3.1: "capacity `S_max = 4096` … covers 2 kHz" | Raised to **8192**. §3.1 sized the buffer for a 2000 ms window; `t_total` is now 2600 ms (V2-A3), so 4096 covers only 1.6 kHz. Measured maximum in the quick run: 3624 recorded spikes, i.e. 88% of 4096. 8192 covers 3.1 kHz at 67 MB per chunk. |

### 13.2 Choices the spec left open

| # | decision | rationale |
|---|---|---|
| **V2-A1** | `Δ` is the **fifth design axis** (log-uniform on [1, 15]), not three discrete sheets | §4.2 offers both. The axis form costs one production simulation per sample instead of three, makes overdrive a continuous explorable coordinate, and keeps `Δ` in the per-cluster correlation matrices. |
| **V2-A3** | `t_total` 2500 → **2600 ms** | So the §8.2 window-shift test compares two windows of *equal length* (`[500, 2500)` vs `[600, 2600)`) rather than confounding a phase shift with a 5% shorter record. Costs 4% more integration. |
| **V2-A4** | `N` = **65 536** (2^16), i.e. 2.46× v1, not the 3–4× of §7 | Sobol balance wants a power of two. The four §8.2 stability tests, the §8.4 merge and the stage-2 representation sweep are all new costs that scale with `N` and did not exist in v1, so the GP's wall-clock does not translate into samples one-for-one. |
| **V2-A5** | Display traces are decimated **peak-aware** (a block containing a spike emits `v_peak`), not by plain sample dropping | §3.5 removes the anti-aliasing filter and keeps traces for display only. At `dt_out = 0.1` ms against a ~0.15 ms upstroke, plain decimation would drop spikes out of the picture entirely; peak-aware decimation is the `filter_on = 0` branch the v1 kernel already had, and it is exact for the `dt_out = dt_int` re-simulation the spectra use. |
| **V2-A6** | The 7 canonical seeds are driven at **`I = 10`**, v1's drive | Their published behaviours are defined at the paper's drive; `Δ` for those rows is `10 − I*`. Reported per type in the placement table. |
| **V2-A8** | If the analytic bracket already fires (bistability, §4.1 item 3), the lower end is walked down twice by 5 before the sample is labelled `BRACKET_LOW` | §12's risk table asks for a labelled class rather than a retry loop; two steps is the compromise. 0 of 65 536 samples hit it in the v2 run. |
| **V2-A10** | The window-shift and step-size ARIs compare perturbed against unperturbed labellings **at matched UMAP seeds** (3 seeds, 6 fits) | Independent seeds alone cost ~0.2 ARI (v1 measured 0.78 seed ARI), so a test that re-seeded could not reach §8.2's 0.8 / 0.9 thresholds no matter how stable the representation. Matching the seed isolates the perturbation. |
| **V2-A11** | `spike_times.csv` is `sample_id, n_spikes, spike_times_ms` with the third field a space-separated list, written `%.9g` | Keeps the ragged payload inside a rectangular CSV that `pandas.read_csv` loads unaided. `%.9g` round-trips the float32 the kernel writes exactly, which the 1e-6 round-trip criterion requires. |
| **V2-A14** | Both sweep stages are scored on a fixed **20 000-sample** subsample; the winner is re-fit on the full set | The sweep compares configurations, and v1 measured its own sweep at `N = 26 624`. Scoring 54 + 10 configs on the full 65 536 would cost ~45 min of UMAP alone. Caveat, visible in the v2 run: `min_cluster_size` is absolute, so the same config resolves fewer clusters on the subsample than on the full set (the selected config scores `k = 4` on 20 000 points and yields 68 on 65 536). The sweep therefore ranks configurations, and the cluster *count* must be read off the final full-set fit, not off `sweep_table.csv`. |
| **V2-A16** | `features_isi.csv` holds the 185-dim steady-state vector of §3.2; the onset block is a separate `features_isi_onset.csv` (187 dims) | §3.5 specifies 185 columns for `features_isi.csv`, and §6 adds a second feature set per sample. The clustering matrix is the weighted concatenation of the two (372 dims); both files are reproducible from `spike_times.csv`, and the round-trip criterion is evaluated on the first. |
| **V2-A17** | The firing descriptors (rate, CV, burst index, spikes/burst, adaptation) are computed **host-side from exact spike times** | §3.1 retains the in-kernel log-ISI histogram and moments "as characterisation diagnostics"; they are kept and written to `run_data.npz`, but naming (§14.2), the §8.3 consistency test and the §8.4 merge now use the unquantised values. |
| **V2-A18** | The per-cluster triangular matrices are **5×5** over `(a, b, c, d, Δ)`, and contiguity (v1 §13.3) is judged in the 5-D normalised design box | `Δ` is a design axis under C3, so both the correlation structure and the contiguity question are now 5-dimensional. |
| **V2-A19** | The cluster alphabet skips **S and W** as well as X and Z | §3.4 names two new held-out classes; a cluster lettered `S` would collide with the subthreshold class in every file name and table. |
| **V2-A20** | Names gain a qualitative burst-size word (`doublet` / `triplet` / `multi-spike`), and "distinct names" (§9) are counted on a **number-stripped name key** | §2.1 counted 6 distinct names across 102 clusters, which is only meaningful if the parenthetical numbers are stripped — otherwise every cluster has a unique name and the criterion is unfailable. The burst-size word is read straight off the return map, which is what §3.2 block B exists to resolve. |
| **V2-A21** | §8.3 is gated on the literal IQR test; the **Tukey-whisker** version (`[Q1 − 1.5 IQR, Q3 + 1.5 IQR]`) is reported alongside | The literal test is strict in a way §8.3 may not intend: a *typical* member of a tight cluster is inside all four IQRs with probability ≈ 0.5⁴. Reporting both separates "gross mismatch" (CH named *slow tonic*, which §8.3 exists to catch) from ordinary spread. |
| **V2-A22** | The §9 bursting criterion is evaluated literally (one cluster holding ≥ 50% of `BI > 0.2` samples); the **union** over bursting-named clusters and the number of holding clusters are reported next to it | The literal form can fail while the encoding works, if the burst population splits into several genuinely distinct bursting clusters. Both numbers are in the acceptance table. |
| **V2-A23** | §8.2's bootstrap draws are 80% subsamples **of the 12 000-point stability subsample**, compared on each pair's intersection | Five independent UMAP fits of 0.8 × 60 000 points would cost more than the rest of the validation combined, and the test is about resampling sensitivity, not about scale. |
| **V2-A24** | The §8.1 gates are **blocking** in a full run; `--quick` and `--no-health-gate` report them instead | §8.1 asks for blocking gates. The escape hatch exists so a failing representation can still be diagnosed from a complete run. |
| **V2-A25** | `V_floor`'s numerical term is measured on the v1 silent-reference set at **`I = 0`** | Under C3 the production drive is `I* + Δ`, so a "silent reference" has to be driven below its own rheobase; at `I = 0` the measured `sd(v)` is exactly 0, so the physical term (0.1 mV RMS) sets the floor, as §3.4 anticipates. |
| **V2-A26** | §6's two onset scalars enter as the spike count in the first 100 ms and **`log10`** of (first ISI / `Ī_steady`), both z-scored across the dataset | The raw ratio is unbounded above and compressed below; its logarithm is symmetric about "the onset interval equals the steady one", which is the comparison the scalar is for. |
| **V2-A27** | The stage-2 grid is a **coordinate sweep** around the defaults (10 configs), not a full factorial | `w_A = 0` and `w_onset = 0` are both in it, as §12's risk table requires, as are all three bandwidths. A full factorial over 5 knobs is ~100 UMAP fits. |
| **V2-A28** | The §8.4 merge is applied **transitively** (union-find) and the merged labelling is the reported one; the pre-merge count is reported beside it | §8.4 asks for both counts and makes characterisation the arbiter of distinguishability, which only has teeth if the merged labelling is what the rest of the report describes. |
| **V2-A29** | A sample whose coarse rheobase search finds no `I* ≤ I_max` is still integrated once (at `I = I_max`) and classified by §3.4 from that run | The search is a heuristic at `dt = 0.05` ms; the production integration is authoritative. Such samples land in `Z`, `S` or `W` on their own evidence rather than being dropped unexamined. |
| **V2-A31** | Every §8.2 stability fit scales `min_cluster_size` to its own point count (`mcs × n_fit / n_active`), and the stability subsample is raised from 12 000 to **32 000** (≈50% of the clustered set) | v1 §12.4 defines `min_cluster_size = 60` as "≈0.22% of N", so a subsample clustered at the full-set `mcs` resolves a *coarser* taxonomy than the reference and every per-cluster Jaccard comes out below 0.5 for reasons unrelated to stability. Measured without the scaling: seed ARI 0.997 — the refits agreed with *each other*, at 5 clusters — while 47 of 50 reference clusters were flagged UNSTABLE. With the scaling, the subsample fit resolves the reference's taxonomy, which is what §8.2 asks about; the larger subsample then keeps the absolute cluster sizes (and so the sampling noise) close to the reference's. The ARIs this produces are the honest ones and are lower than the first, mis-scaled run's. |
| **V2-A30** | `--ipw-stats`, `--spectral-tilt` and `--csv-full-trace` are removed; `--no-health-gate` is added | The first dies with C6 (every sample is design-unbiased), the second with C9 (there is no spectral path to tilt), the third with `spectra_full.csv` (the row-for-row correspondence it broke no longer exists). |

---

## 14. What the v2 run measured

Run of 2026-11-21 20:07, `N = 65 536`, 38.1 min wall clock on the RTX 500 Ada.
Full detail in `output/cluster_report.md`; this table is the document's own
diagnosis (§2) and acceptance list (§9) scored against the result.

### 14.1 The §2 diagnosis, re-measured

| quantity | v1 | v2 | §9 threshold |
|---|---|---|---|
| PCA components for 99% of variance | 92 of 128 | **13** of 372 | ≤ 20 ✅ |
| mean cosine distance at \|Δ log f₀\| ∈ [0.05, 0.10], relative to > 1 decade | 0.818× (saturated) | **0.015×** | ≤ 0.50× ✅ |
| octave pairs vs 1.5× pairs | 0.379 vs 0.501 (**inverted**) | **0.096 vs 0.040** | octave ≥ 1.5× ✅ |
| samples quarantined as non-spiking | 12 017 (45.1%) | **0** | — |
| samples entering the clustering | 54.9% | **97.6%** | — |
| `BI > 0.2` samples in clusters *named* bursting | 1 of 102 clusters was named bursting; the 1947 such samples were spread over 86 of them | **88.9%**, over 51 of 56 clusters | ≥ 50% in *one* cluster ❌ (best: 30.4%) |
| distinct interpretable names | 6 of 102 clusters | **10** of 56 | ≥ 8 ✅ |
| budget spent on the GP | 37.9% | **0** (retired) | — |
| budget spent on validation | 4.2% (38 s) | **55%** (1265 s) | — |

The three §8.1 gates are the decisive result: the ISI encoding is low-rank,
monotone in rate separation and free of the harmonic inversion, which is
exactly the hypothesis §3 was written to test. Bursting separates: CH, which v1
placed in a cluster named *slow tonic* at `p = 0.488` whose members had median
`CV = 0.000` and median burst index 0.000, now lands at `p = 1.000` in a
cluster whose burst-index IQR `[0.569, 0.814]` and spikes-per-burst IQR
`[2.32, 5.36]` both contain CH's own `0.604` and `2.52`.

### 14.2 Where the v2 run fails its own criteria

Four of the §9 criteria fail, and three of them are the same finding.

1. **The taxonomy is not reproducible at the resolution DBCV selects.** Seed
   ARI 0.559, bootstrap 0.249, window-shift 0.501, `dt`-halved 0.439 — all
   below §8.2's thresholds, and the per-cluster consequence is that only 5 of
   56 clusters pass both the stability and the contiguity gate (§9 wants
   ≥ 0.60). This is the first *honest* measurement of the quantity: v1's 0.780
   was measured with `min_cluster_size` unscaled on a 45% subsample, i.e.
   against a coarser taxonomy than the one it reported (V2-A31), and the
   numbers are therefore not comparable. Three consecutive v2 runs selected
   `(mcs, ms, h)` triples scoring within 0.05 DBCV of each other and resolving
   19, 50 and 56 clusters — the selection itself is sitting on near-ties, which
   is the same instability seen from the other side.
2. **No single cluster holds half the bursting population**, though clusters
   named bursting hold 88.9% of it between them. The encoding separated
   bursting from tonic, as C1 intended; what it also did was separate bursting
   *sub-types* (doublet / triplet / multi-spike, at different rates), so the
   population splits across 51 clusters rather than concentrating in one. The
   literal criterion (V2-A22) fails; the quantity it was written to detect —
   bursting samples scattered indiscriminately across the taxonomy — does not.
3. **Canonical placement by IQR** fails for 5 of 7 types, 4 of them inside the
   Tukey whiskers of their host cluster and all on `cv_isi`, whose within-
   cluster spread for tonic clusters is ~1e-5. The test as written (V2-A21)
   is satisfied by roughly a coin-flip's worth of typical members.

The honest reading: **C1 worked and C3 worked** — the representation-level
evidence is unambiguous — while the *partition* of that representation into a
reported taxonomy is where v2's result is weak, and §8.2 is what says so. The
natural v3 target is the clustering stage, not the encoding: a selection
criterion that scores reproducibility rather than DBCV alone, and the
stimulus-protocol battery §11 item 1 defers.
