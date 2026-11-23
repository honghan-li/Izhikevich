# The anatomy of the ISI embedding — nine hand-picked regions

**Date:** 2026-11-22 · **Source run:** 2026-11-21 23:43, `N = 65 536`, ISI encoding
(`IMPROVEMENTS.md` C1–C9) · **Inputs:** `data/samples.csv`, `data/spike_times.csv`
· **Figures:** `output/regions_map.png`, `output/regions_return_maps.png`,
`output/regions_phase.png` · **Reproduce:**
`pixi run python scripts/region_analysis.py` (writes `data/regions.csv`, the
per-sample region label and every descriptor used below, plus the three figures)

---

## 0. The answer in one paragraph

The embedding is not a taxonomy of "types". It is a **bifurcation diagram of the
Izhikevich reset map**, drawn by UMAP. Every point in the right-hand *blob* has a
spike train whose interspike-interval sequence is a **fixed point** of that map —
one ISI, repeated forever (period 1). Every point in the left-hand *neuron* has a
train on a **higher-period orbit or a chaotic attractor** (period ≥ 2, or
aperiodic). The neuron's branches are the successive orbits: the short dendrite
and the axon are **period 2**, the long dendrite is the **period-3 window**, the
onion is **period ≥ 4**, and the cell body — the part HDBSCAN refused to label —
is the **chaotic core** where those branches meet. The blob's three lobes are not
different dynamics at all; they are three bands of one continuous firing-rate
axis. The single parameter that decides which side you land on is the reset
voltage `c`, modulated by the recovery kick `d`.

---

## 1. How the nine regions are defined

Regions were cut geometrically from the 2-D embedding, then checked against the
labels. The definitions below are exact and reproducible from `samples.csv`.

| super-structure | definition |
|---|---|
| **neuron** (15 895) | the connected components of the 2-D embedding that are *not* the largest one (grid-occupancy labelling at 0.30 UMAP units, 8-connectivity). 5 components: the main body plus 4 detached terminal filaments. |
| **blob** (48 095) | the single largest connected component. |
| *(held out)* (1 546) | class `W` (`n_ISI < 10`), no embedding coordinate. |

| region | n | definition | HDBSCAN letters | % noise |
|---|---|---|---|---|
| great blob | 39 828 | blob side, letter ≠ `A` | 47 letters (D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, U, V, AC…CE) | 20% |
| middle blob | 3 756 | letter `A`, `x − y ≤ 8.70` | A | 0% |
| end blob | 4 511 | letter `A`, `x − y > 8.70` | A | 0% |
| dendrite T | 881 | neuron side, letters T, BG | T, BG | 0% |
| axon | 3 742 | the trunk, `y ≲ 14.8` | AR, BB, AB, AA, Y, BO, AD, BI, BM, BV | 0% |
| axon terminals | 2 865 | the fan, `y > 14.8` | BA, BR, CB, CC, AP, AQ, CA, BJ, BY | 36% |
| dendrite C | 3 484 | letter C | C | 0% |
| dendrite B (onion) | 4 386 | letter B | B | 0% |
| cell body | 135 | unclustered points in the junction box `x ∈ [−6.35, −4.85]`, `y ∈ [−1.60, 0.45]` | — | 100% |

A further 402 neuron-side noise points lie on the trunk between labelled axon
segments; they behave identically to `axon` and are folded into it in the scoring
of §5.

The `middle | end` cut `x − y = 8.70` passes through the pinch at
`(2.70, −6.00)`, which is where the **TC** canonical seed (`#56800`) sits — the
boundary you identified, located numerically.

### 1.1 Descriptors used throughout

All measured host-side from exact spike times in the steady window
`[500, 2500)` ms:

- `p` — **ISI orbit period**: the smallest `p ≤ 16` with `|I_i − I_{i+p}| ≤ 0.02·Ī`
  for all `i`; `p = 0` means no such period exists (**aperiodic**).
- `f` — firing rate (Hz); `R = I_max / I_min` — the **ISI contrast** of the orbit.
- `H` — entropy (nats) of `log₁₀(I/Ī)` on the 32-bin `[−1.75, 1.0]` grid
  (the same grid as feature block A). A period-`p` orbit gives `H ≈ ln p`.
- `F` — spectral flatness of the delta spike train over 2–500 Hz
  (1 = white noise, 0 = a pure line spectrum).

---

## 2. Master table

Medians, with interquartile ranges where shown.

| region | n | rate (Hz) | CV | orbit period | aperiodic | `R = I_max/I_min` | `H` | burst idx | spk/burst |
|---|---|---|---|---|---|---|---|---|---|
| great blob | 39 828 | **28** (14–53) | 4.6e−5 | **1** | 0.0% | 1.0 | 0.00 | 0.00 | 1.00 |
| middle blob | 3 756 | **186** (158–226) | 1.7e−5 | **1** | 0.2% | 1.0 | 0.00 | 0.00 | 1.00 |
| end blob | 4 511 | **438** (346–602) | 3.9e−5 | **1** | 0.0% | 1.0 | 0.00 | 0.00 | 1.00 |
| dendrite T | 881 | **128** (86–204) | 0.38 | **2** (69%) | 22% | **2.4** (1.9–3.1) | 0.69 | 0.00 | 1.00 |
| axon | 3 742 | **58** (41–81) | 0.86 | **2** (99%) | 0.4% | **13** (7.6–20) | 0.69 | 0.50 | 2.01 |
| axon terminals | 2 865 | **15** (11–21) | 1.00 | **2** (100%) | 0.0% | **64** (44–97) | 0.69 | 0.52 | 2.07 |
| dendrite C | 3 484 | **50** (25–90) | 1.29 | **3** (99%) | 0.9% | **38** (19–81) | 1.10 | 0.67 | 3.05 |
| dendrite B (onion) | 4 386 | **70** (40–111) | 1.77 | **4–16** (94%) | 6% | **62** (32–120) | 1.56 | 0.76 | 4.13 |
| cell body | 135 | **140** (110–182) | 0.76 | **aperiodic** | **53%** | **7.3** (6.4–9.2) | **1.63** | 0.00 | 1.00 |

Design parameters (interquartile ranges):

| region | `a` | `b` | `c` (mV) | `d` | `Δ` |
|---|---|---|---|---|---|
| great blob | 0.013–0.071 | 0.02–0.24 | **−73 … −59** | 4.0–9.5 | 1.9–6.7 |
| middle blob | 0.044–0.144 | 0.02–0.22 | **−72 … −58** | **0.7–3.8** | 5.7–11.9 |
| end blob | 0.043–0.142 | −0.02–0.19 | **−52 … −43** | **0.5–2.7** | 3.0–10.9 |
| dendrite T | 0.046–0.137 | 0.02–0.23 | −56 … −50 | 2.3–7.1 | 3.9–10.7 |
| axon | 0.029–0.097 | 0.03–0.25 | −52 … −46 | 4.7–9.6 | 2.2–8.2 |
| axon terminals | **0.008–0.019** | 0.00–0.23 | −48 … −43 | 6.7–10.5 | 1.7–5.6 |
| dendrite C | 0.012–0.069 | 0.02–0.24 | −48 … −42 | 4.3–8.1 | 2.0–7.6 |
| dendrite B (onion) | 0.009–0.039 | 0.02–0.24 | −47 … −42 | **1.8–4.6** | 1.9–7.3 |
| cell body | 0.051–0.148 | 0.05–0.24 | −54 … −48 | 2.3–6.6 | 3.6–10.0 |

**`b` is identical in every region** (IQR ≈ 0.02–0.24 everywhere). Of the five
design axes, `b` carries no information about which region a sample lands in.

---

## 3. What separates the "neuron" from the "blob"

### 3.1 The observable: one ISI versus many

This is the cleanest result in the dataset, and it is categorical rather than
statistical.

| | blob | neuron |
|---|---|---|
| samples with `p = 1` (a single repeated ISI) | **99.47%** | **0.00%** (0 of 15 895) |
| median `CV(ISI)` | 2 × 10⁻⁵ | 0.38 – 1.77 |
| median `I_max / I_min` | **1.000** | 2.4 – 64 |
| median log-ISI entropy `H` | **0.00** | 0.69 – 1.63 |
| return map (fig. `regions_return_maps.png`) | **one point on the diagonal** | 2, 3, 4… points, or a filled curve |

Every blob neuron settles onto a limit cycle with exactly one interval. Every
neuron-side neuron settles onto an orbit with two or more distinct intervals, or
onto a chaotic attractor. The separation is all but categorical: **no** neuron-side
sample has `R < 1.2` (minimum 1.31, p1 = 1.69), and only **85 of 48 095** blob
samples (0.18%) have `R > 1.2`. The blob's 99th percentile of `R` is 1.011.

This is also why the two super-structures are *topologically* disconnected in the
embedding rather than merely far apart. Feature block B (the 12×12 return-map
KDE) collapses to a single delta at the origin for every period-1 train, so the
entire blob differs only in block D (`log₁₀ Ī`) and in the onset block — it is a
low-dimensional sheet. The neuron side spreads block B over the plane.

### 3.2 The cause: the reset voltage `c`, gated by `d`

| rule | precision | recall | accuracy |
|---|---|---|---|
| `c > −52 mV` and `d > 1.5` → neuron | 0.76 | 0.82 | **0.89** |
| `c + 2.84·log₁₀ d − 3.95·log₁₀ a > −41.9` | — | — | 0.88 (AUC **0.943**) |
| depth-8 decision tree on `(log a, b, c, d, log Δ)` | — | — | **0.947** (4-fold CV) |
| `c` alone (any threshold) | — | — | 0.87 (AUC 0.936) |

`c` alone carries almost all of it. The mechanism is direct: after a spike the
cell is reset to `v = c` and the recovery variable to `u + d`. If `c` is deeply
hyperpolarised (`c ≲ −60 mV`, the blob) the cell takes a long time to climb back
to threshold, `u` relaxes essentially to completion within one interval, each
spike therefore starts from the same state, and the map has a stable fixed point:
period 1. If `c` is shallow (`c ≳ −52 mV`, the neuron) the next spike arrives
before `u` has relaxed, the `+d` kicks accumulate, and the fixed point loses
stability through a period-doubling.

The exception that proves it is the **end blob**: shallow `c` (−52 … −43) *but*
almost no kick (`d` 0.5–2.7). With nothing accumulating there is nothing to
destabilise, so it stays period 1 — at 438 Hz. That is why the simple `c`
threshold needs the `d > 1.5` conjunct.

### 3.3 The neuron is a period-doubling cascade in one parameter

Define

```
β = d / (c + 62)        (requires c > −62 mV; −62 is the neuron side's floor)
```

β is the recovery kick per millivolt of reset depolarisation above −62 mV, and it
orders the entire left-hand structure (Spearman ρ(β, orbit period) = **−0.80**).
Measured over the 15 895 neuron-side samples:

| β | period 2 | period 3 | period ≥ 4 | aperiodic | dominant region |
|---|---|---|---|---|---|
| 0.9 – 1.5 | **100%** | 0% | 0% | 0% | dendrite T (82%) |
| 0.6 – 0.9 | **97%** | 1% | 1% | 1% | axon (59%), terminals (23%) |
| 0.4 – 0.6 | **75%** | 21% | 2% | 2% | terminals (37%), axon (34%), C (21%) |
| 0.3 – 0.4 | 32% | **52%** | 13% | 3% | dendrite C (52%) |
| 0.2 – 0.3 | 12% | 30% | **56%** | 3% | dendrite B (56%), C (29%) |
| 0.1 – 0.2 | 3% | 7% | **85%** | 4% | dendrite B (87%) |
| < 0.1 | 2% | 1% | **72%** | **26%** | dendrite B (94%) |

Read bottom-up from the blob: as β falls below ≈ 1.5 the fixed point doubles
(dendrite T), the doublet's contrast grows (axon → terminals), a period-3 window
opens (dendrite C), then period 4, 5, 6, … (the onion), and finally the orbit
fails to close at all (chaos). The same `(c, d)` dependence read as a table:

**median orbit period, neuron side, by `c` and `d` quartile**

| `c` \ `d` | 0.0–3.3 | 3.3–5.7 | 5.7–8.5 | 8.5–12 |
|---|---|---|---|---|
| −61 … −50 | 3 | 2 | 2 | 2 |
| −50 … −46 | 5 | 3 | 2 | 2 |
| −46 … −43 | 7 | 4 | 3 | 2 |
| −43 … −40 | **9** | 5 | 3 | 2 |

and the aperiodic fraction over the same cells rises from 0.6% (deep `c`, large
`d`) to **18%** (shallow `c`, small `d`).

`a` plays no part in choosing the branch (ρ(log a, period) = −0.14). What `a`
does is set **where along a branch** a sample sits: within each region,
ρ(log a, log rate) = **+0.75 … +0.89**. The two roles are cleanly separated —
**`(c, d)` choose the orbit, `a` chooses the tempo.**

---

## 4. The nine regions, one at a time

### 4.1 Great blob — *the tonic sheet* (39 828; 62% of the embedded set)

Perfectly periodic firing at 5.5–186 Hz (IQR 14–53). Every one of the 47
HDBSCAN letters inside it is "slow / medium / fast tonic"; η²(log rate | letter)
= **0.83**, i.e. the letters are rate bins. Six of the seven canonical types
(RS, IB, FS, LTS and their neighbours) land here.

It is a two-dimensional sheet, and both axes are measurable:

- **UMAP y ↔ steady-state rate**: ρ(log f, y) = **−0.94**.
- **UMAP x ↔ onset adaptation**: within fixed rate bins,
  ρ(`log₁₀(first ISI / steady ISI)`, x) = **+0.95** (five of eight bins ≥ 0.84).
  The quantity ranges from −1.36 to −0.01, i.e. from "the first interval is 4%
  of the steady one" (violent onset burst, strong adaptation) to "no transient
  at all".

So the great blob's width is **entirely the onset transient**, which only enters
the clustering through the `w_onset` block (`IMPROVEMENTS.md` §6). Its steady-state
feature vector is one scalar.

### 4.2 Middle blob — *fast tonic* (3 756)

Period 1 at **158–226 Hz**. Deep reset (`c` −72 … −58) combined with a *small*
kick (`d` 0.7–3.8) and high overdrive (`Δ` 5.7–11.9) and fast recovery (`a`
0.044–0.144). The canonical **RZ** seed sits here at 195 Hz.

It is the same dynamics as the great blob, one rate band up; UMAP detaches it as
a lobe because at these rates the onset block saturates (96 spikes in the first
500 ms, against 15 for the great blob) and the onset KDE separates sharply.

### 4.3 End blob — *very fast tonic* (4 511)

Period 1 at **346–602 Hz** (full range 191–1389 Hz). The distinguishing
parameters are a *shallow* reset (`c` −52 … −43, unlike the other two lobes) with
a near-zero kick (`d` 0.5–2.7) — the cell is reset close to threshold and nothing
accumulates to slow it down, so it free-runs. Mean membrane potential is
**−34 mV** (against −65 mV in the great blob): these cells barely repolarise
between spikes. 54% are Class 1 excitable.

Spike-train spectral flatness is **2 × 10⁻⁵**, the purest line spectrum in the
dataset. The canonical **TC** seed (`d = 0.05`, 272 Hz) sits exactly at its
boundary with the middle blob.

### 4.4 Dendrite T (+ BG) — *the first period-doubling* (881)

Period 2, but *weak*: ISI contrast `R` = 1.9–3.1 (p99 = 3.5), the lowest of any
`p ≥ 2` region, and the burst index is **0.00** — the short interval is not short
enough to register as a burst. Highest rate on the neuron side (86–204 Hz, up to
316). Deepest `c` of the neuron side (−56 … −50), highest β (0.9–1.5).

This is the strip immediately past the period-doubling bifurcation, where the
two-cycle has just been born and its amplitude is still small. 22% are already
aperiodic — the sliver is thin and sits against the chaotic core. Cluster BG is
its fast tip (260 Hz, `R` ≈ 1.7).

### 4.5 Axon — *the doublet ladder* (3 742)

Period 2 at full amplitude: `R` = 7.6–20, burst index 0.50, 2.01 spikes/burst,
rate 41–81 Hz. Every cluster on the trunk is the same dynamics at a different
tempo, and the ladder is strictly ordered:

| cluster (bottom → top) | rate (Hz) | `R` | `a` | `c` |
|---|---|---|---|---|
| AR | 97 | 4.2 | 0.097 | −52.9 |
| BB | 86 | 5.4 | 0.083 | −51.9 |
| AB | 78 | 7.3 | 0.078 | −50.6 |
| AA | 69 | 10.4 | 0.069 | −48.9 |
| Y | 59 | 14.1 | 0.057 | −47.8 |
| BO | 49 | 17.1 | 0.046 | −47.7 |
| AD | 42 | 20.7 | 0.038 | −47.3 |
| BI | 41 | 25.7 | 0.038 | −46.0 |
| BM | 27 | 28.7 | 0.021 | −48.1 |
| BV | 32 | 32.1 | 0.026 | −46.0 |

`a` falls 4.6×, rate falls 3.6×, contrast rises 7.6× — monotonically, with no
discontinuity anywhere. **The ten clusters are ten bins of one coordinate.**
The invariant `a·Ī` stays within 0.76–1.00 along the whole trunk.

### 4.6 Axon terminals — *the extreme-doublet fan* (2 865)

The same period-2 orbit pushed to its limit: `R` = 44–97 (up to 231) at 11–21 Hz.
Slowest recovery in the dataset (`a` 0.008–0.019) with the largest kick
(`d` 6.7–10.5). Physically these are slow doublet bursters: two spikes ~5 ms
apart, then a ~70 ms silence.

**The fan is an artefact, not phenomenology.** The branch point at `y ≈ 14.8`
is where the 2000 ms window stops containing enough bursts to estimate the ISI
distribution smoothly, and HDBSCAN starts resolving **integer spike counts**:

| cluster | n | distinct `n_ISI` values | share in top 3 |
|---|---|---|---|
| BJ | 214 | **1** (all exactly 23) | 1.00 |
| BY | 156 | **1** (all exactly 15) | 1.00 |
| CC | 135 | 3 | 1.00 |
| CA | 145 | 7 | 0.97 |
| CB | 138 | 8 | 0.96 |
| AQ | 300 | 9 | 0.84 |
| AP | 305 | 22 | 0.56 |
| BR | 169 | 25 | 0.37 |
| BA | 278 | 37 | 0.37 |

Compare the trunk, where the same statistic is 96–168 distinct values per
cluster. The detached filaments BJ and BY are *single spike counts*. The fix, if
you want it, is a longer `t_total` or a rate-normalised ISI count, not a
clustering change.

### 4.7 Dendrite C — *the period-3 window* (3 484)

**98.8% of dendrite C is a period-3 orbit** — one long interval and two short
ones, repeating. Rate 25–90 Hz, `R` = 19–81, burst index 0.67, 3.05
spikes/burst. β = 0.30–0.42, i.e. a clean band between the period-2 and
period-≥4 regions. The long filament reaching out to `x = −11.6` is the slow end
of it (down to 13 Hz).

It is a *window*, in the dynamical-systems sense: a band of parameter space where
a period-3 orbit is stable, flanked on the low-β side by chaos. That is why it
reads as a separate branch rather than as part of a smooth progression.

### 4.8 Dendrite B — *the onion*, multi-spike bursting (4 386)

Period ≥ 4, with a long tail: **36% period 4, 19% period 5, 12% period 6, 7%
period 7**, down to period 16, plus 6% aperiodic. Rate 40–111 Hz, `R` = 32–120,
burst index 0.76, 4.13 spikes/burst, `H` = 1.56 — the broadest ISI distribution
of any *periodic* region. The defining parameter is a **small kick on a shallow
reset**: `d` 1.8–4.6 with `c` −47 … −42, giving β = 0.12–0.25. A small `d` means
many spikes are needed before `u` accumulates enough to terminate the burst,
hence long orbits.

The canonical **CH** seed (`a=0.02, b=0.2, c=−50, d=2`) lands here, on a period-5
orbit at 82.5 Hz with `R` = 26.4 — correctly placed, which is the single clearest
piece of evidence that the C1 encoding change worked.

The "onion" shape — a closed loop with nested rings — is period stratification:
each ring is a different orbit length, and the loop closes because period `n`
and period `n+1` orbits at matched rate are nearest neighbours on both sides.

### 4.9 Cell body — *the chaotic core* (135)

The only region HDBSCAN declined to label, and it is right to: this is not a
mode, it is the **junction where the period-2, period-3 and period-≥4 branches
meet**, and the dynamics there are chaotic.

- **53% aperiodic** — no orbit of period ≤ 16 closes to 2%. The remainder are
  27% period-3 and 10% period-4, i.e. transient windows inside the chaotic band.
- **Highest log-ISI entropy in the dataset**: `H` = 1.63 (p90 = 2.24). All 57
  samples in the whole run with `H > 2.2` lie inside `x ∈ [−5.8, −4.3]`,
  `y ∈ [−4.5, 0.1]` — the soma, exactly.
- Its return map (`regions_return_maps.png`, bottom right) is a **filled unimodal
  curve**, not a set of points: the signature of a 1-D chaotic map.
- Rate 110–182 Hz, CV 0.76, `R` = 6.4–9.2, burst index 0.00.

**Your spectral observation, quantified.** Spike-train spectral flatness over
2–500 Hz (1 = white):

| region | median `F` | p75 `F` | fraction with `F > 0.10` |
|---|---|---|---|
| **cell body** | 0.026 | **0.330** | **40.7%** |
| axon terminals | 0.034 | 0.059 | 7.6% |
| dendrite T | 0.010 | 0.018 | 7.8% |
| dendrite B (onion) | 0.024 | 0.047 | 5.8% |
| dendrite C | 0.015 | 0.034 | 3.9% |
| great blob | 0.011 | 0.023 | 1.0% |
| axon | 0.008 | 0.013 | 0.3% |
| middle blob | 0.0002 | 0.003 | 0.0% |
| end blob | 0.00002 | 0.0003 | 0.3% |

The *median* understates it — a 15 Hz doublet train (axon terminals) has ~30
harmonics inside the 2–500 Hz band, so its comb reads as moderately "flat"
without being broadband. The discriminating statistic is the **upper tail**:
41% of cell-body samples exceed `F = 0.10`, against ≤ 8% everywhere else, and
the cell body's p75 is 0.330 against ≤ 0.06 for every other region. Restricting
to aperiodic samples only, the cell body's median flatness is **0.320** against
0.061 (dendrite C), 0.059 (dendrite B) and 0.047 (dendrite T): aperiodicity
elsewhere is long-orbit or marginal, aperiodicity in the soma is chaos.

Note that `Q1` in the figure you referenced (169 Hz, CV 0.76, BI 0.00, "fast
tonic, +irregular", broadband spectrum) matches the cell-body profile almost
exactly — 140 Hz, CV 0.76, BI 0.00 median.

---

## 5. Numerical constraints

### 5.1 Identification — from a measured spike train

Compute `p`, `f`, `R`, `H` as in §1.1, then:

```
tonic := (p == 1) or (R < 1.20)

if tonic:
    f <= 137 Hz                    ->  great blob
    137 < f <= 286 Hz              ->  middle blob
    f > 286 Hz                     ->  end blob
elif p == 2:
    R < 3.5                        ->  dendrite T
    3.5 <= R < 32                  ->  axon
    R >= 32                        ->  axon terminals
elif p == 3:                       ->  dendrite C
elif p >= 4:                       ->  dendrite B (onion)
elif p == 0 (aperiodic):
    R >= 32                        ->  dendrite B (onion)   # chaotic bursting
    else                           ->  cell body
```

Measured against the geometric region labels on all 63 990 embedded samples:

| region | precision | recall | F1 |
|---|---|---|---|
| great blob | 0.996 | 0.996 | 0.996 |
| middle blob | 0.913 | 0.939 | 0.926 |
| end blob | 0.992 | 0.945 | 0.968 |
| dendrite T | 0.866 | 0.690 | 0.768 |
| axon | 0.979 | 0.966 | 0.973 |
| axon terminals | 0.971 | 0.971 | 0.971 |
| dendrite C | 0.983 | 0.988 | 0.986 |
| dendrite B (onion) | 0.969 | 0.979 | 0.974 |
| cell body | **0.18** | 0.53 | 0.27 |
| **overall accuracy** | | | **0.9793** |

Seven of the nine regions are recovered at F1 ≥ 0.93 from four scalars. The two
that are not are the two transition zones, and their failure is informative:

- **dendrite T** loses recall to its own 22% aperiodic fraction and leaks to the
  blob at the low-contrast end — it is the bifurcation strip, so it has no sharp
  outer boundary by construction.
- **cell body** has 0.18 precision because "aperiodic" also catches the chaotic
  fringes of dendrite T and dendrite B. Tightening to
  `p == 0 and H > 1.7 and 0.8 < log₁₀R < 1.2 and f > 100 Hz` raises precision to
  **0.53** at 0.35 recall — the best available. The cell body is genuinely not a
  separable mode; it is the connecting manifold, which is why HDBSCAN called it
  noise.

### 5.2 Generation — from `(a, b, c, d, Δ)`

Parameter space does **not** partition into these regions cleanly, because the
firing pattern is a strongly nonlinear function of the parameters. The ceiling
for any classifier on `(log a, b, c, d, log Δ)` is 0.947 for the two-way
neuron/blob split and ~0.87 for the nine-way split (depth-8 tree, 4-fold CV).
The rules below are therefore **proposal distributions**: sample inside the box,
simulate, and accept using §5.1. `hit` is the measured fraction of the run's
Sobol samples inside the box that actually land in the region; `recall` is the
fraction of the region the box covers.

Neuron-side regions — with `β = d/(c + 62)`:

| region | `c` (mV) | `β` | `a` | hit | recall |
|---|---|---|---|---|---|
| axon terminals | −50 … −40 | 0.40–0.70 | 0.005–0.022 | **0.79** | 0.63 |
| dendrite B (onion) | −50 … −40 | 0.08–0.26 | 0.005–0.10 | **0.75** | 0.58 |
| axon | −54 … −45 | 0.45–0.80 | 0.02–0.15 | 0.56 | 0.42 |
| dendrite C | −50 … −40 | 0.27–0.45 | 0.005–0.15 | 0.55 | 0.61 |
| dendrite T | −58 … −48 | 0.60–1.50 | ≥ 0.04 | 0.10 | 0.37 |
| cell body | −56 … −46 | 0.30–0.70 | ≥ 0.05 | 0.03 | 0.47 |

Blob regions:

| region | `c` (mV) | `d` | `a` | `Δ` | hit | recall |
|---|---|---|---|---|---|---|
| great blob | −80 … −56 | 2.0–12.0 | any | any | **0.95** | 0.75 |
| end blob | −55 … −40 | 0.0–2.5 | ≥ 0.03 | any | **0.76** | 0.44 |
| middle blob | −80 … −55 | 0.0–3.5 | ≥ 0.03 | ≥ 5 | 0.56 | 0.37 |

Tighter 4-D interquartile boxes reach hit rates of 0.84–1.00 at the cost of
recall below 0.05 — useful if you want a guaranteed exemplar rather than a
sample of the region:

| region | box (`a`, `c`, `d`, `Δ`) | hit | recall |
|---|---|---|---|
| great blob | `a∈[0.011,0.086]`, `c∈[−74.6,−57.8]`, `d∈[3.4,10.1]`, `Δ∈[1.7,7.7]` | 1.00 | 0.12 |
| axon terminals | `a∈[0.0081,0.0171]`, `c∈[−47.1,−43.8]`, `d∈[7.2,10.2]`, `Δ∈[1.9,4.9]` | 1.00 | 0.03 |
| dendrite B | `a∈[0.0119,0.0280]`, `c∈[−45.9,−42.6]`, `d∈[2.3,3.9]`, `Δ∈[2.5,5.6]` | 1.00 | 0.01 |
| end blob | `a∈[0.071,0.110]`, `c∈[−48.2,−45.0]`, `d∈[0.88,1.70]`, `Δ∈[5.0,8.3]` | 1.00 | 0.002 |
| middle blob | `a∈[0.062,0.123]`, `c∈[−69.0,−60.2]`, `d∈[1.09,2.81]`, `Δ∈[7.0,10.7]` | 0.93 | 0.015 |
| axon | `a∈[0.043,0.068]`, `c∈[−50.1,−48.0]`, `d∈[6.3,8.2]`, `Δ∈[3.4,5.7]` | 0.85 | 0.003 |
| dendrite C | `a∈[0.021,0.041]`, `c∈[−45.7,−43.5]`, `d∈[5.6,7.1]`, `Δ∈[3.0,5.1]` | 0.84 | 0.003 |
| dendrite T | `a∈[0.064,0.116]`, `c∈[−55.3,−51.9]`, `d∈[3.2,6.0]`, `Δ∈[5.1,9.2]` | 0.35 | 0.02 |
| cell body | `a∈[0.063,0.125]`, `c∈[−52.8,−48.7]`, `d∈[3.4,5.7]`, `Δ∈[4.7,9.0]` | 0.12 | 0.05 |

In every box, **leave `b` free** — it is uninformative (§2).

### 5.3 The two-sentence generator

> Choose `c` and `d`. If `c < −55` or `d < 1.5`, you get a period-1 tonic cell in
> the blob, and `a` sets its rate (0.005 → slow, 0.19 → fast). Otherwise compute
> `β = d/(c + 62)`: β ≈ 1 gives a weak doublet, β ≈ 0.6 a strong doublet,
> β ≈ 0.35 a triplet, β ≈ 0.15 a multi-spike burst, β < 0.1 chaos — and again `a`
> sets the tempo.

---

## 6. Where the canonical types land

| type | `(a, b, c, d)` | region | rate | period |
|---|---|---|---|---|
| RS | 0.02, 0.20, −65, 8 | great blob (AY) | 22.5 | 1 |
| IB | 0.02, 0.20, −55, 4 | great blob (K) | 32 | 1 |
| **CH** | 0.02, 0.20, −50, 2 | **dendrite B (onion)** | 82.5 | **5** |
| FS | 0.10, 0.20, −65, 2 | great blob (AM), at the middle-blob boundary | 136 | 1 |
| LTS | 0.02, 0.25, −65, 2 | great blob (Q) | 75 | 1 |
| TC | 0.02, 0.25, −65, 0.05 | **end blob**, on the middle/end pinch | 272 | 1 |
| RZ | 0.10, 0.26, −65, 2 | **middle blob** | 195 | 1 |

Six of the seven published types are period-1 at constant drive and therefore all
live in the blob. Only **CH** is on the neuron. That is the honest statement of
what `IMPROVEMENTS.md` §11 item 1 anticipated: at a single constant drive the
paper's taxonomy is mostly *not* distinguishable by steady-state ISI structure,
and the structure the ISI encoding does resolve — the reset map's orbit ladder —
is a different and larger taxonomy than the paper's.

---

## 7. Three caveats

1. **The axon-terminal fan is a finite-window artefact** (§4.6). Clusters BJ and
   BY each consist of exactly one spike count. Nine of the 71 reported clusters,
   holding 2 865 samples, are quantisation bins.
2. **The great blob's 47 clusters are rate bins** (η² = 0.83) differing only in a
   single scalar at steady state; their second dimension is the onset transient,
   which enters only through the `w_onset` block. If `w_onset` were 0 the entire
   blob would collapse to a curve.
3. **The cell body is a boundary, not a class.** Its 135 points are the chaotic
   junction of three branches. Any clustering will either call it noise (as this
   one does) or split it arbitrarily. Reporting it as "the chaotic core" with a
   flatness threshold is more honest than giving it a letter.
