# Izhikevich spiking-neuron cluster discovery — v2 run report

Implements `IMPROVEMENTS.md` (C1–C9) on top of `SPEC.md`.

- Date: 2026-11-21 23:43:35
- Samples `N` = 65536 (single Sobol design over `(a, b, c, d, Δ)`), drive `I = I*(θ) + Δ` (§4)
- Encoding: ISI sequence, 372 dims (185 steady + 187 onset), `h=0.25 wA=1 wB=1 wC=0.5 wD=0.3 wOn=0.3`
- Selected clustering config: `n_neighbors=50`, `min_cluster_size=120`, `min_samples=5`, `cluster_selection_method='eom'`
- Clusters: **71** after the §8.4 merge (114 before), 194 at `leaf`; noise 15.0%; `relative_validity_` 0.0866
- Held out (§3.4): `Z` = 0, `S` = 0, `W` = 1546, `X` = 0; clustered 63990 (97.6% of N — v1 clustered 54.9%)
- PCA: 16 components, 0.9914 of the variance; 99% reached at **16** components (v1: 92 of 128)

## 1. Convergence gate (§5 — C4)

Judged on spike times, not on the feature metric.  Probe set: 24 (24 spiking, 1 sensitivity-limited and excluded as chaotic, 23 used).  Sensitivity baseline (`dt=0.0005` vs `0.001` ms) median interval error 4.88e-04 ms.

| dt (ms) | counts identical | worst \|Δn\| | median per-ISI err (ms) | probes within 0.05 ms | worst per-ISI err | probes within 1% rel L2 | median \|Δt_spike\| (accumulated, ungated) | accepted |
|---|---|---|---|---|---|---|---|---|
| 0.04 | 16/23 (70%) | 3 | 2.40e-02 | 83% | 7.50e-02 | 91% | 1.10e+00 | no |
| 0.02 | 17/23 (74%) | 2 | 1.40e-02 | 100% | 3.86e-02 | 96% | 5.93e-01 | no |
| 0.01 | 21/23 (91%) | 1 | 7.02e-03 | 100% | 1.81e-02 | 96% | 5.40e-01 | **yes** (selected) |
| 0.005 | 22/23 (96%) | 1 | 3.05e-03 | 100% | 8.91e-03 | 96% | 2.49e-01 | **yes** |
| 0.0025 | 23/23 (100%) | 0 | 1.10e-03 | 100% | 3.48e-03 | 100% | 6.65e-02 | **yes** |
| 0.001 | 23/23 (100%) | 0 | 0.00e+00 | 100% | 0.00e+00 | 100% | 0.00e+00 | **yes** |

- selected `dt_int` = **0.01 ms**, the coarsest ladder step meeting all three §5 checks in their §5.1 fractional form: 91% of used probes have an identical spike count and none differs by more than 1 (cap 1); 100% have a median per-ISI error under 0.05 ms and the worst is 1.81e-02 ms (cap 0.25); 96% are inside 1% relative L2 — each against the 90% threshold
- the strict reading of the draft (every probe, every check) would have selected 0.0025 ms instead; §5.1 amends it because an all-probes rule gets monotonically stricter as the probe set grows, which would make the gate's answer depend on how many probes were run (V2-A15)
- `V_floor` = **1.0000e-01 mV** = max(numerical 2.056e-03 from 18 silent references at `I = 0`, physical 1.000e-01 = 0.1 mV RMS) — `sd(v)` replaces v1's `ac_power`/`P_floor` (§3.4)

## 2. Drive relative to rheobase (§4 — C3)

- `I*` bracketed for **65517** of 65536 samples (19 have no rheobase at `I <= 30`, 0 fire already at the analytic bracket — bistable corners, §4.1 item 3)
- `I*` quartiles: 1.61 / 8.65 / 15.72; production `I` median 13.81 (v1 held `I = 10` for every sample)
- excitability class (§4.1): Class 1 (`b <= a`) 22260, Class 2 (`b > a`) 43276
- f–I onset shape (§4.3): firing rate measured at `I*` itself, over a 200 ms window (so the measurement floor is 5 Hz).  Median 10.0 Hz overall; **16.4%** of samples start above 20 Hz, which a continuous (Class 1) onset cannot do at a bisection tolerance of 0.1 in `I`.  Split by the analytic class: median 10.0 Hz for Class 1 (`b <= a`, no Hopf) against 10.0 Hz for Class 2 — the discrete excitability split v1 could not observe
- zero-spike samples: **0** (0.0%), against **45.1%** in v1 — the §4.3 prediction

## 3. Representation health (§8.1 — the gates that would have caught v1)

| gate | threshold | v1 | v2 | result |
|---|---|---|---|---|
| PCA components for 99% variance | <= 20 | 92 | 16 | PASS |
| mean cosine distance at \|Δ log f₀\| ∈ [0.05, 0.10] relative to \|Δ log f₀\| > 1 | <= 0.5 | 0.818 | 0.047 | PASS |
| octave-pair distance >= 1.5x-pair distance | required | violated (0.379 vs 0.501) | 0.016 vs 0.011 | PASS |

Measured on 4000 pure-tonic samples (`CV < 0.02`, `BI < 0.02`), the same subset definition §2.3 used.  The v2 version of the §2.3 saturation table:

| \|Δ log f₀\| | pairs | mean cosine distance (v2) |
|---|---|---|
| 0 – 0.002 | 44947 | 0.0094 |
| 0.002 – 0.01 | 44239 | 0.0056 |
| 0.01 – 0.02 | 79600 | 0.0075 |
| 0.02 – 0.05 | 261361 | 0.0079 |
| 0.05 – 0.1 | 423284 | 0.0083 |
| 0.1 – 0.2 | 830748 | 0.0099 |
| 0.2 – 0.5 | 2209851 | 0.0199 |
| 0.5 – 1 | 2558878 | 0.0593 |
| 1 – 2 | 1511960 | 0.1726 |
| 2 – 5 | 33132 | 0.3646 |

## 4. Clusters

Names follow v1 §14.2 as amended by §8.4; flags follow §8.2 (UNSTABLE) and v1 §13.3 (MULTIMODAL).  Every clustered sample fires and has at least 10 ISIs, so the v1 silent/subthreshold names are gone — those populations are the `Z`/`S`/`W` classes above.

### A — fast tonic

- size 8267, mean membership prob 0.994, noise adjacency 0.000
- medians: rate 296.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 592 ISIs
- drive: median `I` 18.38 = `I*` 10.92 + `Δ` 7.83; 49% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.09443, b=0.09812, c=-55.78, d=2.199, Δ=7.783
- persistence 0.00, DBCV contribution +0.0046  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b        c       d   delta  sd_log_a
a      0.0572                                     0.3789
b      0.0070  0.1278                                   
c     -0.0320 -0.0358  11.3292                          
d      0.5319 -0.0788  -0.0560  2.1858                  
delta  0.0234  0.1192  -0.4085  0.3272  4.1815          
```

### B — medium multi-spike bursting (4.1 spikes/burst, 17 Hz interburst), +adapting

- size 4386, mean membership prob 0.992, noise adjacency 0.000
- medians: rate 70.0 Hz, CV(ISI) 1.772, burst index 0.758, spikes/burst 4.13, adaptation +0.335, 139 ISIs
- drive: median `I` 13.49 = `I*` 8.32 + `Δ` 3.73; 73% Class 2; bursting members 90%, irregular 100%
- mean coefficient vector: a=0.03013, b=0.1309, c=-44.95, d=3.321, Δ=5.029
- persistence 0.00, DBCV contribution +0.0666  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0311                                    0.3907
b      0.0566  0.1298                                  
c      0.1687 -0.0238  3.8610                          
d      0.5241 -0.1418  0.5967  1.8782                  
delta -0.0466  0.0035  0.0383  0.1751  3.7533          
```

### C — medium triplet bursting (3.1 spikes/burst, 16 Hz interburst), +adapting

- size 3484, mean membership prob 0.958, noise adjacency 0.000
- medians: rate 49.5 Hz, CV(ISI) 1.289, burst index 0.673, spikes/burst 3.05, adaptation +0.201, 98 ISIs
- drive: median `I` 13.82 = `I*` 8.65 + `Δ` 3.90; 68% Class 2; bursting members 96%, irregular 100%
- mean coefficient vector: a=0.04724, b=0.1274, c=-45.36, d=6.223, Δ=5.197
- persistence 0.00, DBCV contribution +0.0529  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0458                                    0.4413
b      0.0337  0.1296                                  
c      0.1240  0.0170  3.9810                          
d      0.3299 -0.1097  0.8758  2.5215                  
delta -0.0813  0.0035  0.0282  0.2038  3.8140          
```

### D — medium tonic

- size 3453, mean membership prob 0.928, noise adjacency 0.006
- medians: rate 25.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 50 ISIs
- drive: median `I` 12.58 = `I*` 9.24 + `Δ` 2.22; 64% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.04745, b=0.1173, c=-64.76, d=7.855, Δ=3.405
- persistence 0.00, DBCV contribution -0.0254  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0250                                    0.2643
b     -0.2990  0.1293                                  
c     -0.0363  0.0760  8.5791                          
d      0.3347  0.0469  0.1010  2.7605                  
delta -0.6375  0.0456  0.1589 -0.0616  3.0623          
```

### E — slow tonic

- size 3130, mean membership prob 0.952, noise adjacency 0.013
- medians: rate 11.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 22 ISIs
- drive: median `I` 13.60 = `I*` 8.08 + `Δ` 4.25; 76% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.01185, b=0.1319, c=-60.31, d=7.534, Δ=5.347
- persistence 0.00, DBCV contribution -0.0273  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0068                                    0.2238
b     -0.2163  0.1306                                  
c     -0.0591  0.0532  8.9705                          
d      0.2937 -0.0749  0.1691  2.8676                  
delta -0.6037 -0.0311  0.0010  0.2945  3.8125          
```

### F — slow tonic

- size 2589, mean membership prob 0.971, noise adjacency 0.016
- medians: rate 17.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 33 ISIs
- drive: median `I` 13.68 = `I*` 10.53 + `Δ` 2.09; 65% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.02893, b=0.1046, c=-64.76, d=7.855, Δ=3.367
- persistence 0.00, DBCV contribution -0.0207  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0148                                    0.2616
b     -0.3267  0.1288                                  
c      0.0360  0.0224  8.6381                          
d      0.4032  0.0240  0.1144  2.8019                  
delta -0.6580  0.0470 -0.0218  0.0070  3.0902          
```

### G — slow tonic

- size 1592, mean membership prob 0.820, noise adjacency 0.010
- medians: rate 8.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 15 ISIs
- drive: median `I` 13.21 = `I*` 9.62 + `Δ` 2.83; 74% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.009875, b=0.116, c=-67.77, d=7.369, Δ=3.572
- persistence 0.00, DBCV contribution -0.0051  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0047                                    0.1825
b     -0.2515  0.1277                                  
c      0.2269 -0.0783  7.5307                          
d      0.3475  0.0082  0.1755  2.9523                  
delta -0.5288 -0.0286 -0.1016  0.4323  2.3322          
```

### H — medium tonic

- size 1512, mean membership prob 0.968, noise adjacency 0.000
- medians: rate 56.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 112 ISIs
- drive: median `I` 10.21 = `I*` 6.99 + `Δ` 2.65; 58% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1207, b=0.1432, c=-67.95, d=6.828, Δ=3.067
- persistence 0.20, DBCV contribution +0.0231  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0420                                    0.1772
b      0.1511  0.1292                                  
c      0.2405  0.1117  7.6961                          
d      0.1515  0.0624  0.2416  3.2542                  
delta -0.5655 -0.2061 -0.0112  0.5793  1.6195          
```

### I — slow tonic

- size 1241, mean membership prob 0.962, noise adjacency 0.014
- medians: rate 7.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 13 ISIs
- drive: median `I` 13.42 = `I*` 11.36 + `Δ` 1.71; 68% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.01143, b=0.09506, c=-65.4, d=8.715, Δ=2.011
- persistence 0.40, DBCV contribution +0.0000  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0032                                    0.1139
b     -0.3434  0.1265                                  
c      0.0885 -0.0011  8.4694                          
d      0.2511  0.1657  0.1066  2.3046                  
delta -0.4992 -0.0922  0.0024  0.3269  0.9617          
```

### J — fast tonic

- size 1197, mean membership prob 0.903, noise adjacency 0.012
- medians: rate 104.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 208 ISIs
- drive: median `I` 15.62 = `I*` 7.22 + `Δ` 8.27; 65% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.08054, b=0.14, c=-67.1, d=4.464, Δ=8.38
- persistence 0.00, DBCV contribution -0.0025  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0515                                    0.3622
b      0.0572  0.1284                                  
c      0.0819  0.0230  6.9322                          
d      0.5495 -0.0063  0.1368  3.3532                  
delta -0.2811  0.0363 -0.0385  0.4863  3.6743          
```

### K — medium tonic

- size 962, mean membership prob 0.948, noise adjacency 0.024
- medians: rate 35.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 69 ISIs
- drive: median `I` 13.71 = `I*` 6.44 + `Δ` 6.65; 73% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.04304, b=0.1451, c=-62.21, d=6.511, Δ=6.904
- persistence 0.00, DBCV contribution +0.0150  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0375                                    0.3834
b     -0.2177  0.1276                                  
c      0.0284  0.1052  8.7319                          
d      0.5143 -0.1095  0.3296  3.2143                  
delta -0.7474  0.0012 -0.0016 -0.0898  4.4905          
```

### L — medium tonic

- size 915, mean membership prob 0.882, noise adjacency 0.026
- medians: rate 52.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 104 ISIs
- drive: median `I` 13.88 = `I*` 5.92 + `Δ` 8.24; 77% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.03993, b=0.1539, c=-65.06, d=5.249, Δ=8.054
- persistence 0.00, DBCV contribution -0.0088  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 210 parameter-space components, 2 of them holding >= 5% of the cluster, largest 48%
    - n=443: a=0.03482, b=0.1609, c=-61.65, d=6.319, Δ=10.16
    - n=109: a=0.05576, b=0.1855, c=-71.49, d=4.232, Δ=4.219

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0259                                    0.3064
b      0.1551  0.1268                                  
c      0.3562  0.0873  8.3783                          
d      0.3846  0.0174  0.4115  3.3408                  
delta -0.4535 -0.0949  0.1183  0.4327  3.9703          
```

### M — fast tonic

- size 846, mean membership prob 0.960, noise adjacency 0.015
- medians: rate 81.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 161 ISIs
- drive: median `I` 14.42 = `I*` 6.82 + `Δ` 6.18; 63% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.08449, b=0.1388, c=-67.36, d=5.038, Δ=6.93
- persistence 0.00, DBCV contribution -0.0051  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 294 parameter-space components, 2 of them holding >= 5% of the cluster, largest 11%
    - n=97: a=0.04514, b=0.2326, c=-68.5, d=4.901, Δ=9.98
    - n=44: a=0.1042, b=0.2935, c=-61.67, d=6.946, Δ=7.366

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0533                                    0.3529
b      0.0699  0.1295                                  
c      0.1793  0.0869  7.2377                          
d      0.3890  0.0426  0.2600  3.4822                  
delta -0.5093  0.0167  0.0103  0.4155  3.7421          
```

### N — medium tonic

- size 830, mean membership prob 0.869, noise adjacency 0.022
- medians: rate 29.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 57 ISIs
- drive: median `I` 12.68 = `I*` 7.29 + `Δ` 5.47; 77% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.02419, b=0.1403, c=-67.74, d=5.344, Δ=5.792
- persistence 0.00, DBCV contribution -0.0061  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0117                                    0.2032
b      0.1574  0.1283                                  
c      0.5461 -0.0173  8.0811                          
d      0.2619 -0.0198  0.2081  3.2054                  
delta -0.3552 -0.1771 -0.1682  0.6866  3.3774          
```

### O — medium tonic

- size 740, mean membership prob 0.940, noise adjacency 0.012
- medians: rate 42.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 83 ISIs
- drive: median `I` 12.81 = `I*` 7.71 + `Δ` 3.42; 64% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.06994, b=0.1328, c=-64.36, d=6.548, Δ=5.193
- persistence 0.00, DBCV contribution -0.0067  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 315 parameter-space components, 2 of them holding >= 5% of the cluster, largest 6%
    - n=43: a=0.01514, b=0.2328, c=-68.37, d=3.876, Δ=10.83
    - n=41: a=0.1535, b=0.03122, c=-57.59, d=7.851, Δ=1.459

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0502                                    0.3723
b     -0.2763  0.1309                                  
c      0.0831  0.1088  8.6835                          
d      0.4043 -0.0710  0.2611  3.1020                  
delta -0.7307  0.0962  0.0325 -0.0078  4.2021          
```

### P — slow tonic

- size 693, mean membership prob 0.659, noise adjacency 0.003
- medians: rate 5.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 10 ISIs
- drive: median `I` 13.81 = `I*` 11.92 + `Δ` 1.63; 69% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.008644, b=0.09599, c=-65.08, d=8.917, Δ=1.916
- persistence 0.60, DBCV contribution +0.0000

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0022                                    0.1066
b     -0.3205  0.1269                                  
c      0.1354  0.0026  8.7169                          
d      0.2218  0.1776  0.0997  2.2321                  
delta -0.4099 -0.1837 -0.0093  0.2919  0.9103          
```

### Q — medium tonic

- size 684, mean membership prob 0.881, noise adjacency 0.006
- medians: rate 67.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 134 ISIs
- drive: median `I` 13.78 = `I*` 5.56 + `Δ` 8.55; 74% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.06294, b=0.1592, c=-64.87, d=5.193, Δ=8.343
- persistence 0.40, DBCV contribution -0.0025  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0549                                    0.4270
b      0.0317  0.1291                                  
c      0.1394  0.0411  7.5630                          
d      0.5154  0.0029  0.4278  3.4180                  
delta -0.6788 -0.0745  0.0019  0.0516  4.0375          
```

### R — medium tonic

- size 657, mean membership prob 0.986, noise adjacency 0.027
- medians: rate 45.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 89 ISIs
- drive: median `I` 12.46 = `I*` 7.15 + `Δ` 4.74; 70% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.05983, b=0.1423, c=-66.16, d=5.923, Δ=5.432
- persistence 0.00, DBCV contribution -0.0056  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 320 parameter-space components, 2 of them holding >= 5% of the cluster, largest 8%
    - n=54: a=0.03632, b=0.1651, c=-64.46, d=7.439, Δ=8.893
    - n=34: a=0.02962, b=0.183, c=-71.55, d=4.403, Δ=6.475

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0441                                    0.3243
b      0.0203  0.1291                                  
c      0.3623  0.0868  8.3717                          
d      0.3878 -0.0907  0.2313  3.3381                  
delta -0.6084 -0.1573 -0.2245  0.3109  3.7478          
```

### T — fast tonic

- size 649, mean membership prob 0.959, noise adjacency 0.006
- medians: rate 104.5 Hz, CV(ISI) 0.443, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 208 ISIs
- drive: median `I` 15.86 = `I*` 8.34 + `Δ` 7.01; 60% Class 2; bursting members 0%, irregular 31%
- mean coefficient vector: a=0.09009, b=0.1282, c=-54.69, d=5.013, Δ=7.359
- persistence 0.40, DBCV contribution +0.0000  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0539                                    0.3683
b      0.0870  0.1283                                  
c      0.6989  0.0091  3.2794                          
d      0.5948 -0.1407  0.7167  3.1377                  
delta -0.1085  0.0163  0.2143  0.4775  3.9517          
```

### U — medium tonic

- size 645, mean membership prob 0.949, noise adjacency 0.020
- medians: rate 75.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 149 ISIs
- drive: median `I` 12.74 = `I*` 7.13 + `Δ` 4.95; 62% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.09984, b=0.1417, c=-68.85, d=5.966, Δ=5.63
- persistence 0.00, DBCV contribution -0.0047  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 228 parameter-space components, 2 of them holding >= 5% of the cluster, largest 14%
    - n=92: a=0.09899, b=0.1521, c=-65.36, d=5.474, Δ=5.33
    - n=35: a=0.1238, b=0.2775, c=-69.5, d=5.919, Δ=4.089

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0503                                    0.2626
b      0.1388  0.1284                                  
c      0.3128  0.0120  7.1332                          
d      0.3589 -0.0484  0.2040  3.5261                  
delta -0.4458 -0.1484 -0.1370  0.5408  3.0528          
```

### V — medium tonic

- size 629, mean membership prob 0.978, noise adjacency 0.010
- medians: rate 37.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 73 ISIs
- drive: median `I` 11.54 = `I*` 8.27 + `Δ` 2.36; 65% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.06892, b=0.1251, c=-67.67, d=7.1, Δ=3.191
- persistence 0.00, DBCV contribution -0.0007  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0348                                    0.2246
b     -0.1597  0.1274                                  
c      0.2531  0.0360  7.8674                          
d      0.3709  0.0094  0.1439  3.1693                  
delta -0.5921 -0.0812 -0.1814  0.3609  2.1915          
```

### Y — medium doublet bursting (2.0 spikes/burst, 29 Hz interburst)

- size 622, mean membership prob 0.982, noise adjacency 0.013
- medians: rate 59.0 Hz, CV(ISI) 0.874, burst index 0.504, spikes/burst 2.02, adaptation +0.000, 117 ISIs
- drive: median `I` 12.26 = `I*` 6.97 + `Δ` 4.10; 68% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.06745, b=0.1424, c=-48.27, d=7.385, Δ=5.236
- persistence 0.20, DBCV contribution +0.0000  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0464                                    0.3440
b     -0.0230  0.1272                                  
c      0.7420  0.1590  3.4379                          
d      0.5400 -0.0474  0.8851  2.8717                  
delta -0.5150 -0.0537 -0.0647  0.1126  3.8386          
```

### AA — medium doublet bursting (2.0 spikes/burst, 34 Hz interburst)

- size 620, mean membership prob 0.995, noise adjacency 0.018
- medians: rate 68.8 Hz, CV(ISI) 0.829, burst index 0.503, spikes/burst 2.01, adaptation +0.000, 136 ISIs
- drive: median `I` 12.24 = `I*` 6.71 + `Δ` 4.41; 67% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.07777, b=0.1446, c=-49.32, d=7.107, Δ=5.511
- persistence 0.00, DBCV contribution +0.0097  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 144 parameter-space components, 3 of them holding >= 5% of the cluster, largest 31%
    - n=191: a=0.09906, b=0.2292, c=-47.79, d=7.744, Δ=4.635
    - n=139: a=0.09453, b=0.03276, c=-48.09, d=8.647, Δ=6.345
    - n=48: a=0.04982, b=0.225, c=-50.6, d=5.386, Δ=6.181

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0487                                    0.3431
b      0.0453  0.1268                                  
c      0.7419  0.1735  3.2378                          
d      0.5253 -0.0760  0.8870  2.9353                  
delta -0.4583 -0.0588 -0.0108  0.1473  3.7950          
```

### AB — medium doublet bursting (2.0 spikes/burst, 39 Hz interburst) [mixed: 61% bursting]

- size 619, mean membership prob 0.981, noise adjacency 0.005
- medians: rate 78.0 Hz, CV(ISI) 0.763, burst index 0.500, spikes/burst 2.00, adaptation +0.000, 155 ISIs
- drive: median `I` 12.83 = `I*` 6.90 + `Δ` 5.11; 64% Class 2; bursting members 61%, irregular 100%
- mean coefficient vector: a=0.08463, b=0.1438, c=-51.01, d=6.442, Δ=5.98
- persistence 0.40, DBCV contribution +0.0000  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0509                                    0.3440
b      0.0293  0.1313                                  
c      0.7262  0.1779  3.1377                          
d      0.5111 -0.0684  0.8940  3.0551                  
delta -0.3818  0.0289  0.0827  0.2679  3.8346          
```

### AC — medium tonic

- size 591, mean membership prob 0.921, noise adjacency 0.012
- medians: rate 51.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 101 ISIs
- drive: median `I` 9.90 = `I*` 7.44 + `Δ` 2.00; 52% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1202, b=0.1325, c=-66.6, d=6.744, Δ=2.824
- persistence 0.20, DBCV contribution -0.0028  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 158 parameter-space components, 2 of them holding >= 5% of the cluster, largest 34%
    - n=202: a=0.1519, b=0.1705, c=-63.76, d=7.041, Δ=1.709
    - n=31: a=0.04292, b=0.215, c=-71.2, d=2.033, Δ=2.924

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0524                                    0.2589
b     -0.0080  0.1281                                  
c      0.2755  0.0627  8.0946                          
d      0.3524  0.1057  0.2079  3.2224                  
delta -0.6698 -0.1093 -0.1803  0.2217  2.2008          
```

### AD — medium doublet bursting (2.0 spikes/burst, 21 Hz interburst)

- size 556, mean membership prob 0.909, noise adjacency 0.018
- medians: rate 42.0 Hz, CV(ISI) 0.919, burst index 0.506, spikes/burst 2.02, adaptation +0.000, 83 ISIs
- drive: median `I` 12.22 = `I*` 7.42 + `Δ` 3.79; 69% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.04551, b=0.1378, c=-47.49, d=7.723, Δ=5.115
- persistence 0.60, DBCV contribution +0.0087  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 144 parameter-space components, 2 of them holding >= 5% of the cluster, largest 39%
    - n=216: a=0.04441, b=0.1513, c=-46.06, d=9.143, Δ=6.22
    - n=30: a=0.06568, b=0.2714, c=-46.08, d=6.93, Δ=1.729

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0316                                    0.3229
b     -0.1331  0.1308                                  
c      0.7257  0.0691  3.3658                          
d      0.5329 -0.0757  0.8853  2.8066                  
delta -0.4915 -0.0121 -0.0283  0.0927  3.8027          
```

### AE — medium tonic

- size 545, mean membership prob 0.840, noise adjacency 0.000
- medians: rate 32.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 64 ISIs
- drive: median `I` 9.96 = `I*` 7.64 + `Δ` 2.14; 65% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.068, b=0.1332, c=-65.51, d=8.223, Δ=2.416
- persistence 0.60, DBCV contribution +0.0003

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0201                                    0.1201
b     -0.1414  0.1273                                  
c      0.0947  0.0544  8.2703                          
d      0.1902  0.0775  0.1610  2.5272                  
delta -0.5938 -0.3094  0.0308  0.4353  1.1331          
```

### AF — medium tonic

- size 525, mean membership prob 0.949, noise adjacency 0.010
- medians: rate 39.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 78 ISIs
- drive: median `I` 13.13 = `I*` 9.81 + `Δ` 1.96; 53% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.08626, b=0.1107, c=-66.26, d=7.197, Δ=3.334
- persistence 0.00, DBCV contribution -0.0028  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 210 parameter-space components, 2 of them holding >= 5% of the cluster, largest 14%
    - n=76: a=0.1152, b=0.09512, c=-60.5, d=7.402, Δ=1.599
    - n=60: a=0.1431, b=-0.02991, c=-67.39, d=8.845, Δ=1.753

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0487                                    0.3268
b     -0.2265  0.1283                                  
c      0.2688  0.0450  8.3194                          
d      0.5041 -0.0383  0.2003  3.2172                  
delta -0.6726 -0.0579 -0.2668  0.0656  3.0846          
```

### AG — slow tonic

- size 524, mean membership prob 0.963, noise adjacency 0.013
- medians: rate 13.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 25 ISIs
- drive: median `I` 13.25 = `I*` 10.47 + `Δ` 1.99; 69% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.02079, b=0.1032, c=-66.47, d=7.806, Δ=2.93
- persistence 0.00, DBCV contribution -0.0010  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0087                                    0.2223
b     -0.3216  0.1236                                  
c      0.1945 -0.0444  8.5299                          
d      0.5106  0.0194  0.1343  2.8520                  
delta -0.6597  0.0097 -0.1915  0.0260  2.5008          
```

### AH — fast tonic

- size 510, mean membership prob 0.878, noise adjacency 0.014
- medians: rate 115.8 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 230 ISIs
- drive: median `I` 16.62 = `I*` 9.50 + `Δ` 7.07; 50% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.111, b=0.1132, c=-69.96, d=4.767, Δ=7.163
- persistence 0.00, DBCV contribution -0.0022  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 187 parameter-space components, 2 of them holding >= 5% of the cluster, largest 9%
    - n=46: a=0.1312, b=0.2314, c=-69.33, d=6.613, Δ=8.828
    - n=34: a=0.1344, b=0.1046, c=-72.44, d=8.786, Δ=10.04

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0525                                    0.2988
b      0.1113  0.1324                                  
c      0.2567 -0.0305  6.5709                          
d      0.5026  0.0383  0.1478  3.5574                  
delta -0.0235  0.0803 -0.1376  0.7468  3.2650          
```

### AI — slow tonic

- size 501, mean membership prob 0.871, noise adjacency 0.012
- medians: rate 14.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 27 ISIs
- drive: median `I` 13.03 = `I*` 9.31 + `Δ` 2.79; 72% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.01931, b=0.1203, c=-66.41, d=7.74, Δ=3.298
- persistence 0.00, DBCV contribution -0.0025  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0070                                    0.1370
b     -0.1992  0.1298                                  
c      0.1772 -0.0069  7.8678                          
d      0.2367  0.0763  0.2396  2.8099                  
delta -0.5891 -0.1236  0.0169  0.4866  1.9320          
```

### AJ — fast tonic

- size 493, mean membership prob 0.731, noise adjacency 0.002
- medians: rate 86.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 172 ISIs
- drive: median `I` 12.67 = `I*` 8.17 + `Δ` 4.55; 47% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1358, b=0.1255, c=-70.47, d=5.944, Δ=4.748
- persistence 0.60, DBCV contribution +0.0001  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 93 parameter-space components, 2 of them holding >= 5% of the cluster, largest 55%
    - n=271: a=0.1437, b=0.2018, c=-70.9, d=6.536, Δ=4.978
    - n=40: a=0.1096, b=0.001635, c=-74.9, d=4.565, Δ=4.746

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0394                                    0.1629
b      0.2265  0.1307                                  
c      0.2590  0.0317  6.4042                          
d      0.2577  0.1110  0.2144  3.6195                  
delta -0.1457  0.0371  0.0406  0.8520  2.0156          
```

### AK — fast tonic

- size 491, mean membership prob 0.924, noise adjacency 0.000
- medians: rate 96.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 191 ISIs
- drive: median `I` 14.61 = `I*` 9.39 + `Δ` 5.18; 46% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1301, b=0.1177, c=-70.66, d=5.518, Δ=5.362
- persistence 0.60, DBCV contribution +0.0000  **MULTIMODAL** (4 components)
- pooled mean is **not interpretable**: 139 parameter-space components, 4 of them holding >= 5% of the cluster, largest 18%
    - n=86: a=0.1442, b=0.2287, c=-66.14, d=8.679, Δ=7.18
    - n=42: a=0.1308, b=0.0009477, c=-71.92, d=10.04, Δ=8.685
    - n=39: a=0.1292, b=0.2166, c=-71.86, d=2.444, Δ=3.952
    - n=34: a=0.1408, b=0.0007309, c=-67.6, d=5.031, Δ=4.921

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0433                                    0.1977
b      0.2266  0.1307                                  
c      0.2064 -0.0174  6.1689                          
d      0.4022  0.1575  0.1417  3.7163                  
delta  0.0572  0.1475 -0.0097  0.8767  2.4023          
```

### AL — medium tonic

- size 390, mean membership prob 0.947, noise adjacency 0.036
- medians: rate 21.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 41 ISIs
- drive: median `I` 13.65 = `I*` 6.35 + `Δ` 7.13; 83% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.015, b=0.1541, c=-61.73, d=5.915, Δ=7.358
- persistence 0.00, DBCV contribution +0.0000  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 178 parameter-space components, 2 of them holding >= 5% of the cluster, largest 11%
    - n=42: a=0.01392, b=0.1567, c=-59.61, d=5.222, Δ=6.32
    - n=20: a=0.00893, b=0.2835, c=-66.53, d=5.335, Δ=10.82

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0090                                    0.2233
b      0.0613  0.1256                                  
c      0.7036 -0.0314  8.5887                          
d      0.4447 -0.0726  0.3834  2.9795                  
delta -0.5153 -0.2043 -0.3356  0.3990  3.8242          
```

### AM — fast tonic

- size 370, mean membership prob 0.723, noise adjacency 0.000
- medians: rate 134.0 Hz, CV(ISI) 0.001, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 267 ISIs
- drive: median `I` 15.99 = `I*` 8.53 + `Δ` 7.73; 49% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1194, b=0.1251, c=-69.95, d=4.549, Δ=7.969
- persistence 0.60, DBCV contribution +0.0002  **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 129 parameter-space components, 3 of them holding >= 5% of the cluster, largest 11%
    - n=42: a=0.122, b=0.1441, c=-73.05, d=2.358, Δ=6.452
    - n=28: a=0.1097, b=0.308, c=-74.19, d=3.429, Δ=8.58
    - n=22: a=0.111, b=0.2566, c=-66.18, d=5.344, Δ=9.977

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0488                                    0.2566
b      0.0550  0.1289                                  
c      0.1596  0.0284  6.6663                          
d      0.4990  0.0496  0.1368  3.2983                  
delta  0.0526  0.1983 -0.1125  0.7988  3.1788          
```

### AN — slow tonic

- size 356, mean membership prob 0.967, noise adjacency 0.022
- medians: rate 9.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 18 ISIs
- drive: median `I` 13.65 = `I*` 11.80 + `Δ` 1.63; 65% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.01731, b=0.09356, c=-65.18, d=8.429, Δ=1.995
- persistence 0.40, DBCV contribution -0.0017  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0047                                    0.1146
b     -0.4087  0.1327                                  
c      0.0458  0.0726  8.6215                          
d      0.3073  0.1845  0.1901  2.3960                  
delta -0.5765 -0.0412  0.0096  0.2653  1.0289          
```

### AO — slow tonic

- size 354, mean membership prob 0.925, noise adjacency 0.011
- medians: rate 19.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 38 ISIs
- drive: median `I` 13.55 = `I*` 4.57 + `Δ` 8.83; 88% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.01299, b=0.173, c=-57.55, d=6.432, Δ=8.751
- persistence 0.00, DBCV contribution +0.0055  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 100 parameter-space components, 3 of them holding >= 5% of the cluster, largest 36%
    - n=129: a=0.008705, b=0.2256, c=-59.51, d=4.975, Δ=9.507
    - n=36: a=0.01401, b=0.1547, c=-53.34, d=8.594, Δ=9.816
    - n=19: a=0.01308, b=-0.00521, c=-52.66, d=10.1, Δ=12.38

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0082                                    0.2278
b      0.0889  0.1215                                  
c      0.6446 -0.0363  7.1338                          
d      0.5289 -0.0992  0.5439  2.9422                  
delta -0.6269 -0.2370 -0.3862  0.1968  3.9814          
```

### AP — slow doublet bursting (2.1 spikes/burst, 9 Hz interburst)

- size 305, mean membership prob 0.789, noise adjacency 0.007
- medians: rate 19.0 Hz, CV(ISI) 0.993, burst index 0.514, spikes/burst 2.06, adaptation +0.000, 37 ISIs
- drive: median `I` 13.36 = `I*` 8.74 + `Δ` 4.14; 73% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.01826, b=0.123, c=-44.02, d=9.127, Δ=5.296
- persistence 0.00, DBCV contribution +0.0000  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 66 parameter-space components, 2 of them holding >= 5% of the cluster, largest 32%
    - n=99: a=0.01883, b=0.0896, c=-43.34, d=9.724, Δ=5.01
    - n=17: a=0.02241, b=0.3009, c=-42.63, d=9.969, Δ=1.751

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0090                                    0.2168
b     -0.2046  0.1308                                  
c      0.4724  0.0936  2.1011                          
d      0.3611 -0.0237  0.8206  1.9453                  
delta -0.7011 -0.0095 -0.0684  0.0184  3.7899          
```

### AQ — slow doublet bursting (2.1 spikes/burst, 5 Hz interburst)

- size 300, mean membership prob 0.842, noise adjacency 0.000
- medians: rate 11.0 Hz, CV(ISI) 1.027, burst index 0.524, spikes/burst 2.10, adaptation +0.000, 21 ISIs
- drive: median `I` 14.81 = `I*` 11.50 + `Δ` 2.42; 70% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.009101, b=0.1081, c=-46.21, d=8.16, Δ=3.162
- persistence 0.00, DBCV contribution +0.0023  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 104 parameter-space components, 3 of them holding >= 5% of the cluster, largest 18%
    - n=53: a=0.00965, b=0.03067, c=-46.1, d=8.382, Δ=3.707
    - n=34: a=0.008437, b=0.1642, c=-46.08, d=8.241, Δ=2.349
    - n=23: a=0.008563, b=0.2669, c=-45.09, d=9.067, Δ=2.171

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0033                                    0.1447
b     -0.2368  0.1306                                  
c      0.2656  0.0274  2.5589                          
d      0.1913  0.0099  0.8327  2.3817                  
delta -0.4380 -0.1349  0.2085  0.2576  2.3689          
```

### AR — fast tonic, +irregular

- size 300, mean membership prob 0.878, noise adjacency 0.003
- medians: rate 96.8 Hz, CV(ISI) 0.620, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 192 ISIs
- drive: median `I` 14.21 = `I*` 7.17 + `Δ` 5.99; 59% Class 2; bursting members 0%, irregular 100%
- mean coefficient vector: a=0.09878, b=0.1396, c=-53.03, d=5.996, Δ=6.755
- persistence 0.40, DBCV contribution +0.0002  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 139 parameter-space components, 3 of them holding >= 5% of the cluster, largest 9%
    - n=27: a=0.1031, b=0.2366, c=-52.98, d=4.942, Δ=5.571
    - n=24: a=0.118, b=0.0259, c=-52.74, d=6.724, Δ=6.622
    - n=19: a=0.1481, b=0.2059, c=-50.44, d=8.695, Δ=8.519

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0521                                    0.3024
b      0.0110  0.1270                                  
c      0.7299  0.0984  2.6478                          
d      0.5179 -0.1466  0.8779  2.9907                  
delta -0.2884  0.0967  0.1047  0.3563  3.7858          
```

### AT — fast tonic

- size 294, mean membership prob 0.846, noise adjacency 0.007
- medians: rate 126.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, 251 ISIs
- drive: median `I` 17.80 = `I*` 11.14 + `Δ` 7.06; 40% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1287, b=0.09914, c=-70.77, d=4.869, Δ=7.107
- persistence 0.60, DBCV contribution +0.0000  **MULTIMODAL** (5 components)
- pooled mean is **not interpretable**: 114 parameter-space components, 5 of them holding >= 5% of the cluster, largest 8%
    - n=23: a=0.1367, b=0.2455, c=-73.25, d=2.488, Δ=5.87
    - n=22: a=0.1349, b=0.05003, c=-73.04, d=4.595, Δ=7.39
    - n=16: a=0.1214, b=0.06078, c=-70.58, d=0.5595, Δ=3.205
    - n=16: a=0.1349, b=0.2464, c=-76.67, d=6.675, Δ=9.172
    - n=16: a=0.08956, b=0.03673, c=-72.22, d=3.309, Δ=7.197

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0436                                    0.2015
b      0.2669  0.1326                                  
c      0.2104 -0.2017  5.6230                          
d      0.3862  0.1631 -0.0054  3.6170                  
delta  0.1040  0.2240 -0.2554  0.8951  3.2391          
```

### AU — medium tonic

- size 289, mean membership prob 0.831, noise adjacency 0.007
- medians: rate 43.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 86 ISIs
- drive: median `I` 8.05 = `I*` 5.81 + `Δ` 2.20; 64% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1019, b=0.1538, c=-65.85, d=8.253, Δ=2.315
- persistence 0.00, DBCV contribution +0.0045  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 123 parameter-space components, 2 of them holding >= 5% of the cluster, largest 10%
    - n=28: a=0.1069, b=0.05969, c=-62.19, d=9.379, Δ=2.635
    - n=22: a=0.09228, b=0.1675, c=-61, d=5.848, Δ=2.088

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0231                                    0.0944
b      0.0752  0.1219                                  
c      0.1417  0.1349  8.1140                          
d      0.2587  0.0360  0.1784  2.5084                  
delta -0.6207 -0.4581 -0.0126  0.3998  0.9214          
```

### AV — medium tonic

- size 288, mean membership prob 0.956, noise adjacency 0.038
- medians: rate 24.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 47 ISIs
- drive: median `I` 12.54 = `I*` 7.02 + `Δ` 4.33; 77% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.01838, b=0.1387, c=-70, d=5.025, Δ=5.243
- persistence 0.00, DBCV contribution +0.0000  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0060                                    0.1300
b      0.1936  0.1267                                  
c      0.6466  0.0019  6.7343                          
d      0.2496  0.0037  0.0918  3.3399                  
delta -0.0524 -0.1754 -0.1693  0.9104  3.3117          
```

### AY — medium tonic

- size 279, mean membership prob 0.748, noise adjacency 0.011
- medians: rate 22.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 43 ISIs
- drive: median `I` 12.84 = `I*` 8.27 + `Δ` 4.20; 73% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.0211, b=0.1263, c=-69.89, d=6.052, Δ=4.58
- persistence 0.20, DBCV contribution +0.0043  **UNSTABLE** **MULTIMODAL** (6 components)
- pooled mean is **not interpretable**: 61 parameter-space components, 6 of them holding >= 5% of the cluster, largest 32%
    - n=88: a=0.02143, b=0.1478, c=-71.99, d=4.479, Δ=3.3
    - n=22: a=0.02049, b=-0.03327, c=-71.38, d=8.853, Δ=7.47
    - n=20: a=0.02477, b=0.2981, c=-74.46, d=3.6, Δ=1.777
    - n=19: a=0.01997, b=0.1277, c=-64.09, d=10.03, Δ=7.95
    - n=18: a=0.02136, b=0.256, c=-63.7, d=7.68, Δ=4.913
    - n=17: a=0.02045, b=-0.03014, c=-66.93, d=2.47, Δ=2.467

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0034                                    0.0772
b      0.3632  0.1310                                  
c      0.1172 -0.0593  5.9452                          
d     -0.0952  0.1163  0.1112  3.1786                  
delta -0.2829 -0.1419  0.0782  0.9460  2.5177          
```

### BA — medium doublet bursting (2.0 spikes/burst, 13 Hz interburst)

- size 278, mean membership prob 0.984, noise adjacency 0.025
- medians: rate 26.0 Hz, CV(ISI) 0.963, burst index 0.509, spikes/burst 2.04, adaptation +0.000, 51 ISIs
- drive: median `I` 13.39 = `I*` 8.92 + `Δ` 4.09; 70% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.0248, b=0.1233, c=-46.93, d=7.882, Δ=4.96
- persistence 0.00, DBCV contribution -0.0025  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0159                                    0.2813
b     -0.1493  0.1334                                  
c      0.7033  0.0814  3.1397                          
d      0.5054 -0.0387  0.8684  2.6936                  
delta -0.4477 -0.0311  0.0552  0.1292  3.4984          
```

### BB — fast tonic, +irregular

- size 273, mean membership prob 0.934, noise adjacency 0.026
- medians: rate 86.0 Hz, CV(ISI) 0.690, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 171 ISIs
- drive: median `I` 13.70 = `I*` 7.14 + `Δ` 5.75; 63% Class 2; bursting members 0%, irregular 100%
- mean coefficient vector: a=0.09252, b=0.1379, c=-52.1, d=6.234, Δ=6.431
- persistence 0.00, DBCV contribution +0.0000  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 162 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0541                                    0.3463
b      0.2201  0.1309                                  
c      0.7524  0.2987  3.1240                          
d      0.5115  0.0258  0.8928  3.1630                  
delta -0.2879 -0.0594  0.1335  0.3909  4.0598          
```

### BC — slow tonic

- size 246, mean membership prob 0.917, noise adjacency 0.000
- medians: rate 6.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 11 ISIs
- drive: median `I` 11.67 = `I*` 8.32 + `Δ` 2.98; 78% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.006115, b=0.1349, c=-67.86, d=7.736, Δ=3.035
- persistence 0.60, DBCV contribution +0.0001  **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 73 parameter-space components, 3 of them holding >= 5% of the cluster, largest 31%
    - n=77: a=0.006152, b=0.2668, c=-68.13, d=8.275, Δ=2.744
    - n=16: a=0.00589, b=0.1499, c=-61.17, d=10.45, Δ=3.582
    - n=14: a=0.006515, b=0.2755, c=-66.35, d=5.397, Δ=1.475

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0005                                    0.0350
b      0.0409  0.1299                                  
c      0.0732  0.0067  7.4208                          
d     -0.2352  0.1987  0.1133  2.5735                  
delta -0.2883 -0.1360  0.0402  0.7565  1.3349          
```

### BD — medium tonic

- size 235, mean membership prob 0.930, noise adjacency 0.000
- medians: rate 64.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 128 ISIs
- drive: median `I` 8.87 = `I*` 5.47 + `Δ` 3.05; 57% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1445, b=0.1634, c=-67.41, d=7.616, Δ=3.182
- persistence 0.40, DBCV contribution +0.0001  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 96 parameter-space components, 3 of them holding >= 5% of the cluster, largest 13%
    - n=31: a=0.1477, b=0.2957, c=-59.66, d=8.887, Δ=3.21
    - n=15: a=0.1557, b=0.2942, c=-76.06, d=8.214, Δ=2.554
    - n=12: a=0.148, b=0.2027, c=-63.74, d=5.274, Δ=2.523

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0286                                    0.0950
b      0.2700  0.1230                                  
c      0.1188  0.0972  8.0187                          
d     -0.0181  0.0555  0.2453  2.9637                  
delta -0.5289 -0.2996  0.2106  0.7500  1.1519          
```

### BE — medium tonic

- size 234, mean membership prob 0.939, noise adjacency 0.013
- medians: rate 70.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, 140 ISIs
- drive: median `I` 12.40 = `I*` 8.98 + `Δ` 2.94; 39% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1617, b=0.1176, c=-68.98, d=6.346, Δ=3.027
- persistence 0.60, DBCV contribution +0.0036  **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 84 parameter-space components, 3 of them holding >= 5% of the cluster, largest 10%
    - n=24: a=0.1703, b=0.1622, c=-72.67, d=7.187, Δ=3.029
    - n=16: a=0.1537, b=-0.04823, c=-74.6, d=2.139, Δ=1.773
    - n=13: a=0.1596, b=-0.03415, c=-69.95, d=4.044, Δ=2.526

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0256                                    0.0757
b      0.2785  0.1333                                  
c      0.1203  0.1687  7.0412                          
d      0.0643  0.1270  0.2490  3.4038                  
delta -0.3044 -0.0960  0.1875  0.8587  1.1424          
```

### BF — medium tonic

- size 233, mean membership prob 0.922, noise adjacency 0.017
- medians: rate 31.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 62 ISIs
- drive: median `I` 10.79 = `I*` 7.19 + `Δ` 3.78; 76% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.03959, b=0.1481, c=-69.12, d=6.606, Δ=3.904
- persistence 0.60, DBCV contribution +0.0000  **MULTIMODAL** (4 components)
- pooled mean is **not interpretable**: 73 parameter-space components, 4 of them holding >= 5% of the cluster, largest 10%
    - n=24: a=0.04451, b=0.2496, c=-74.81, d=3.862, Δ=1.862
    - n=20: a=0.03876, b=0.2954, c=-70.38, d=6.792, Δ=3.493
    - n=18: a=0.03815, b=0.1287, c=-61.4, d=10.14, Δ=6.068
    - n=16: a=0.03648, b=0.02753, c=-66.7, d=8.985, Δ=6.134

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0054                                    0.0581
b      0.3484  0.1278                                  
c      0.0199 -0.0163  6.8663                          
d     -0.2610  0.0435  0.1617  3.1004                  
delta -0.4820 -0.2499  0.1513  0.9185  1.9214          
```

### BG — fast tonic

- size 232, mean membership prob 0.883, noise adjacency 0.009
- medians: rate 259.8 Hz, CV(ISI) 0.223, burst index 0.000, spikes/burst 1.00, adaptation -0.000, 518 ISIs
- drive: median `I` 16.68 = `I*` 9.86 + `Δ` 7.06; 52% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1007, b=0.1127, c=-47.07, d=4.704, Δ=7.37
- persistence 0.60, DBCV contribution +0.0002  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 154 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0548                                    0.3351
b      0.0132  0.1285                                  
c      0.0482 -0.0271  4.3067                          
d      0.7737 -0.2882  0.1463  2.7609                  
delta  0.0172  0.0130 -0.1646  0.4282  4.2226          
```

### BH — medium tonic

- size 222, mean membership prob 0.994, noise adjacency 0.014
- medians: rate 48.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 96 ISIs
- drive: median `I` 8.55 = `I*` 6.56 + `Δ` 2.08; 58% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1212, b=0.1484, c=-66.18, d=7.864, Δ=2.249
- persistence 0.00, DBCV contribution +0.0000  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0289                                    0.0971
b     -0.1236  0.1232                                  
c      0.0773  0.0719  7.9902                          
d      0.0855  0.0722  0.1974  2.7199                  
delta -0.5911 -0.2839  0.1002  0.5685  0.9022          
```

### BI — medium doublet bursting (2.0 spikes/burst, 20 Hz interburst)

- size 219, mean membership prob 0.966, noise adjacency 0.041
- medians: rate 41.0 Hz, CV(ISI) 0.936, burst index 0.506, spikes/burst 2.02, adaptation +0.000, 81 ISIs
- drive: median `I` 11.04 = `I*` 5.97 + `Δ` 3.83; 78% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.04173, b=0.1502, c=-46.18, d=8.197, Δ=5.04
- persistence 0.00, DBCV contribution +0.0000  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 117 parameter-space components, 2 of them holding >= 5% of the cluster, largest 6%
    - n=13: a=0.06357, b=0.2811, c=-42.26, d=10.22, Δ=3.071
    - n=11: a=0.0195, b=0.04785, c=-48.55, d=6.055, Δ=8.045

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0232                                    0.2711
b     -0.0932  0.1277                                  
c      0.7036  0.1329  3.0290                          
d      0.5289 -0.0117  0.8618  2.4763                  
delta -0.5638 -0.0302 -0.0217  0.0531  3.7677          
```

### BJ — slow doublet bursting (2.1 spikes/burst, 6 Hz interburst)

- size 214, mean membership prob 0.788, noise adjacency 0.000
- medians: rate 12.0 Hz, CV(ISI) 1.020, burst index 0.522, spikes/burst 2.09, adaptation +0.000, 23 ISIs
- drive: median `I` 12.18 = `I*` 8.82 + `Δ` 2.78; 76% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.01023, b=0.1276, c=-45.58, d=8.462, Δ=3.65
- persistence 0.40, DBCV contribution +0.0032  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 77 parameter-space components, 3 of them holding >= 5% of the cluster, largest 15%
    - n=32: a=0.01508, b=-0.002953, c=-43.87, d=10.31, Δ=2.226
    - n=31: a=0.008881, b=0.2455, c=-43.29, d=10.01, Δ=4.494
    - n=11: a=0.009525, b=0.2814, c=-46.02, d=7.991, Δ=2.657

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0039                                    0.1624
b     -0.3750  0.1301                                  
c      0.2249  0.0249  3.0417                          
d      0.2476 -0.0603  0.8410  2.4609                  
delta -0.6657  0.0376  0.2033  0.2499  2.7147          
```

### BK — medium tonic

- size 210, mean membership prob 0.880, noise adjacency 0.000
- medians: rate 47.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 93 ISIs
- drive: median `I` 11.22 = `I*` 9.57 + `Δ` 1.54; 40% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.1487, b=0.1166, c=-65, d=8.427, Δ=1.695
- persistence 0.00, DBCV contribution +0.0004  **UNSTABLE** **MULTIMODAL** (4 components)
- pooled mean is **not interpretable**: 85 parameter-space components, 4 of them holding >= 5% of the cluster, largest 9%
    - n=19: a=0.1511, b=0.2313, c=-68.42, d=9.461, Δ=1.348
    - n=15: a=0.148, b=0.1212, c=-55.8, d=8.716, Δ=1.808
    - n=11: a=0.1403, b=0.01593, c=-57.48, d=7.146, Δ=1.963
    - n=11: a=0.1487, b=0.2734, c=-52.41, d=8.898, Δ=1.321

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0230                                    0.0675
b     -0.1296  0.1255                                  
c      0.0024  0.2656  8.9469                          
d     -0.0025  0.2332  0.0221  2.3273                  
delta -0.5578 -0.4428 -0.0509  0.3986  0.5433          
```

### BL — slow tonic

- size 189, mean membership prob 0.997, noise adjacency 0.011
- medians: rate 6.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 12 ISIs
- drive: median `I` 11.80 = `I*` 8.43 + `Δ` 3.43; 79% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.005742, b=0.1336, c=-68.72, d=7.139, Δ=3.699
- persistence 0.60, DBCV contribution +0.0000  **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 72 parameter-space components, 3 of them holding >= 5% of the cluster, largest 11%
    - n=20: a=0.005699, b=0.2585, c=-64.66, d=7.429, Δ=3.128
    - n=19: a=0.005915, b=0.2231, c=-72.05, d=4.059, Δ=1.534
    - n=10: a=0.005608, b=0.2505, c=-59.95, d=10.91, Δ=5.401

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0005                                    0.0376
b      0.1157  0.1263                                  
c     -0.0263  0.0850  6.6397                          
d     -0.1890 -0.0512  0.2103  2.8209                  
delta -0.2111 -0.2735  0.1490  0.8723  1.8973          
```

### BM — medium doublet bursting (2.0 spikes/burst, 13 Hz interburst)

- size 189, mean membership prob 0.981, noise adjacency 0.032
- medians: rate 27.0 Hz, CV(ISI) 0.948, burst index 0.508, spikes/burst 2.03, adaptation +0.000, 53 ISIs
- drive: median `I` 13.28 = `I*` 8.29 + `Δ` 3.48; 72% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.02951, b=0.1269, c=-48, d=7.141, Δ=4.879
- persistence 0.00, DBCV contribution +0.0000  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 130 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0240                                    0.3417
b     -0.2217  0.1259                                  
c      0.7507 -0.0093  3.6799                          
d      0.5947 -0.1030  0.9128  2.9700                  
delta -0.4415  0.0816  0.0378  0.1286  3.6284          
```

### BN — slow tonic

- size 187, mean membership prob 0.913, noise adjacency 0.000
- medians: rate 6.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 11 ISIs
- drive: median `I` 11.06 = `I*` 8.45 + `Δ` 2.27; 82% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.00704, b=0.1366, c=-51.39, d=8.024, Δ=2.689
- persistence 0.60, DBCV contribution +0.0002  **MULTIMODAL** (4 components)
- pooled mean is **not interpretable**: 64 parameter-space components, 4 of them holding >= 5% of the cluster, largest 17%
    - n=31: a=0.0068, b=0.2551, c=-50.42, d=8.943, Δ=2.428
    - n=28: a=0.006448, b=0.09733, c=-50.2, d=10.18, Δ=4.419
    - n=17: a=0.006137, b=0.2847, c=-54.37, d=5.034, Δ=1.672
    - n=16: a=0.00696, b=0.09843, c=-54.09, d=4.705, Δ=1.936

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0018                                    0.1003
b     -0.1783  0.1244                                  
c      0.4538  0.0164  2.8768                          
d      0.2097  0.0604  0.7733  2.6443                  
delta -0.3910 -0.1057  0.1554  0.4603  1.5070          
```

### BO — medium doublet bursting (2.0 spikes/burst, 24 Hz interburst)

- size 185, mean membership prob 0.981, noise adjacency 0.016
- medians: rate 49.0 Hz, CV(ISI) 0.898, burst index 0.505, spikes/burst 2.02, adaptation +0.000, 97 ISIs
- drive: median `I` 13.43 = `I*` 8.74 + `Δ` 3.23; 64% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.05402, b=0.1282, c=-48.11, d=7.262, Δ=4.654
- persistence 0.60, DBCV contribution +0.0000  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 123 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0335                                    0.2954
b      0.0776  0.1339                                  
c      0.7115  0.3887  3.3678                          
d      0.5211  0.2273  0.8917  2.8009                  
delta -0.4828  0.0419  0.0432  0.1412  3.4916          
```

### BP — medium tonic

- size 175, mean membership prob 0.999, noise adjacency 0.011
- medians: rate 20.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 39 ISIs
- drive: median `I` 13.31 = `I*` 7.88 + `Δ` 4.70; 77% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.01392, b=0.132, c=-70.4, d=4.887, Δ=5.297
- persistence 0.40, DBCV contribution +0.0027  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 70 parameter-space components, 3 of them holding >= 5% of the cluster, largest 12%
    - n=21: a=0.01094, b=-0.02238, c=-71.26, d=3.028, Δ=4.5
    - n=17: a=0.01413, b=0.2451, c=-73.29, d=4.976, Δ=5.263
    - n=11: a=0.01469, b=0.2153, c=-71.64, d=1.522, Δ=1.592

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0033                                    0.1126
b      0.3722  0.1278                                  
c      0.4809 -0.0679  6.0551                          
d      0.2800  0.0085  0.0291  3.2926                  
delta  0.0701 -0.1863 -0.1159  0.9569  3.2334          
```

### BQ — medium tonic

- size 169, mean membership prob 0.964, noise adjacency 0.000
- medians: rate 63.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 125 ISIs
- drive: median `I` 11.02 = `I*` 5.33 + `Δ` 4.90; 72% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.07288, b=0.1563, c=-68.94, d=5.347, Δ=4.994
- persistence 0.00, DBCV contribution +0.0026  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 75 parameter-space components, 2 of them holding >= 5% of the cluster, largest 8%
    - n=14: a=0.06659, b=0.2491, c=-72.64, d=0.8941, Δ=1.851
    - n=9: a=0.07394, b=0.2579, c=-66.86, d=7.366, Δ=6.069

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0243                                    0.1672
b      0.3318  0.1231                                  
c      0.6716  0.0207  6.9875                          
d      0.0430 -0.0539  0.1419  3.5121                  
delta -0.1653 -0.1974 -0.0853  0.9411  2.6252          
```

### BR — medium doublet bursting (2.0 spikes/burst, 12 Hz interburst)

- size 169, mean membership prob 1.000, noise adjacency 0.012
- medians: rate 25.0 Hz, CV(ISI) 0.976, burst index 0.510, spikes/burst 2.04, adaptation +0.000, 49 ISIs
- drive: median `I` 13.63 = `I*` 8.51 + `Δ` 4.27; 74% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.02348, b=0.125, c=-45.32, d=8.406, Δ=5.268
- persistence 0.40, DBCV contribution +0.0000  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0129                                    0.2448
b     -0.1550  0.1298                                  
c      0.6345  0.1264  2.4889                          
d      0.4837 -0.0710  0.8669  2.2677                  
delta -0.6501  0.0029 -0.0298  0.0673  3.9197          
```

### BT — medium tonic

- size 162, mean membership prob 0.973, noise adjacency 0.031
- medians: rate 60.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, 119 ISIs
- drive: median `I` 13.29 = `I*` 7.53 + `Δ` 5.29; 73% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.05255, b=0.1395, c=-68.68, d=4.97, Δ=5.942
- persistence 0.00, DBCV contribution +0.0025  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 91 parameter-space components, 3 of them holding >= 5% of the cluster, largest 6%
    - n=10: a=0.04287, b=0.01784, c=-73.24, d=4.655, Δ=6.861
    - n=9: a=0.06038, b=0.2374, c=-65.3, d=2.383, Δ=3.358
    - n=9: a=0.05043, b=0.1161, c=-75.4, d=8.555, Δ=9.978

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0277                                    0.2396
b      0.2005  0.1174                                  
c      0.6997 -0.0093  7.2948                          
d      0.3280 -0.1568  0.1804  3.6071                  
delta -0.0140 -0.2686 -0.2019  0.8941  3.1323          
```

### BU — medium tonic

- size 161, mean membership prob 0.996, noise adjacency 0.006
- medians: rate 33.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 66 ISIs
- drive: median `I` 12.38 = `I*` 6.81 + `Δ` 4.99; 72% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.03432, b=0.132, c=-69.46, d=6.195, Δ=4.891
- persistence 0.00, DBCV contribution +0.0004  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0058                                    0.0878
b      0.3857  0.1292                                  
c      0.2341 -0.0079  6.0526                          
d     -0.0991  0.1065  0.2324  3.3413                  
delta -0.2136 -0.1085  0.1711  0.9648  2.5035          
```

### BV — medium doublet bursting (2.0 spikes/burst, 16 Hz interburst)

- size 159, mean membership prob 0.989, noise adjacency 0.025
- medians: rate 32.0 Hz, CV(ISI) 0.954, burst index 0.508, spikes/burst 2.03, adaptation +0.000, 63 ISIs
- drive: median `I` 14.05 = `I*` 8.86 + `Δ` 3.59; 72% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.03252, b=0.1247, c=-46.21, d=8.145, Δ=5.117
- persistence 0.40, DBCV contribution +0.0000  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 107 parameter-space components, 3 of them holding >= 5% of the cluster, largest 8%
    - n=12: a=0.02595, b=0.05491, c=-43.64, d=9.967, Δ=10.95
    - n=8: a=0.02094, b=0.2224, c=-48.52, d=5.989, Δ=3.115
    - n=8: a=0.04254, b=0.1832, c=-45.07, d=8.952, Δ=1.912

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0208                                    0.2744
b     -0.1618  0.1204                                  
c      0.6740  0.0663  2.7896                          
d      0.4060  0.0192  0.8369  2.4216                  
delta -0.5643 -0.0616 -0.0333  0.1183  3.7852          
```

### BY — slow doublet bursting (2.1 spikes/burst, 4 Hz interburst)

- size 156, mean membership prob 0.850, noise adjacency 0.000
- medians: rate 8.0 Hz, CV(ISI) 1.050, burst index 0.533, spikes/burst 2.14, adaptation +0.000, 15 ISIs
- drive: median `I` 14.34 = `I*` 11.48 + `Δ` 1.93; 65% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.007539, b=0.08887, c=-45.98, d=8.22, Δ=2.404
- persistence 1.00, DBCV contribution +0.0024  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 75 parameter-space components, 2 of them holding >= 5% of the cluster, largest 10%
    - n=16: a=0.00749, b=-0.04062, c=-48.32, d=5.682, Δ=2.3
    - n=11: a=0.007646, b=0.1913, c=-45.53, d=8.525, Δ=1.912

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0021                                    0.1130
b     -0.3246  0.1262                                  
c      0.1414  0.0504  3.0161                          
d      0.1788  0.0642  0.8235  2.4373                  
delta -0.4612 -0.0700  0.2341  0.1955  1.4199          
```

### CA — slow doublet bursting (2.1 spikes/burst, 4 Hz interburst)

- size 145, mean membership prob 1.000, noise adjacency 0.000
- medians: rate 9.0 Hz, CV(ISI) 1.043, burst index 0.529, spikes/burst 2.12, adaptation +0.000, 17 ISIs
- drive: median `I` 12.96 = `I*` 9.05 + `Δ` 2.51; 75% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.008616, b=0.127, c=-43.2, d=9.555, Δ=3.578
- persistence 0.00, DBCV contribution +0.0003  **UNSTABLE**

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0034                                    0.1506
b     -0.0865  0.1344                                  
c      0.1753  0.1881  1.7495                          
d      0.0648  0.1306  0.8020  1.7113                  
delta -0.3519 -0.1103  0.1902  0.1544  2.7985          
```

### CB — slow doublet bursting (2.1 spikes/burst, 7 Hz interburst)

- size 138, mean membership prob 0.966, noise adjacency 0.007
- medians: rate 15.0 Hz, CV(ISI) 1.004, burst index 0.517, spikes/burst 2.07, adaptation +0.000, 29 ISIs
- drive: median `I` 15.00 = `I*` 10.14 + `Δ` 3.70; 72% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.01489, b=0.1159, c=-44.76, d=8.742, Δ=4.712
- persistence 0.60, DBCV contribution +0.0021  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 94 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0087                                    0.2232
b     -0.2425  0.1252                                  
c      0.4206 -0.0029  2.6207                          
d      0.3180 -0.0215  0.8403  2.0989                  
delta -0.5406  0.0007  0.1060  0.1983  3.4078          
```

### CC — slow doublet bursting (2.1 spikes/burst, 7 Hz interburst)

- size 135, mean membership prob 1.000, noise adjacency 0.000
- medians: rate 14.0 Hz, CV(ISI) 1.000, burst index 0.519, spikes/burst 2.08, adaptation +0.000, 27 ISIs
- drive: median `I` 14.10 = `I*` 9.80 + `Δ` 3.51; 69% Class 2; bursting members 100%, irregular 100%
- mean coefficient vector: a=0.01526, b=0.12, c=-45.64, d=8.456, Δ=4.31
- persistence 0.00, DBCV contribution +0.0021  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 106 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0083                                    0.2261
b     -0.2026  0.1372                                  
c      0.3811  0.0588  2.8047                          
d      0.2477  0.0675  0.8150  2.4683                  
delta -0.5325  0.0060  0.0983  0.2036  3.0773          
```

### CD — slow tonic

- size 125, mean membership prob 1.000, noise adjacency 0.016
- medians: rate 18.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 35 ISIs
- drive: median `I` 11.66 = `I*` 8.33 + `Δ` 3.91; 75% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.0219, b=0.1357, c=-68.74, d=7.307, Δ=3.82
- persistence 0.00, DBCV contribution +0.0005  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 75 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0028                                    0.0553
b      0.3210  0.1326                                  
c     -0.2005 -0.0296  7.1979                          
d     -0.1662  0.0245  0.2512  2.7542                  
delta -0.5220 -0.3763  0.2465  0.8516  1.7784          
```

### CE — slow tonic

- size 125, mean membership prob 1.000, noise adjacency 0.008
- medians: rate 9.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, 17 ISIs
- drive: median `I` 11.82 = `I*` 8.94 + `Δ` 2.69; 78% Class 2; bursting members 0%, irregular 0%
- mean coefficient vector: a=0.01222, b=0.1239, c=-67.06, d=8.291, Δ=2.819
- persistence 0.00, DBCV contribution +0.0019  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 66 parameter-space components, 3 of them holding >= 5% of the cluster, largest 8%
    - n=10: a=0.01239, b=0.1383, c=-66, d=9.898, Δ=3.259
    - n=7: a=0.0117, b=0.3004, c=-54.57, d=9.64, Δ=2.766
    - n=7: a=0.01278, b=0.1147, c=-55.58, d=9.242, Δ=2.93

```
coefficient vector: SD on the diagonal, Pearson r below it
            a       b       c       d   delta  sd_log_a
a      0.0009                                    0.0307
b      0.0793  0.1231                                  
c     -0.0341 -0.0106  8.0142                          
d     -0.1084  0.2283  0.0177  2.4237                  
delta -0.4441 -0.3217  0.0440  0.7542  1.0731          
```

## 5. Nameability merge (§8.4)

114 clusters -> **71**: 61 of 6441 pairs were indistinguishable on every descriptor at the Bonferroni-corrected threshold α = 3.11e-07 (0.01 / (6441 pairs × 5 descriptors)).

| merged group (pre-merge ids) |
|---|
| 9, 55 |
| 11, 29, 58, 59 |
| 12, 56, 106 |
| 13, 36, 87, 103 |
| 17, 21, 81 |
| 18, 102, 107 |
| 20, 42, 92, 94 |
| 25, 95 |
| 27, 53, 67, 105, 109 |
| 28, 72 |
| 31, 68 |
| 34, 79, 80, 82, 96, 100, 101 |
| 40, 43, 113 |
| 49, 91, 111 |
| 52, 99, 108 |
| 57, 84 |
| 62, 66, 97, 98 |
| 74, 83 |
| 77, 104 |
| 78, 89 |
| 86, 93 |

Full pair table in `data/merge_pairs.csv`.

## 6. Canonical type placement (§8.3 — with teeth)

Each type must land in a cluster whose interquartile range contains the type's own measured `(CV, burst index, spikes/burst, rate)`.  v1's criterion ('lands in a non-noise cluster') passed while CH landed in a cluster named *slow tonic* (§2.2).

| type | a | b | c | d | I | cluster | p | CV | BI | spikes/burst | rate | consistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RS | 0.02 | 0.2 | -65 | 8 | 10.00 | AY | 1.000 | 1.45e-06 in [1.46e-06, 0.000159] **out** | 0 in [0, 0] ok | 1 in [1, 1] ok | 22.5 in [21, 23] ok | **no** (within whiskers) |
| IB | 0.02 | 0.2 | -55 | 4 | 10.00 | K | 1.000 | 1.46e-06 in [2.2e-06, 0.000173] **out** | 0 in [0, 0] ok | 1 in [1, 1] ok | 32 in [34, 36] **out** | **no** (within whiskers) |
| CH | 0.02 | 0.2 | -50 | 2 | 10.00 | B | 1.000 | 1.54 in [1.55, 2.18] **out** | 0.604 in [0.562, 0.814] ok | 2.52 in [2.28, 5.36] ok | 82.5 in [40.5, 111] ok | **no** (within whiskers) |
| FS | 0.1 | 0.2 | -65 | 2 | 10.00 | AM | 0.674 | 0.000979 in [8.59e-06, 0.000732] **out** | 0 in [0, 0] ok | 1 in [1, 1] ok | 136 in [130, 138] ok | **no** (within whiskers) |
| LTS | 0.02 | 0.25 | -65 | 2 | 10.00 | Q | 0.868 | 0.000372 in [4.43e-06, 0.000404] ok | 0 in [0, 0] ok | 1 in [1, 1] ok | 75 in [65.5, 70.5] **out** | **no** (within whiskers) |
| TC | 0.02 | 0.25 | -65 | 0.05 | 10.00 | A | 1.000 | 1.86e-05 in [1.6e-05, 0.00118] ok | 0 in [0, 0] ok | 1 in [1, 1] ok | 272 in [194, 460] ok | exempt |
| RZ | 0.1 | 0.26 | -65 | 2 | 10.00 | A | 0.943 | 1.27e-05 in [1.6e-05, 0.00118] **out** | 0 in [0, 0] ok | 1 in [1, 1] ok | 195 in [194, 460] ok | exempt |

TC and RZ are exempt **by name** (§8.3): their published behaviours need a hyperpolarising step and a timed brief stimulus respectively, neither reachable at constant drive (§11 item 1).

## 7. Stability (§8.2 — tests that probe something)

| test | varies | mean pairwise ARI | threshold | result |
|---|---|---|---|---|
| seed | UMAP random_state, features and sample set fixed (v1 §13.2); min_cluster_size 60 at subsample scale | 0.248 | 0.6 | FAIL |
| bootstrap | 80% subsamples, 5 draws, ARI on each pair's intersection | 0.190 | 0.6 | FAIL |
| window | analysis window shifted by +100 ms | 0.125 | 0.8 | FAIL |
| dt | dt_int halved | 0.370 | 0.9 | FAIL |

- UNSTABLE clusters (persistence < 0.6): A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, Q, R, T, U, V, Y, AA, AB, AC, AF, AG, AH, AI, AL, AN, AO, AP, AQ, AR, AU, AV, AY, BA, BB, BD, BH, BI, BJ, BK, BM, BP, BQ, BR, BT, BU, BV, CA, CC, CD, CE
- v1 §13.2 varied only the UMAP seed, so ARI 0.780 mostly established that UMAP is reproducible.  The window-shift and step-size tests compare perturbed against unperturbed features **at matched UMAP seeds**, so they measure the perturbation rather than seed noise.

## 8. Contiguity in parameter space (v1 §13.3)

- single-linkage cut at the 95th percentile of the global k=5 NN distance = 0.1349, in the **5-D** normalised design box (`Δ` is now a design axis)
- MULTIMODAL clusters: L (210 components, largest 48%), M (294 components, largest 11%), O (315 components, largest 6%), R (320 components, largest 8%), U (228 components, largest 14%), AA (144 components, largest 31%), AC (158 components, largest 34%), AD (144 components, largest 39%), AF (210 components, largest 14%), AH (187 components, largest 9%), AJ (93 components, largest 55%), AK (139 components, largest 18%), AL (178 components, largest 11%), AM (129 components, largest 11%), AO (100 components, largest 36%), AP (66 components, largest 32%), AQ (104 components, largest 18%), AR (139 components, largest 9%), AT (114 components, largest 8%), AU (123 components, largest 10%), AY (61 components, largest 32%), BB (162 components, largest 3%), BC (73 components, largest 31%), BD (96 components, largest 13%), BE (84 components, largest 10%), BF (73 components, largest 10%), BG (154 components, largest 4%), BI (117 components, largest 6%), BJ (77 components, largest 15%), BK (85 components, largest 9%), BL (72 components, largest 11%), BM (130 components, largest 3%), BN (64 components, largest 17%), BO (123 components, largest 4%), BP (70 components, largest 12%), BQ (75 components, largest 8%), BT (91 components, largest 6%), BV (107 components, largest 8%), BY (75 components, largest 10%), CB (94 components, largest 4%), CC (106 components, largest 3%), CD (75 components, largest 5%), CE (66 components, largest 8%)

## 10. eom → leaf containment (v1 §12.4)

| eom | size | leaf | containment |
|---|---|---|---|
| B | 4386 | -1 | 0.211 |
| BY | 156 | 0 | 1.000 |
| H | 1512 | -1 | 0.288 |
| BJ | 214 | 1 | 1.000 |
| CA | 145 | 2 | 1.000 |
| C | 3484 | -1 | 0.259 |
| AQ | 300 | 3 | 1.000 |
| CC | 135 | 4 | 1.000 |
| CB | 138 | 6 | 1.000 |
| BA | 278 | 91 | 0.568 |
| BQ | 169 | 8 | 1.000 |
| V | 629 | 10 | 0.342 |
| O | 740 | 92 | 0.389 |
| R | 657 | 161 | 0.347 |
| AM | 370 | 16 | 1.000 |
| A | 8267 | -1 | 0.263 |
| BU | 161 | 19 | 1.000 |
| U | 645 | 138 | 0.392 |
| L | 915 | 181 | 0.490 |
| BE | 234 | 27 | 1.000 |
| M | 846 | 167 | 0.304 |
| BG | 232 | 30 | 1.000 |
| T | 649 | 41 | 0.451 |
| BD | 235 | 32 | 1.000 |
| Q | 684 | 168 | 0.773 |
| AR | 300 | 35 | 1.000 |
| F | 2589 | -1 | 0.167 |
| AC | 591 | 44 | 0.723 |
| AU | 289 | 46 | 1.000 |
| AI | 501 | 49 | 0.535 |
| BR | 169 | 50 | 1.000 |
| AP | 305 | 51 | 1.000 |
| E | 3130 | -1 | 0.205 |
| BH | 222 | 63 | 1.000 |
| BK | 210 | 65 | 1.000 |
| BB | 273 | 68 | 1.000 |
| AB | 619 | 79 | 0.415 |
| D | 3453 | -1 | 0.334 |
| AY | 279 | 70 | 1.000 |
| AK | 491 | -1 | 0.407 |
| AJ | 493 | 73 | 1.000 |
| BF | 233 | 81 | 1.000 |
| BT | 162 | 82 | 1.000 |
| AE | 545 | 83 | 1.000 |
| N | 830 | 84 | 0.512 |
| AA | 620 | -1 | 0.377 |
| BN | 187 | 87 | 1.000 |
| K | 962 | 88 | 0.273 |
| BV | 159 | 90 | 1.000 |
| AF | 525 | 93 | 0.632 |
| Y | 622 | 119 | 0.418 |
| BO | 185 | 97 | 1.000 |
| G | 1592 | 107 | 0.388 |
| BL | 189 | 104 | 1.000 |
| BC | 246 | 105 | 1.000 |
| CD | 125 | 106 | 1.000 |
| AD | 556 | 111 | 1.000 |
| P | 693 | 112 | 1.000 |
| I | 1241 | -1 | 0.446 |
| CE | 125 | 114 | 1.000 |
| AN | 356 | 140 | 0.610 |
| BM | 189 | 117 | 1.000 |
| BI | 219 | 118 | 1.000 |
| AG | 524 | 137 | 0.275 |
| AH | 510 | 126 | 0.661 |
| BP | 175 | 159 | 1.000 |
| J | 1197 | 160 | 0.391 |
| AT | 294 | 162 | 1.000 |
| AV | 288 | 164 | 1.000 |
| AO | 354 | 190 | 1.000 |
| AL | 390 | 192 | 1.000 |

## 11. Selection sweep (v1 §12.4, stage 2 per §3.3)

| stage | representation | nn | mcs | ms | method | k | noise | rel. validity | score |
|---|---|---|---|---|---|---|---|---|---|
| 2 | h=0.25 wA=1 wB=1 wC=0.5 wD=0.3 wOn=0.3 | 50 | 120 | 5 | eom | 7 | 0.001 | 0.8654 | 0.8654 |
| 1 | stage1 | 50 | 120 | 5 | eom | 6 | 0.002 | 0.8375 | 0.8375 |
| 1 | stage1 | 50 | 30 | 25 | eom | 28 | 0.024 | 0.8353 | 0.8353 |
| 2 | h=0.25 wA=0.5 wB=1 wC=1 wD=0.3 wOn=0.3 | 50 | 120 | 5 | eom | 12 | 0.014 | 0.8214 | 0.8214 |
| 1 | stage1 | 50 | 120 | 25 | eom | 5 | 0.000 | 0.8142 | 0.8142 |
| 1 | stage1 | 50 | 60 | 25 | eom | 6 | 0.000 | 0.8046 | 0.8046 |
| 2 | h=0.4 wA=0.5 wB=1 wC=0.5 wD=0.3 wOn=0.3 | 50 | 120 | 5 | eom | 7 | 0.027 | 0.7980 | 0.7980 |
| 1 | stage1 | 15 | 60 | 25 | eom | 24 | 0.026 | 0.7267 | 0.7267 |
| 1 | stage1 | 50 | 120 | 10 | eom | 32 | 0.076 | 0.6523 | 0.6523 |
| 2 | h=0.25 wA=0 wB=1 wC=0.5 wD=0.3 wOn=0.3 | 50 | 120 | 5 | eom | 12 | 0.008 | 0.5938 | 0.5938 |
| 1 | stage1 | 30 | 120 | 5 | eom | 13 | 0.002 | 0.5684 | 0.5684 |
| 2 | h=0.25 wA=0.5 wB=1 wC=0.5 wD=0.3 wOn=0 | 50 | 120 | 5 | eom | 68 | 0.022 | 0.4971 | 0.4971 |
| 1 | stage1 | 15 | 60 | 5 | eom | 70 | 0.041 | 0.4709 | 0.4709 |
| 1 | stage1 | 15 | 120 | 25 | eom | 16 | 0.011 | 0.4690 | 0.4690 |
| 1 | stage1 | 15 | 30 | 25 | eom | 114 | 0.154 | 0.4679 | 0.4679 |
| 2 | h=0.25 wA=0.5 wB=1 wC=0.5 wD=0.3 wOn=1 | 50 | 120 | 5 | eom | 35 | 0.075 | 0.4616 | 0.4616 |
| 1 | stage1 | 50 | 60 | 10 | eom | 75 | 0.128 | 0.4493 | 0.4493 |
| 2 | h=0.25 wA=0.5 wB=1 wC=0.5 wD=0.1 wOn=0.3 | 50 | 120 | 5 | eom | 28 | 0.032 | 0.4339 | 0.4339 |
| 1 | stage1 | 30 | 120 | 10 | eom | 11 | 0.002 | 0.4273 | 0.4273 |
| 1 | stage1 | 30 | 60 | 25 | eom | 9 | 0.001 | 0.4273 | 0.4273 |

(top 20 of 64 configs; full table in `sweep_table.csv`)

## 12. Acceptance criteria (§9)

| criterion | result | detail |
|---|---|---|
| convergence gate passes all three checks (§5, fractional form §5.1) | PASS | dt_int 0.01 ms: counts identical for 91% of 23 used probes (worst |Δn| 1), per-ISI error under 0.05 ms for 100% (worst 1.81e-02 ms), within 1% rel L2 for 96% |
| zero samples labelled X; zero overflow flags set (§3.1) | PASS | X = 0, overflow = 0 (max spikes recorded 3609 of S_max = 8192) |
| noise fraction <= 25% at the selected config | PASS | noise = 0.150 |
| PCA components for 99% of variance <= 20 (§8.1) | PASS | 16 components (v1: 92 of 128) |
| metric monotonicity and octave gates (§8.1) | PASS | near/far ratio 0.047 (<= 0.5, v1 0.818); octave distance 0.016 vs 1.5x 0.011 (v1 0.379 vs 0.501, inverted) |
| fraction of reported clusters passing both the stability and contiguity gates >= 0.60 | FAIL | 2/71 = 0.028 (v1: 0.216 — 22 of 102) |
| distinct names after the §8.4 merge >= 8 | PASS | 8 distinct names across 71 clusters (v1: 6 of 102): fast tonic; fast tonic, +irregular; medium doublet bursting; medium multi-spike bursting, +adapting; medium tonic; medium triplet bursting, +adapting; slow doublet bursting; slow tonic |
| >= 1 cluster named bursting holding >= 50% of all BI > 0.2 samples | FAIL | best single bursting-named cluster holds 30.4% of the 13658 burst-index samples; all bursting-named clusters together 89.9%; those samples are spread over 19 clusters (v1: 86 of 102) |
| seed ARI >= 0.6 (§8.2) | FAIL | mean ARI 0.248 over 5 fits of 32000 points — UMAP random_state, features and sample set fixed (v1 §13.2); min_cluster_size 60 at subsample scale |
| bootstrap ARI >= 0.6 (§8.2) | FAIL | mean ARI 0.190 over 5 fits of 25600 points — 80% subsamples, 5 draws, ARI on each pair's intersection |
| window ARI >= 0.8 (§8.2) | FAIL | mean ARI 0.125 over 6 fits of 32000 points — analysis window shifted by +100 ms |
| dt ARI >= 0.9 (§8.2) | FAIL | mean ARI 0.370 over 6 fits of 15955 points — dt_int halved |
| canonical placement by descriptor consistency (§8.3), TC+RZ exempt | FAIL | 5 of 5 are inside the Tukey whiskers (RS, IB, CH, FS, LTS) and so are ordinary-spread misses rather than gross mismatches; inconsistent: RS in AY (cv_isi outside the host IQR), IB in K (cv_isi, rate outside the host IQR), CH in B (cv_isi outside the host IQR), FS in AM (cv_isi outside the host IQR), LTS in Q (rate outside the host IQR) |
| every artefact present and non-empty | PASS | all present |
| every row-per-sample file has exactly N rows and identical sample_id ordering | PASS | rows = {'samples.csv': 65536, 'features_isi.csv': 65536, 'features_isi_onset.csv': 65536, 'traces.csv': 65536, 'spike_times.csv': 65536}, first/last sample_id consistent = True |
| round trip: spike times -> features reproduces features_isi.csv to 1e-06 relative | PASS | max rel L2 0.00e+00 over 1002 rows (mean 0.00e+00) |

## 13. Timings

| stage | seconds |
|---|---|
| stability | 971.1 |
| csv_bulk | 310.4 |
| sweep_stage2 | 227.1 |
| umap10_seeded | 128.5 |
| features | 119.1 |
| umap2_seeded | 112.4 |
| figures | 104.6 |
| perturbed_features | 93.8 |
| sampling_loop | 83.8 |
| sweep_stage1 | 65.3 |
| simulate | 56.3 |
| simulate_dt_half | 52.9 |
| rheobase | 25.1 |
| csv_reps | 18.6 |
| merge | 11.3 |
| gate | 10.8 |
| csv_samples | 10.7 |
| csv_stats | 10.4 |
| representatives | 4.1 |
| health_gates | 2.7 |
| contiguity | 2.1 |
| hdbscan_final | 1.4 |
| isi_sets | 1.3 |
| roundtrip | 0.6 |
| pca | 0.2 |
| **total** | **2207.0** |

v1 spent 346.4 s (37.9%) in `gp_fit` + `acquire` against 85.0 s of integration (§2.6); C6 removes that stage entirely and spends the budget on samples and on validation instead.
