# Izhikevich spiking-neuron cluster discovery — run report

- Date: 2026-11-21 10:50:43
- Samples `N` = 26624 (12 iterations), drive `I` = 10.0
- Selected config: `n_neighbors=15`, `min_cluster_size=60`, `min_samples=25`, `cluster_selection_method='eom'`, spectral tilt `alpha=0.0`
- Clusters: **102** at `eom`, 114 at `leaf`; noise 7.7%; `relative_validity_` 0.5416
- Quarantine `Z` = 12017, non-finite `X` = 0, `P_floor` = 7.500e+05
- Mean pairwise ARI over 5 embeddings: **0.780**
- PCA: 92 components retaining 0.9902 of the variance (at 50 components: 0.8540; §19 A4)

## 1. Convergence gate (§7.4, amended per §19 A2/A3)

Probe set: 24 (17 spiking, 4 sensitivity-limited and excluded, 13 used).  Sensitivity baseline (dt=0.0005 vs 0.001 ms) median 6.83e-04.

| dt (ms) | median rel L2 | p90 | max | probes > 1% | max local error | mean spikes |
|---|---|---|---|---|---|---|
| 0.04 | 0.30049 | 0.6247 | 0.7234 | 11/13 | 3.58e-05 | 172.4 |
| 0.02 | 0.10573 | 0.2585 | 0.2819 | 9/13 | 1.47e-06 | 172.7 |
| 0.01 | 0.00942 | 0.1634 | 0.2078 | 6/13 | 5.26e-08 | 173.0 |
| 0.005 | 0.00589 | 0.0548 | 0.0996 | 5/13 | 1.79e-09 | 173.0 |
| 0.0025 | 0.00084 | 0.0104 | 0.0970 | 2/13 | 5.8e-11 | 173.0 |
| 0.001 | 0.00000 | 0.0000 | 0.0000 | 0/13 | 6.12e-13 | 173.0 |

- selected `dt_int` = **0.0025 ms** (coarsest ladder step with median <= 0.0025; the spec default 0.01 ms has median 0.00942)
- dt check: **PASS**
- aliasing: against a `dt_out = 0.0125` ms reference, filtered output is 0.0306 away and naive decimation 0.2160, i.e. the §7.2 filter buys **7.1x** (**PASS**, need >= 5x)
- literal `dt_out`-halving stability (reported, not gating): median 0.01174, max 0.0211
- `P_floor` = 7.4997e+05 = max(numerical 1.318e-01 from 18 silent references, physical 7.500e+05 for 0.1 mV RMS)

## 2. Quarantine `Z` and non-finite `X` (§10)

- `Z` (no spikes **and** in-band AC power below `P_floor` = 7.500e+05): **12017** samples (45.1%), held out before PCA/UMAP
- `X` (non-finite after the §7.3 refinement pass): **0** — this is an error-budget metric and should be zero
- zero-spike samples: 12034; of those, 17 are **retained** (AC power above `P_floor`) rather than quarantined, which is the subthreshold/resonant case §10 exists to preserve

Quarantined parameter region (mean ± SD, and range):

| coefficient | mean | SD | min | max |
|---|---|---|---|---|
| a | 0.06691 | 0.0578 | 0.005003 | 0.2 |
| b | 0.01158 | 0.06473 | -0.1 | 0.101 |
| c | -60.98 | 12.11 | -80 | -40 |
| d | 6.077 | 3.777 | 0.0001812 | 12 |

100.0% of quarantined samples have `b < 0.1010`, the value below which a rest state exists at `I = 10` (§2) — i.e. the quarantine is essentially the sub-bifurcation half of the box, as §2 predicts.

## 3. Clusters

Names follow §14.2; flags follow §13.2 (UNSTABLE) and §13.3 (MULTIMODAL).

### A — medium tonic

- size 365 (LHS-only 119), mean membership prob 0.948, noise adjacency 0.000
- medians: rate 34.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -63.3 mV, sd v 10.44 mV
- mean coefficient vector (all): a=0.0603, b=0.1997, c=-63.21, d=6.413
- mean coefficient vector (LHS-only, authoritative): a=0.04802, b=0.2185, c=-60.06, d=6.723
- persistence 0.20, DBCV contribution +0.0185  **UNSTABLE**

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0510                             0.4109
b -0.6909  0.0777                           
c  0.2178  0.0901  11.2851                  
d  0.2312  0.1179   0.1976  3.5857          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0380                             0.3157
b -0.6568  0.0681                           
c  0.2774  0.1182  11.3339                  
d  0.2162  0.0680   0.0975  3.1178          
```

### B — medium tonic

- size 355 (LHS-only 128), mean membership prob 0.799, noise adjacency 0.000
- medians: rate 58.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -60.8 mV, sd v 12.31 mV
- mean coefficient vector (all): a=0.08236, b=0.229, c=-62.59, d=6.764
- mean coefficient vector (LHS-only, authoritative): a=0.06909, b=0.2455, c=-62.19, d=7.128
- persistence 1.00, DBCV contribution +0.0105  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 113 parameter-space components, 2 of them holding >= 5% of the cluster, largest 15%
  component means:
    - n=53: a=0.1174, b=0.2694, c=-45.42, d=9.473
    - n=26: a=0.05642, b=0.2552, c=-71.97, d=10.19

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0536                             0.3651
b -0.5521  0.0724                           
c  0.2904  0.3155  11.5551                  
d  0.2706  0.3202   0.3869  3.5591          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0445                             0.3220
b -0.5493  0.0628                           
c  0.4386  0.1911  10.5453                  
d  0.2558  0.2161   0.2747  3.2241          
```

### C — slow tonic [mixed: 27% bursting]

- size 327 (LHS-only 165), mean membership prob 0.976, noise adjacency 0.000
- medians: rate 6.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -68.3 mV, sd v 6.47 mV
- mean coefficient vector (all): a=0.01225, b=0.1559, c=-60.41, d=6.857
- mean coefficient vector (LHS-only, authoritative): a=0.009412, b=0.1828, c=-54.75, d=6.97
- persistence 1.00, DBCV contribution +0.0169  **MULTIMODAL** (4 components)
- pooled mean is **not interpretable**: 90 parameter-space components, 4 of them holding >= 5% of the cluster, largest 14%
  component means:
    - n=46: a=0.01245, b=0.1404, c=-56.24, d=10.81
    - n=28: a=0.008826, b=0.1565, c=-77.56, d=9.498
    - n=27: a=0.007839, b=0.1383, c=-66.6, d=6.142
    - n=18: a=0.006078, b=0.2736, c=-44.13, d=9.275

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0094                             0.2439
b -0.5112  0.0589                           
c -0.1141  0.5068  12.4279                  
d  0.0225  0.0481  -0.0667  3.5097          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0049                             0.1867
b -0.6449  0.0639                           
c -0.0436  0.5053  12.1961                  
d  0.0071 -0.0937  -0.2676  3.3653          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.20 (> 0.2); trust the LHS-only matrix.

### D — medium tonic

- size 272 (LHS-only 96), mean membership prob 0.998, noise adjacency 0.000
- medians: rate 26.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.4 mV, sd v 9.56 mV
- mean coefficient vector (all): a=0.05251, b=0.1964, c=-62.03, d=6.707
- mean coefficient vector (LHS-only, authoritative): a=0.04036, b=0.2174, c=-59.88, d=7.006
- persistence 1.00, DBCV contribution +0.0157

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0484                             0.4443
b -0.7282  0.0800                           
c  0.2243  0.0026  11.1684                  
d  0.2162  0.0175   0.1975  3.4196          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0390                             0.3404
b -0.6653  0.0742                           
c  0.1701  0.0281  11.3317                  
d  0.0529  0.1906   0.0197  3.0804          
```

### E — slow tonic

- size 261 (LHS-only 110), mean membership prob 0.898, noise adjacency 0.000
- medians: rate 10.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.8 mV, sd v 7.32 mV
- mean coefficient vector (all): a=0.01782, b=0.1779, c=-62.72, d=7.234
- mean coefficient vector (LHS-only, authoritative): a=0.01308, b=0.2034, c=-57.44, d=6.913
- persistence 0.40, DBCV contribution +0.0142  **UNSTABLE**

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0175                             0.3244
b -0.5707  0.0715                           
c -0.0208  0.1902  10.6243                  
d  0.0808  0.2312  -0.0834  3.4944          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0112                             0.2614
b -0.6114  0.0682                           
c  0.1933  0.0350  10.6149                  
d  0.1164  0.1098  -0.2823  3.3448          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.21 (> 0.2); trust the LHS-only matrix.

### F — medium tonic [mixed: 20% bursting]

- size 247 (LHS-only 83), mean membership prob 0.978, noise adjacency 0.000
- medians: rate 38.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.9 mV, sd v 10.88 mV
- mean coefficient vector (all): a=0.06191, b=0.2108, c=-61.64, d=6.73
- mean coefficient vector (LHS-only, authoritative): a=0.04842, b=0.219, c=-60.44, d=6.498
- persistence 1.00, DBCV contribution +0.0109

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0494                             0.4011
b -0.6789  0.0775                           
c  0.2214  0.1230  11.8220                  
d  0.2427  0.1108   0.1976  3.5087          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0324                             0.3225
b -0.4692  0.0634                           
c  0.4378  0.2311  12.0918                  
d  0.3534  0.1370   0.0856  3.3995          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.22 (> 0.2); trust the LHS-only matrix.

### G — medium tonic

- size 247 (LHS-only 70), mean membership prob 0.850, noise adjacency 0.000
- medians: rate 50.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -61.6 mV, sd v 11.89 mV
- mean coefficient vector (all): a=0.07168, b=0.2327, c=-61.52, d=6.821
- mean coefficient vector (LHS-only, authoritative): a=0.06743, b=0.2354, c=-61.41, d=7.477
- persistence 0.40, DBCV contribution +0.0044  **UNSTABLE**

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0526                             0.3922
b -0.6473  0.0733                           
c  0.2915  0.2058  11.3860                  
d  0.2796  0.1298   0.3055  3.5251          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0443                             0.2938
b -0.6465  0.0633                           
c  0.2610  0.1878  10.6509                  
d  0.2324  0.1885   0.2365  2.9103          
```

### H — slow tonic [mixed: 22% bursting]

- size 246 (LHS-only 118), mean membership prob 0.697, noise adjacency 0.000
- medians: rate 11.2 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.9 mV, sd v 7.51 mV
- mean coefficient vector (all): a=0.01891, b=0.187, c=-61.76, d=7.23
- mean coefficient vector (LHS-only, authoritative): a=0.01582, b=0.2059, c=-56.25, d=6.85
- persistence 1.00, DBCV contribution +0.0066

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0172                             0.3116
b -0.6267  0.0734                           
c  0.0911  0.1477  12.3284                  
d -0.0508  0.2873  -0.0732  3.3321          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0132                             0.2623
b -0.6298  0.0699                           
c  0.1750  0.1270  12.1812                  
d -0.2820  0.2628  -0.2046  3.1288          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.23 (> 0.2); trust the LHS-only matrix.

### I — slow tonic [mixed: 36% bursting]

- size 239 (LHS-only 142), mean membership prob 0.995, noise adjacency 0.000
- medians: rate 5.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -68.4 mV, sd v 6.30 mV
- mean coefficient vector (all): a=0.01052, b=0.1353, c=-57.66, d=7.061
- mean coefficient vector (LHS-only, authoritative): a=0.009223, b=0.1484, c=-53.99, d=7.248
- persistence 1.00, DBCV contribution +0.0149  **MULTIMODAL** (4 components)
- pooled mean is **not interpretable**: 72 parameter-space components, 4 of them holding >= 5% of the cluster, largest 18%
  component means:
    - n=43: a=0.007233, b=0.1438, c=-50.62, d=3.674
    - n=22: a=0.008555, b=0.1566, c=-46.15, d=10.19
    - n=19: a=0.01607, b=0.1102, c=-52.55, d=11.05
    - n=13: a=0.008969, b=0.13, c=-71.99, d=7.707

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0057                             0.2021
b -0.5539  0.0366                           
c -0.1049  0.4351  11.9721                  
d  0.1999  0.1538  -0.0593  3.5584          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0046                             0.1781
b -0.5806  0.0405                           
c -0.0784  0.4319  11.3843                  
d  0.1251  0.1383  -0.1642  3.3760          
```

### J — medium tonic

- size 220 (LHS-only 73), mean membership prob 0.961, noise adjacency 0.000
- medians: rate 23.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.7 mV, sd v 9.24 mV
- mean coefficient vector (all): a=0.04304, b=0.1912, c=-62.32, d=6.466
- mean coefficient vector (LHS-only, authoritative): a=0.03031, b=0.2169, c=-57.75, d=6.834
- persistence 0.80, DBCV contribution +0.0021  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 147 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0390                             0.3997
b -0.6956  0.0761                           
c  0.0450  0.1503  10.8290                  
d  0.1506  0.0968   0.0106  3.3953          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0213                             0.2946
b -0.6984  0.0680                           
c  0.2711  0.0661  11.4061                  
d  0.1392  0.0273  -0.1432  2.9636          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.23 (> 0.2); trust the LHS-only matrix.

### K — medium tonic

- size 216 (LHS-only 70), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 36.2 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.6 mV, sd v 10.69 mV
- mean coefficient vector (all): a=0.07282, b=0.1954, c=-60.85, d=6.514
- mean coefficient vector (LHS-only, authoritative): a=0.05599, b=0.2145, c=-60.05, d=6.678
- persistence 1.00, DBCV contribution +0.0138  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 109 parameter-space components, 2 of them holding >= 5% of the cluster, largest 8%
  component means:
    - n=17: a=0.1051, b=0.1292, c=-71.3, d=9.467
    - n=11: a=0.1603, b=0.1263, c=-46.27, d=6.706

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0533                             0.4045
b -0.6867  0.0791                           
c  0.1768  0.2284  11.5250                  
d  0.3014 -0.0258   0.1538  3.3558          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0449                             0.3773
b -0.6453  0.0759                           
c  0.1509  0.2759  11.0848                  
d  0.4656 -0.1116   0.2317  3.1791          
```

### L — medium tonic

- size 215 (LHS-only 64), mean membership prob 0.614, noise adjacency 0.000
- medians: rate 41.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.8 mV, sd v 11.04 mV
- mean coefficient vector (all): a=0.06882, b=0.2115, c=-61.44, d=6.665
- mean coefficient vector (LHS-only, authoritative): a=0.05298, b=0.235, c=-59.04, d=6.855
- persistence 1.00, DBCV contribution +0.0009  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 131 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0515                             0.4039
b -0.6828  0.0805                           
c  0.2305  0.1724  11.7037                  
d  0.2336  0.0836   0.2864  3.4990          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0357                             0.3353
b -0.6336  0.0740                           
c  0.3425  0.2431  10.9203                  
d  0.2217  0.0650   0.3234  2.9028          
```

### M — medium tonic

- size 210 (LHS-only 86), mean membership prob 0.796, noise adjacency 0.000
- medians: rate 71.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -59.8 mV, sd v 12.85 mV
- mean coefficient vector (all): a=0.08622, b=0.2351, c=-65.13, d=5.73
- mean coefficient vector (LHS-only, authoritative): a=0.08629, b=0.2391, c=-64.07, d=6.382
- persistence 1.00, DBCV contribution +0.0114  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 109 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0598                             0.4209
b -0.5249  0.0730                           
c  0.3585  0.1842  10.1889                  
d  0.3940  0.2534   0.3280  3.6590          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0558                            0.3455
b -0.4563  0.0674                          
c  0.3939  0.1891  9.8424                  
d  0.3328  0.3506  0.3296  3.5514          
```

### N — slow tonic [mixed: 24% bursting]

- size 200 (LHS-only 103), mean membership prob 0.982, noise adjacency 0.000
- medians: rate 9.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.6 mV, sd v 6.88 mV
- mean coefficient vector (all): a=0.01651, b=0.164, c=-61.45, d=6.455
- mean coefficient vector (LHS-only, authoritative): a=0.01209, b=0.1899, c=-56.14, d=6.814
- persistence 0.80, DBCV contribution +0.0015  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 110 parameter-space components, 2 of them holding >= 5% of the cluster, largest 6%
  component means:
    - n=13: a=0.007626, b=0.1336, c=-58.12, d=1.869
    - n=10: a=0.006214, b=0.231, c=-60.81, d=9.188

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0162                             0.2883
b -0.4745  0.0647                           
c -0.1324  0.3930  11.5294                  
d -0.0079  0.2631  -0.0673  3.3526          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0081                             0.2181
b -0.5854  0.0644                           
c  0.1067  0.3519  11.4872                  
d -0.0127  0.0766  -0.2316  3.0232          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.24 (> 0.2); trust the LHS-only matrix.

### O — medium tonic

- size 198 (LHS-only 56), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 32.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -63.9 mV, sd v 10.14 mV
- mean coefficient vector (all): a=0.0602, b=0.1962, c=-62.6, d=6.791
- mean coefficient vector (LHS-only, authoritative): a=0.04637, b=0.2092, c=-58.96, d=6.517
- persistence 1.00, DBCV contribution +0.0111

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0523                             0.4292
b -0.7233  0.0770                           
c  0.1420  0.0712  11.2972                  
d  0.3255 -0.0703   0.1736  3.4388          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0360                             0.3395
b -0.6953  0.0691                           
c  0.4008  0.0921  11.0119                  
d  0.3987 -0.2781  -0.0093  2.9288          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.26 (> 0.2); trust the LHS-only matrix.

### P — fast tonic

- size 191 (LHS-only 126), mean membership prob 0.626, noise adjacency 0.000
- medians: rate 191.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -51.9 mV, sd v 16.24 mV
- mean coefficient vector (all): a=0.09743, b=0.2641, c=-64, d=2.574
- mean coefficient vector (LHS-only, authoritative): a=0.1041, b=0.2737, c=-65.69, d=2.818
- persistence 0.80, DBCV contribution +0.0111  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 69 parameter-space components, 2 of them holding >= 5% of the cluster, largest 16%
  component means:
    - n=31: a=0.1443, b=0.2974, c=-67.62, d=4.559
    - n=23: a=0.129, b=0.2157, c=-70.49, d=0.8081

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0634                            0.4550
b -0.0359  0.0614                          
c  0.0634 -0.1960  9.8296                  
d  0.6553  0.5071  0.2106  2.4916          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0584                            0.3797
b  0.0233  0.0547                          
c -0.0801 -0.2084  8.9429                  
d  0.6880  0.5647  0.0470  2.4141          
```

### Q — slow tonic

- size 186 (LHS-only 66), mean membership prob 0.811, noise adjacency 0.000
- medians: rate 18.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -65.5 mV, sd v 8.31 mV
- mean coefficient vector (all): a=0.03459, b=0.1866, c=-61.79, d=6.401
- mean coefficient vector (LHS-only, authoritative): a=0.02382, b=0.2137, c=-56.35, d=6.076
- persistence 1.00, DBCV contribution +0.0091

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0336                             0.3983
b -0.6190  0.0813                           
c  0.0916  0.1953  10.3655                  
d  0.0588  0.2035   0.0316  3.4158          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0221                             0.3243
b -0.6459  0.0742                           
c  0.2105  0.1957  10.1454                  
d  0.0818 -0.0488   0.0476  2.8275          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.25 (> 0.2); trust the LHS-only matrix.

### R — slow tonic

- size 182 (LHS-only 61), mean membership prob 0.726, noise adjacency 0.000
- medians: rate 17.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -66.1 mV, sd v 8.21 mV
- mean coefficient vector (all): a=0.03321, b=0.1888, c=-62.06, d=6.843
- mean coefficient vector (LHS-only, authoritative): a=0.02014, b=0.2245, c=-57.49, d=7.153
- persistence 1.00, DBCV contribution +0.0101  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 119 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0347                             0.4029
b -0.6110  0.0834                           
c  0.0556  0.0955  10.5061                  
d  0.1725  0.2112   0.0916  3.4940          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0135                             0.2787
b -0.7297  0.0726                           
c  0.3922 -0.0118  11.6299                  
d  0.0014  0.1385  -0.0508  2.8161          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.34 (> 0.2); trust the LHS-only matrix.

### S — medium tonic

- size 178 (LHS-only 64), mean membership prob 0.757, noise adjacency 0.000
- medians: rate 20.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -65.0 mV, sd v 8.49 mV
- mean coefficient vector (all): a=0.04173, b=0.1841, c=-64, d=6.286
- mean coefficient vector (LHS-only, authoritative): a=0.0308, b=0.2102, c=-61.11, d=6.12
- persistence 1.00, DBCV contribution +0.0024  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 110 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0432                             0.4387
b -0.6539  0.0796                           
c  0.2261  0.1001  10.1334                  
d  0.2587  0.0634   0.1927  3.3274          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0386                             0.3937
b -0.6314  0.0742                           
c  0.3276 -0.0082  10.7824                  
d  0.2164  0.0541   0.0185  3.0278          
```

### T — slow tonic

- size 174 (LHS-only 72), mean membership prob 0.794, noise adjacency 0.000
- medians: rate 19.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -65.6 mV, sd v 8.54 mV
- mean coefficient vector (all): a=0.03618, b=0.1922, c=-63.06, d=6.565
- mean coefficient vector (LHS-only, authoritative): a=0.0282, b=0.2132, c=-59.12, d=6.98
- persistence 1.00, DBCV contribution +0.0068  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 122 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0364                             0.4180
b -0.6737  0.0798                           
c  0.0785  0.0364  11.1995                  
d  0.1059  0.1203   0.0621  3.2787          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0278                             0.3423
b -0.6284  0.0732                           
c  0.1343  0.0729  12.3173                  
d  0.0102  0.0714  -0.0904  3.0053          
```

### U — medium tonic

- size 169 (LHS-only 76), mean membership prob 0.747, noise adjacency 0.000
- medians: rate 21.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.9 mV, sd v 8.86 mV
- mean coefficient vector (all): a=0.04249, b=0.1914, c=-62.77, d=6.201
- mean coefficient vector (LHS-only, authoritative): a=0.02693, b=0.2343, c=-59.08, d=6.544
- persistence 1.00, DBCV contribution +0.0082  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 121 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0422                             0.4284
b -0.6359  0.0812                           
c  0.0665  0.1287  10.8377                  
d  0.2229  0.0291  -0.0025  3.4563          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0296                             0.3315
b -0.4525  0.0735                           
c  0.3206  0.0948  11.2417                  
d  0.1869 -0.0626  -0.2547  3.4093          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.25 (> 0.2); trust the LHS-only matrix.

### V — slow tonic

- size 165 (LHS-only 53), mean membership prob 0.785, noise adjacency 0.000
- medians: rate 15.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -66.4 mV, sd v 7.76 mV
- mean coefficient vector (all): a=0.0277, b=0.1748, c=-64.78, d=6.488
- mean coefficient vector (LHS-only, authoritative): a=0.02021, b=0.1898, c=-60.33, d=6.577
- persistence 1.00, DBCV contribution +0.0036  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 108 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0271                             0.3707
b -0.6000  0.0724                           
c  0.0558  0.0882  10.1583                  
d  0.0545  0.2381   0.0903  3.3218          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0151                             0.2933
b -0.6647  0.0640                           
c  0.1573  0.2262  11.7094                  
d  0.3150 -0.0565   0.0480  2.8276          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.29 (> 0.2); trust the LHS-only matrix.

### W — medium tonic

- size 156 (LHS-only 51), mean membership prob 0.705, noise adjacency 0.000
- medians: rate 44.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.4 mV, sd v 11.41 mV
- mean coefficient vector (all): a=0.07226, b=0.2108, c=-63.23, d=6.712
- mean coefficient vector (LHS-only, authoritative): a=0.06469, b=0.2231, c=-61.03, d=7.887
- persistence 1.00, DBCV contribution +0.0016  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 100 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0542                             0.3969
b -0.6608  0.0762                           
c  0.2555  0.1589  11.3458                  
d  0.1605  0.2662   0.2048  3.4968          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0413                             0.2675
b -0.7542  0.0635                           
c  0.3396  0.0566  10.6518                  
d -0.0724  0.1435  -0.0828  2.6573          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.29 (> 0.2); trust the LHS-only matrix.

### Y — slow tonic [mixed: 20% bursting]

- size 152 (LHS-only 55), mean membership prob 0.757, noise adjacency 0.000
- medians: rate 12.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.6 mV, sd v 7.64 mV
- mean coefficient vector (all): a=0.02022, b=0.1962, c=-62.06, d=7.387
- mean coefficient vector (LHS-only, authoritative): a=0.01273, b=0.233, c=-58.5, d=7.115
- persistence 1.00, DBCV contribution +0.0068

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0196                             0.3480
b -0.6560  0.0832                           
c  0.1008  0.2666  11.6487                  
d  0.0152  0.2196  -0.0396  3.3485          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0085                             0.2300
b -0.7127  0.0770                           
c  0.0084  0.3702  12.3879                  
d -0.2361  0.2360  -0.1109  3.1155          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.25 (> 0.2); trust the LHS-only matrix.

### AA — fast tonic

- size 149 (LHS-only 98), mean membership prob 0.583, noise adjacency 0.000
- medians: rate 220.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -48.3 mV, sd v 15.94 mV
- mean coefficient vector (all): a=0.08991, b=0.2608, c=-60.52, d=1.883
- mean coefficient vector (LHS-only, authoritative): a=0.09335, b=0.2669, c=-61.56, d=1.972
- persistence 0.00, DBCV contribution +0.0006  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 74 parameter-space components, 3 of them holding >= 5% of the cluster, largest 6%
  component means:
    - n=9: a=0.03864, b=0.2575, c=-51.98, d=1.524
    - n=8: a=0.0543, b=0.2785, c=-72.31, d=0.792
    - n=8: a=0.1092, b=0.2378, c=-74.12, d=0.3489

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0603                             0.4102
b -0.0914  0.0602                           
c -0.0177 -0.3226  10.8527                  
d  0.6165  0.3207   0.2936  1.7953          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0550                             0.3412
b  0.0308  0.0503                           
c -0.0474 -0.2720  11.0775                  
d  0.6373  0.4140   0.3110  1.7803          
```

### AB — medium tonic

- size 149 (LHS-only 45), mean membership prob 0.901, noise adjacency 0.000
- medians: rate 22.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -65.0 mV, sd v 9.09 mV
- mean coefficient vector (all): a=0.0447, b=0.1872, c=-64.26, d=6.751
- mean coefficient vector (LHS-only, authoritative): a=0.03016, b=0.223, c=-59.64, d=6.938
- persistence 0.60, DBCV contribution +0.0004  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 104 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0439                             0.4284
b -0.6622  0.0776                           
c  0.2494  0.0165  10.6122                  
d  0.2233  0.0377   0.0186  3.6709          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0279                             0.3118
b -0.7001  0.0644                           
c  0.3680 -0.0413  11.9096                  
d  0.0534 -0.0107  -0.1374  3.2081          
```

### AC — medium tonic

- size 145 (LHS-only 52), mean membership prob 0.745, noise adjacency 0.000
- medians: rate 53.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -61.1 mV, sd v 11.95 mV
- mean coefficient vector (all): a=0.07405, b=0.2276, c=-63.26, d=6.305
- mean coefficient vector (LHS-only, authoritative): a=0.0722, b=0.2217, c=-63.94, d=6.433
- persistence 1.00, DBCV contribution +0.0096  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 102 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0564                             0.4391
b -0.5218  0.0745                           
c  0.3795  0.1884  12.0720                  
d  0.3775  0.2041   0.4011  3.6037          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0535                             0.3923
b -0.5572  0.0652                           
c  0.3734  0.1785  10.9619                  
d  0.4294  0.0862   0.2980  3.2952          
```

### AD — medium tonic

- size 144 (LHS-only 51), mean membership prob 0.613, noise adjacency 0.000
- medians: rate 55.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -60.9 mV, sd v 12.03 mV
- mean coefficient vector (all): a=0.08119, b=0.2174, c=-62.42, d=6.667
- mean coefficient vector (LHS-only, authoritative): a=0.07365, b=0.2339, c=-60.97, d=7.477
- persistence 1.00, DBCV contribution +0.0002  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 102 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0528                             0.3726
b -0.5514  0.0719                           
c  0.4599  0.0756  11.9433                  
d  0.3418  0.2333   0.3490  3.5297          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0435                             0.3085
b -0.6345  0.0678                           
c  0.4890  0.0108  10.3607                  
d  0.2636  0.1498   0.2432  2.8705          
```

### AE — medium tonic

- size 142 (LHS-only 43), mean membership prob 0.838, noise adjacency 0.000
- medians: rate 28.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -64.0 mV, sd v 9.67 mV
- mean coefficient vector (all): a=0.05179, b=0.195, c=-63.45, d=6.109
- mean coefficient vector (LHS-only, authoritative): a=0.0306, b=0.2334, c=-60.35, d=6.123
- persistence 1.00, DBCV contribution +0.0024

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0463                             0.4318
b -0.7345  0.0818                           
c  0.1682  0.0292  11.0037                  
d  0.2148  0.0201   0.1401  3.2264          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0221                             0.3256
b -0.7755  0.0697                           
c  0.3313 -0.0592  11.5188                  
d  0.1778 -0.0495  -0.1506  2.8061          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.29 (> 0.2); trust the LHS-only matrix.

### AF — fast tonic

- size 142 (LHS-only 78), mean membership prob 0.779, noise adjacency 0.000
- medians: rate 84.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -59.5 mV, sd v 13.64 mV
- mean coefficient vector (all): a=0.1011, b=0.2352, c=-66.31, d=6.333
- mean coefficient vector (LHS-only, authoritative): a=0.1005, b=0.2483, c=-64.51, d=6.862
- persistence 1.00, DBCV contribution +0.0093  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 83 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0593                            0.3854
b -0.4746  0.0659                          
c  0.2529  0.1685  9.4224                  
d  0.4721  0.2496  0.1700  3.7139          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0524                            0.3200
b -0.5272  0.0608                          
c  0.3355  0.1892  9.3412                  
d  0.4641  0.1276  0.2218  3.1382          
```

### AG — medium tonic

- size 142 (LHS-only 51), mean membership prob 0.837, noise adjacency 0.000
- medians: rate 64.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -60.4 mV, sd v 12.56 mV
- mean coefficient vector (all): a=0.08307, b=0.2285, c=-64.16, d=6.337
- mean coefficient vector (LHS-only, authoritative): a=0.07497, b=0.2371, c=-61.58, d=6.98
- persistence 1.00, DBCV contribution +0.0065

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0552                             0.3888
b -0.4801  0.0714                           
c  0.1713  0.4039  10.6234                  
d  0.3943  0.2835   0.2949  3.6478          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0358                             0.2390
b -0.3105  0.0633                           
c  0.4570  0.3211  10.3736                  
d  0.3446  0.4263   0.3953  3.1156          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.29 (> 0.2); trust the LHS-only matrix.

### AH — slow tonic [mixed: 24% bursting]

- size 140 (LHS-only 64), mean membership prob 0.892, noise adjacency 0.000
- medians: rate 7.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -67.9 mV, sd v 6.68 mV
- mean coefficient vector (all): a=0.01455, b=0.1475, c=-61.54, d=6.909
- mean coefficient vector (LHS-only, authoritative): a=0.01197, b=0.1681, c=-54.17, d=6.712
- persistence 1.00, DBCV contribution +0.0085  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 79 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0087                             0.2576
b -0.6470  0.0513                           
c -0.1078  0.3582  12.3225                  
d  0.0296  0.2083  -0.0634  3.4951          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0068                             0.2186
b -0.6149  0.0552                           
c  0.1577  0.3136  11.2672                  
d -0.1518  0.1619  -0.1578  3.1516          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.27 (> 0.2); trust the LHS-only matrix.

### AI — slow tonic

- size 139 (LHS-only 63), mean membership prob 0.821, noise adjacency 0.000
- medians: rate 14.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -66.7 mV, sd v 7.85 mV
- mean coefficient vector (all): a=0.02251, b=0.1982, c=-61.96, d=6.63
- mean coefficient vector (LHS-only, authoritative): a=0.01653, b=0.2245, c=-58.93, d=6.931
- persistence 1.00, DBCV contribution +0.0063  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 104 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0230                             0.3629
b -0.6428  0.0810                           
c -0.0378  0.2172  10.3986                  
d  0.1505  0.1951   0.0723  3.4931          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0143                             0.2735
b -0.6495  0.0744                           
c  0.0039  0.1874  10.9279                  
d  0.1650  0.0782  -0.1455  3.2705          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.22 (> 0.2); trust the LHS-only matrix.

### AJ — medium tonic

- size 139 (LHS-only 44), mean membership prob 0.717, noise adjacency 0.000
- medians: rate 61.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -60.9 mV, sd v 12.40 mV
- mean coefficient vector (all): a=0.07946, b=0.2313, c=-62.59, d=6.691
- mean coefficient vector (LHS-only, authoritative): a=0.06732, b=0.235, c=-64.78, d=6.524
- persistence 1.00, DBCV contribution +0.0069

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0533                             0.3887
b -0.4595  0.0692                           
c  0.4605  0.2310  11.1130                  
d  0.4229  0.2850   0.4319  3.7256          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0463                            0.3716
b -0.4901  0.0631                          
c  0.4862  0.2123  9.5136                  
d  0.3334  0.3453  0.4449  3.6260          
```

### AK — fast tonic

- size 137 (LHS-only 65), mean membership prob 0.812, noise adjacency 0.000
- medians: rate 94.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -58.8 mV, sd v 14.03 mV
- mean coefficient vector (all): a=0.09645, b=0.243, c=-67.82, d=5.713
- mean coefficient vector (LHS-only, authoritative): a=0.09451, b=0.2541, c=-69.03, d=6.836
- persistence 1.00, DBCV contribution +0.0085  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 74 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0603                            0.4453
b -0.4176  0.0645                          
c  0.3001  0.2031  9.0266                  
d  0.5016  0.3629  0.2559  3.8770          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0459                            0.2849
b -0.5946  0.0581                          
c  0.2282  0.0988  6.8970                  
d  0.3673  0.3960  0.2232  3.3348          
```

### AL — slow tonic

- size 136 (LHS-only 46), mean membership prob 0.718, noise adjacency 0.000
- medians: rate 16.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -65.9 mV, sd v 8.25 mV
- mean coefficient vector (all): a=0.02565, b=0.1922, c=-62.45, d=6.078
- mean coefficient vector (LHS-only, authoritative): a=0.01931, b=0.2051, c=-58.51, d=5.423
- persistence 1.00, DBCV contribution +0.0062  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 100 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0263                             0.3557
b -0.6163  0.0756                           
c  0.1090  0.1108  10.6096                  
d  0.0497  0.2018  -0.0717  3.2909          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0155                             0.2865
b -0.6530  0.0656                           
c  0.2594  0.1841  10.8374                  
d  0.0731  0.2348  -0.0590  2.9291          
```

### AM — medium tonic

- size 136 (LHS-only 47), mean membership prob 0.760, noise adjacency 0.000
- medians: rate 22.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -65.6 mV, sd v 9.08 mV
- mean coefficient vector (all): a=0.03933, b=0.1932, c=-62.44, d=6.749
- mean coefficient vector (LHS-only, authoritative): a=0.03057, b=0.2265, c=-58.53, d=7.132
- persistence 0.20, DBCV contribution +0.0066  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 99 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0374                             0.3890
b -0.6497  0.0817                           
c  0.1883  0.0421  10.9019                  
d  0.0129  0.3153   0.1566  3.5946          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0283                             0.3461
b -0.7212  0.0798                           
c  0.3815 -0.1103  11.5920                  
d  0.0107  0.2110  -0.2005  2.9497          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.36 (> 0.2); trust the LHS-only matrix.

### AN — fast tonic

- size 133 (LHS-only 87), mean membership prob 0.882, noise adjacency 0.000
- medians: rate 293.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -48.1 mV, sd v 15.84 mV
- mean coefficient vector (all): a=0.09149, b=0.2649, c=-60.05, d=2.409
- mean coefficient vector (LHS-only, authoritative): a=0.0924, b=0.2658, c=-60.7, d=2.604
- persistence 0.40, DBCV contribution +0.0004  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 92 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0594                             0.4085
b -0.0300  0.0691                           
c -0.1363 -0.2498  11.3789                  
d  0.5195  0.2413  -0.0387  2.3867          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0576                             0.3637
b  0.1142  0.0621                           
c -0.2251 -0.3488  11.6888                  
d  0.6022  0.2129  -0.2521  2.3714          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.21 (> 0.2); trust the LHS-only matrix.

### AO — medium tonic

- size 129 (LHS-only 46), mean membership prob 0.840, noise adjacency 0.000
- medians: rate 31.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.1 mV, sd v 10.12 mV
- mean coefficient vector (all): a=0.06031, b=0.1984, c=-62.52, d=6.898
- mean coefficient vector (LHS-only, authoritative): a=0.0396, b=0.217, c=-61.67, d=6.496
- persistence 1.00, DBCV contribution +0.0073  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 91 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0498                             0.4110
b -0.7346  0.0822                           
c  0.2114  0.1274  11.6511                  
d  0.2032  0.0418   0.2437  3.2853          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0343                             0.3425
b -0.6740  0.0685                           
c  0.2737  0.1608  11.7723                  
d  0.3097 -0.0257   0.1783  3.2298          
```

### AP — medium tonic

- size 129 (LHS-only 44), mean membership prob 0.934, noise adjacency 0.000
- medians: rate 29.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.0 mV, sd v 10.11 mV
- mean coefficient vector (all): a=0.05376, b=0.2, c=-63.15, d=6.82
- mean coefficient vector (LHS-only, authoritative): a=0.03791, b=0.2105, c=-61.08, d=5.936
- persistence 1.00, DBCV contribution +0.0017  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 91 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0482                             0.4318
b -0.7476  0.0803                           
c  0.2805 -0.1216  11.1659                  
d  0.1934  0.0442   0.0355  3.6813          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0295                             0.3331
b -0.6338  0.0730                           
c  0.3764 -0.1126  10.8807                  
d  0.1789  0.1014  -0.0487  3.3802          
```

### AQ — slow tonic

- size 128 (LHS-only 38), mean membership prob 0.899, noise adjacency 0.000
- medians: rate 18.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -65.7 mV, sd v 8.42 mV
- mean coefficient vector (all): a=0.0311, b=0.1875, c=-65.12, d=6.219
- mean coefficient vector (LHS-only, authoritative): a=0.01776, b=0.222, c=-59.21, d=6.434
- persistence 0.00, DBCV contribution +0.0064  **UNSTABLE**

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0344                             0.3856
b -0.5789  0.0760                           
c -0.0175  0.1977  10.3214                  
d  0.1050  0.2494   0.1201  3.5568          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0097                             0.2145
b -0.5944  0.0672                           
c  0.2139  0.1958  10.9648                  
d  0.0379  0.2862  -0.0327  3.3843          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.23 (> 0.2); trust the LHS-only matrix.

### AR — medium tonic

- size 127 (LHS-only 56), mean membership prob 0.771, noise adjacency 0.000
- medians: rate 20.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -65.2 mV, sd v 8.70 mV
- mean coefficient vector (all): a=0.04617, b=0.1751, c=-61.54, d=6.368
- mean coefficient vector (LHS-only, authoritative): a=0.02576, b=0.2152, c=-58.43, d=6.45
- persistence 1.00, DBCV contribution +0.0029  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 99 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0448                             0.4215
b -0.6160  0.0798                           
c  0.1079  0.2296  10.5502                  
d  0.2271  0.1299   0.1091  3.3485          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0228                             0.3209
b -0.6491  0.0791                           
c  0.2488  0.1518  10.7137                  
d -0.0212  0.2855  -0.0283  3.1261          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.25 (> 0.2); trust the LHS-only matrix.

### AS — medium tonic

- size 126 (LHS-only 43), mean membership prob 0.713, noise adjacency 0.000
- medians: rate 25.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.9 mV, sd v 9.40 mV
- mean coefficient vector (all): a=0.04418, b=0.2, c=-62.27, d=6.711
- mean coefficient vector (LHS-only, authoritative): a=0.0324, b=0.2269, c=-58.32, d=7.328
- persistence 1.00, DBCV contribution +0.0063  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 100 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0379                             0.4123
b -0.7244  0.0864                           
c  0.1188  0.0465  11.2676                  
d  0.2176 -0.0245   0.2156  3.5081          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0281                             0.3117
b -0.5966  0.0748                           
c  0.1685  0.1677  12.1265                  
d  0.1033  0.1199   0.2130  3.1854          
```

### AT — medium tonic

- size 126 (LHS-only 40), mean membership prob 0.928, noise adjacency 0.000
- medians: rate 62.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -60.3 mV, sd v 12.37 mV
- mean coefficient vector (all): a=0.08286, b=0.2297, c=-63.79, d=6.168
- mean coefficient vector (LHS-only, authoritative): a=0.07381, b=0.2551, c=-62.82, d=7.022
- persistence 1.00, DBCV contribution +0.0055  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 85 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0592                             0.4201
b -0.5123  0.0735                           
c  0.4001  0.1789  10.8937                  
d  0.3622  0.2992   0.3654  3.9371          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0477                             0.3344
b -0.4085  0.0669                           
c  0.3858  0.2823  11.1823                  
d  0.5659  0.1367   0.2781  3.2902          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.20 (> 0.2); trust the LHS-only matrix.

### AU — slow tonic

- size 122 (LHS-only 47), mean membership prob 0.865, noise adjacency 0.000
- medians: rate 19.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -65.5 mV, sd v 8.81 mV
- mean coefficient vector (all): a=0.04037, b=0.1799, c=-61.73, d=7.219
- mean coefficient vector (LHS-only, authoritative): a=0.02822, b=0.2043, c=-58.26, d=7.665
- persistence 1.00, DBCV contribution +0.0076  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 89 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0377                             0.3948
b -0.6455  0.0759                           
c  0.0732  0.0976  10.4644                  
d  0.1506  0.1494   0.0802  3.7176          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0219                             0.2971
b -0.6532  0.0705                           
c  0.1487  0.0478  10.8216                  
d  0.0248  0.2037  -0.0081  3.4148          
```

### AV — fast tonic

- size 121 (LHS-only 87), mean membership prob 0.818, noise adjacency 0.000
- medians: rate 267.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -46.0 mV, sd v 16.63 mV
- mean coefficient vector (all): a=0.09958, b=0.2806, c=-60.98, d=1.826
- mean coefficient vector (LHS-only, authoritative): a=0.101, b=0.2793, c=-61.9, d=1.661
- persistence 0.40, DBCV contribution +0.0035  **UNSTABLE** **MULTIMODAL** (3 components)
- pooled mean is **not interpretable**: 48 parameter-space components, 3 of them holding >= 5% of the cluster, largest 18%
  component means:
    - n=22: a=0.1203, b=0.321, c=-72.56, d=1.195
    - n=22: a=0.1224, b=0.2835, c=-54.62, d=3.097
    - n=8: a=0.04724, b=0.3055, c=-74.93, d=0.2493

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0575                            0.3925
b  0.0217  0.0640                          
c -0.0193 -0.5039  9.9029                  
d  0.5892  0.1526  0.4584  1.5941          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0530                            0.3370
b  0.0846  0.0629                          
c -0.0674 -0.6037  9.4275                  
d  0.5864  0.0988  0.4347  1.3481          
```

### AW — medium tonic

- size 121 (LHS-only 39), mean membership prob 0.903, noise adjacency 0.000
- medians: rate 47.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -61.9 mV, sd v 11.58 mV
- mean coefficient vector (all): a=0.07506, b=0.2153, c=-62.05, d=6.679
- mean coefficient vector (LHS-only, authoritative): a=0.05795, b=0.2479, c=-60.46, d=7.347
- persistence 1.00, DBCV contribution +0.0039  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 89 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0528                             0.3767
b -0.6382  0.0764                           
c  0.2859  0.2221  11.6383                  
d  0.3770  0.0406   0.3355  3.4072          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0402                             0.2717
b -0.4754  0.0609                           
c  0.4799  0.1531  11.3600                  
d  0.1810  0.1755   0.1739  2.8026          
```

### AY — slow tonic [mixed: 20% bursting]

- size 120 (LHS-only 57), mean membership prob 0.878, noise adjacency 0.000
- medians: rate 9.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.4 mV, sd v 7.14 mV
- mean coefficient vector (all): a=0.01895, b=0.1671, c=-62.34, d=6.966
- mean coefficient vector (LHS-only, authoritative): a=0.01183, b=0.2057, c=-58.76, d=7.525
- persistence 1.00, DBCV contribution +0.0077  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 92 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0166                             0.3099
b -0.5736  0.0702                           
c -0.1125  0.1896  11.2116                  
d -0.0994  0.3305   0.0085  3.6018          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0093                             0.2294
b -0.5532  0.0661                           
c  0.1458  0.1142  11.7266                  
d -0.0217  0.2608  -0.1606  3.0730          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.26 (> 0.2); trust the LHS-only matrix.

### BA — medium tonic

- size 119 (LHS-only 56), mean membership prob 0.732, noise adjacency 0.000
- medians: rate 77.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -59.4 mV, sd v 13.35 mV
- mean coefficient vector (all): a=0.09227, b=0.2331, c=-64.98, d=6.147
- mean coefficient vector (LHS-only, authoritative): a=0.08824, b=0.2479, c=-64.29, d=6.497
- persistence 0.00, DBCV contribution +0.0067  **UNSTABLE**

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0573                             0.4033
b -0.2786  0.0710                           
c  0.3734  0.2415  10.3004                  
d  0.4514  0.4998   0.3578  3.8970          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0536                            0.3615
b -0.3522  0.0642                          
c  0.3408  0.2499  9.9380                  
d  0.4680  0.3993  0.4360  3.3745          
```

### BB — medium tonic

- size 116 (LHS-only 44), mean membership prob 0.994, noise adjacency 0.000
- medians: rate 32.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -63.7 mV, sd v 10.12 mV
- mean coefficient vector (all): a=0.05287, b=0.21, c=-62.85, d=6.378
- mean coefficient vector (LHS-only, authoritative): a=0.03884, b=0.2242, c=-63.72, d=6.479
- persistence 1.00, DBCV contribution +0.0075

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0479                             0.4196
b -0.7238  0.0752                           
c  0.2829  0.0186  11.8766                  
d  0.1766  0.1093   0.2312  3.5846          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0329                             0.3642
b -0.6973  0.0707                           
c  0.3718  0.0508  11.7898                  
d  0.3357  0.0489   0.2703  3.3016          
```

### BC — slow tonic [mixed: 20% bursting]

- size 115 (LHS-only 49), mean membership prob 0.953, noise adjacency 0.000
- medians: rate 12.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.2 mV, sd v 7.61 mV
- mean coefficient vector (all): a=0.0238, b=0.1756, c=-62.39, d=7.1
- mean coefficient vector (LHS-only, authoritative): a=0.02016, b=0.198, c=-58.21, d=7.559
- persistence 1.00, DBCV contribution +0.0029

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0237                             0.3492
b -0.4027  0.0752                           
c  0.1546  0.1724  12.0511                  
d  0.2029  0.2127   0.0564  3.2996          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0218                             0.2971
b -0.1444  0.0740                           
c  0.1804  0.2438  13.1800                  
d  0.2368  0.0711  -0.1472  3.0118          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.26 (> 0.2); trust the LHS-only matrix.

### BD — medium tonic

- size 115 (LHS-only 36), mean membership prob 0.995, noise adjacency 0.000
- medians: rate 68.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -60.1 mV, sd v 12.74 mV
- mean coefficient vector (all): a=0.09196, b=0.2216, c=-63.42, d=6.241
- mean coefficient vector (LHS-only, authoritative): a=0.07861, b=0.2391, c=-64.3, d=5.941
- persistence 1.00, DBCV contribution +0.0073

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0602                             0.4076
b -0.4587  0.0660                           
c  0.4424  0.1864  10.4840                  
d  0.6070  0.0747   0.4429  3.6862          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0548                            0.3431
b -0.5477  0.0653                          
c  0.2247  0.3548  9.8609                  
d  0.4992  0.1548  0.4369  3.0274          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.22 (> 0.2); trust the LHS-only matrix.

### BE — medium tonic

- size 115 (LHS-only 33), mean membership prob 0.929, noise adjacency 0.000
- medians: rate 39.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.4 mV, sd v 10.79 mV
- mean coefficient vector (all): a=0.07192, b=0.2015, c=-60.54, d=6.35
- mean coefficient vector (LHS-only, authoritative): a=0.06704, b=0.2154, c=-56.61, d=6.322
- persistence 0.00, DBCV contribution +0.0010  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 93 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0543                             0.4188
b -0.6064  0.0794                           
c  0.1872  0.3058  11.2061                  
d  0.3317  0.1256   0.2415  3.5434          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0515                             0.3872
b -0.6092  0.0684                           
c  0.5649 -0.0563  11.8079                  
d  0.4980 -0.1351   0.4949  2.4888          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.38 (> 0.2); trust the LHS-only matrix.

### BF — fast tonic

- size 114 (LHS-only 65), mean membership prob 0.779, noise adjacency 0.000
- medians: rate 108.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -58.0 mV, sd v 14.55 mV
- mean coefficient vector (all): a=0.09346, b=0.2406, c=-67.48, d=4.944
- mean coefficient vector (LHS-only, authoritative): a=0.1047, b=0.2473, c=-67.51, d=6.041
- persistence 1.00, DBCV contribution +0.0059

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0587                            0.4405
b -0.1726  0.0643                          
c  0.1595  0.0611  8.4268                  
d  0.6526  0.4413  0.0321  3.8187          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0512                            0.3148
b -0.2451  0.0525                          
c  0.0275  0.1969  8.0254                  
d  0.6803  0.3849 -0.0227  3.5075          
```

### BG — slow tonic

- size 113 (LHS-only 37), mean membership prob 0.919, noise adjacency 0.000
- medians: rate 14.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -66.6 mV, sd v 7.77 mV
- mean coefficient vector (all): a=0.02416, b=0.1939, c=-62.21, d=6.677
- mean coefficient vector (LHS-only, authoritative): a=0.01621, b=0.2228, c=-56.29, d=6.715
- persistence 1.00, DBCV contribution +0.0058  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 80 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0256                             0.3766
b -0.6250  0.0821                           
c  0.1326  0.0150  11.3054                  
d  0.1017  0.2753   0.0420  3.4218          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0108                             0.2601
b -0.7256  0.0752                           
c  0.3915 -0.1402  11.0630                  
d -0.3699  0.3858  -0.2719  3.3963          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.47 (> 0.2); trust the LHS-only matrix.

### BH — medium tonic

- size 111 (LHS-only 42), mean membership prob 0.960, noise adjacency 0.000
- medians: rate 74.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -59.9 mV, sd v 13.04 mV
- mean coefficient vector (all): a=0.08826, b=0.2408, c=-64.53, d=6.154
- mean coefficient vector (LHS-only, authoritative): a=0.09158, b=0.2502, c=-63.32, d=7.092
- persistence 1.00, DBCV contribution +0.0062

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0604                             0.3986
b -0.2695  0.0727                           
c  0.3924  0.3815  11.1915                  
d  0.6136  0.3430   0.4844  3.9090          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0508                             0.3072
b -0.2533  0.0676                           
c  0.4611  0.5246  11.5272                  
d  0.4908  0.3974   0.5833  3.6902          
```

### BI — slow tonic

- size 109 (LHS-only 47), mean membership prob 0.967, noise adjacency 0.000
- medians: rate 15.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -66.4 mV, sd v 7.92 mV
- mean coefficient vector (all): a=0.03144, b=0.1734, c=-63.71, d=6.744
- mean coefficient vector (LHS-only, authoritative): a=0.02114, b=0.1983, c=-61.76, d=7.093
- persistence 0.80, DBCV contribution +0.0053  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 80 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0283                             0.3787
b -0.6340  0.0762                           
c  0.1383  0.0983  10.7596                  
d  0.2268  0.0828   0.0183  3.3095          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0151                             0.2689
b -0.6719  0.0714                           
c  0.0663  0.1146  12.2723                  
d -0.0531  0.0511  -0.2626  3.1064          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.28 (> 0.2); trust the LHS-only matrix.

### BJ — slow tonic

- size 109 (LHS-only 38), mean membership prob 0.763, noise adjacency 0.000
- medians: rate 13.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -66.5 mV, sd v 7.55 mV
- mean coefficient vector (all): a=0.02461, b=0.1793, c=-64.88, d=6.225
- mean coefficient vector (LHS-only, authoritative): a=0.01865, b=0.2008, c=-59.26, d=5.908
- persistence 1.00, DBCV contribution +0.0023  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 86 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0228                             0.3635
b -0.6532  0.0750                           
c -0.0096  0.0924  11.6162                  
d -0.0275  0.3041  -0.1535  3.3049          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0163                             0.2972
b -0.6601  0.0671                           
c  0.4147 -0.1859  13.1344                  
d -0.1349  0.2620  -0.3909  2.9801          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.42 (> 0.2); trust the LHS-only matrix.

### BK — slow tonic

- size 109 (LHS-only 46), mean membership prob 0.896, noise adjacency 0.000
- medians: rate 13.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.0 mV, sd v 7.67 mV
- mean coefficient vector (all): a=0.02321, b=0.1837, c=-62.52, d=6.59
- mean coefficient vector (LHS-only, authoritative): a=0.01942, b=0.2009, c=-57.85, d=6.607
- persistence 1.00, DBCV contribution +0.0021  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 81 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0266                             0.3463
b -0.5096  0.0754                           
c  0.0293  0.3099  10.5339                  
d  0.0163  0.2632   0.0283  3.3270          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0149                             0.3034
b -0.7441  0.0770                           
c  0.0643  0.2102  11.4425                  
d -0.1446  0.3306  -0.0612  3.0170          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.23 (> 0.2); trust the LHS-only matrix.

### BL — medium tonic

- size 108 (LHS-only 33), mean membership prob 0.844, noise adjacency 0.000
- medians: rate 29.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -64.0 mV, sd v 9.80 mV
- mean coefficient vector (all): a=0.0544, b=0.1912, c=-62.36, d=6.121
- mean coefficient vector (LHS-only, authoritative): a=0.04029, b=0.2278, c=-58.43, d=6.18
- persistence 1.00, DBCV contribution +0.0000  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 81 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0452                             0.4389
b -0.7267  0.0767                           
c  0.1724  0.1699  11.2182                  
d  0.2887  0.0176   0.1715  3.6000          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0362                             0.3445
b -0.7331  0.0613                           
c -0.0468  0.3212  11.1626                  
d  0.1825  0.0181  -0.1017  3.1707          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.27 (> 0.2); trust the LHS-only matrix.

### BM — medium tonic [mixed: 20% bursting]

- size 108 (LHS-only 37), mean membership prob 0.928, noise adjacency 0.000
- medians: rate 42.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.7 mV, sd v 11.15 mV
- mean coefficient vector (all): a=0.06647, b=0.2184, c=-60.87, d=6.941
- mean coefficient vector (LHS-only, authoritative): a=0.05469, b=0.2279, c=-61.05, d=7.179
- persistence 0.80, DBCV contribution +0.0070  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 92 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0500                             0.3923
b -0.6554  0.0758                           
c  0.4380  0.0850  12.1419                  
d  0.3437  0.0157   0.3962  3.3693          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0359                             0.3312
b -0.7138  0.0768                           
c  0.3148  0.2136  10.7861                  
d  0.2554  0.1519   0.4003  3.0279          
```

### BN — medium tonic

- size 105 (LHS-only 34), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 30.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.0 mV, sd v 10.14 mV
- mean coefficient vector (all): a=0.0568, b=0.1976, c=-62.84, d=6.542
- mean coefficient vector (LHS-only, authoritative): a=0.03721, b=0.2117, c=-63.52, d=6.159
- persistence 1.00, DBCV contribution +0.0018

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0503                             0.4237
b -0.7125  0.0803                           
c  0.2574  0.0009  11.5421                  
d  0.1604  0.0086   0.1826  3.5725          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0293                             0.3713
b -0.6846  0.0733                           
c  0.3558  0.0111  12.1089                  
d  0.2094 -0.0480   0.1052  3.1014          
```

### BO — medium tonic

- size 105 (LHS-only 23), mean membership prob 0.955, noise adjacency 0.000
- medians: rate 46.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -61.9 mV, sd v 11.42 mV
- mean coefficient vector (all): a=0.07545, b=0.2202, c=-62.56, d=6.372
- mean coefficient vector (LHS-only, authoritative): a=0.06501, b=0.2255, c=-58.67, d=6.81
- persistence 0.00, DBCV contribution +0.0066  **UNSTABLE**

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0596                             0.4476
b -0.7078  0.0805                           
c  0.2057  0.1448  11.7444                  
d  0.3476 -0.0518   0.2792  3.4056          

LHS-only: insufficient-sample (n=23 < 30)
```

### BP — medium tonic

- size 105 (LHS-only 39), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 25.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.8 mV, sd v 9.48 mV
- mean coefficient vector (all): a=0.04786, b=0.199, c=-63.48, d=6.554
- mean coefficient vector (LHS-only, authoritative): a=0.02868, b=0.2493, c=-58.05, d=6.718
- persistence 1.00, DBCV contribution +0.0046  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 84 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0458                             0.4371
b -0.7241  0.0827                           
c  0.0263  0.2073  10.9979                  
d  0.2830 -0.0596   0.0520  3.5269          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0268                             0.2979
b -0.6778  0.0717                           
c  0.2186  0.1564  11.5190                  
d  0.1074  0.1994  -0.0354  3.2175          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.26 (> 0.2); trust the LHS-only matrix.

### BQ — slow tonic

- size 103 (LHS-only 37), mean membership prob 0.848, noise adjacency 0.000
- medians: rate 12.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -66.8 mV, sd v 7.45 mV
- mean coefficient vector (all): a=0.02131, b=0.1805, c=-64.12, d=6.641
- mean coefficient vector (LHS-only, authoritative): a=0.01468, b=0.2122, c=-59.66, d=6.694
- persistence 1.00, DBCV contribution +0.0015  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 81 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0206                             0.3374
b -0.5546  0.0794                           
c  0.1207  0.0919  11.1907                  
d  0.0859  0.3814   0.0721  3.4499          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0099                             0.2511
b -0.6273  0.0805                           
c  0.2019  0.1465  11.9892                  
d -0.2385  0.5998   0.0196  3.2487          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.32 (> 0.2); trust the LHS-only matrix.

### BR — fast tonic

- size 101 (LHS-only 64), mean membership prob 0.883, noise adjacency 0.000
- medians: rate 169.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -53.5 mV, sd v 15.89 mV
- mean coefficient vector (all): a=0.09118, b=0.2638, c=-64.81, d=3.031
- mean coefficient vector (LHS-only, authoritative): a=0.09804, b=0.2729, c=-67.62, d=3.371
- persistence 1.00, DBCV contribution +0.0068  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 60 parameter-space components, 2 of them holding >= 5% of the cluster, largest 8%
  component means:
    - n=8: a=0.1184, b=0.2696, c=-76.47, d=3.042
    - n=6: a=0.05438, b=0.3326, c=-69.72, d=2.627

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0613                             0.4284
b -0.0393  0.0614                           
c  0.0696 -0.1616  10.5700                  
d  0.7283  0.4880   0.1486  2.7671          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0614                            0.3779
b  0.0168  0.0553                          
c -0.1630 -0.2558  9.0194                  
d  0.7564  0.4933 -0.0933  2.7385          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.24 (> 0.2); trust the LHS-only matrix.

### BS — medium tonic

- size 101 (LHS-only 30), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 51.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -61.7 mV, sd v 11.95 mV
- mean coefficient vector (all): a=0.0669, b=0.2371, c=-62.78, d=6.857
- mean coefficient vector (LHS-only, authoritative): a=0.05967, b=0.2361, c=-62.18, d=7.28
- persistence 0.20, DBCV contribution +0.0056  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 72 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0513                             0.4132
b -0.6110  0.0737                           
c  0.3301  0.1703  11.8583                  
d  0.5184 -0.0635   0.3181  3.5955          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0444                            0.3355
b -0.7581  0.0628                          
c  0.4491 -0.0498  8.8446                  
d  0.5086 -0.2053  0.3362  3.2355          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.22 (> 0.2); trust the LHS-only matrix.

### BT — medium tonic

- size 100 (LHS-only 30), mean membership prob 0.983, noise adjacency 0.000
- medians: rate 30.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.0 mV, sd v 10.12 mV
- mean coefficient vector (all): a=0.04934, b=0.2036, c=-62.73, d=6.676
- mean coefficient vector (LHS-only, authoritative): a=0.03962, b=0.231, c=-59.09, d=7.105
- persistence 1.00, DBCV contribution +0.0049  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 85 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0442                             0.4119
b -0.6921  0.0762                           
c  0.1756  0.1583  10.9721                  
d  0.3665 -0.1429   0.1260  3.4527          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0279                             0.3041
b -0.6113  0.0684                           
c  0.2168  0.1294  10.9442                  
d  0.2175 -0.2551  -0.2979  2.9161          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.42 (> 0.2); trust the LHS-only matrix.

### BU — slow tonic

- size 99 (LHS-only 31), mean membership prob 0.990, noise adjacency 0.000
- medians: rate 16.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -66.2 mV, sd v 8.00 mV
- mean coefficient vector (all): a=0.03312, b=0.1771, c=-64.16, d=6.54
- mean coefficient vector (LHS-only, authoritative): a=0.02134, b=0.2019, c=-59.6, d=7.005
- persistence 0.80, DBCV contribution +0.0043  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 75 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0341                             0.4239
b -0.6214  0.0773                           
c  0.1471 -0.0057  10.7168                  
d  0.1948  0.1657   0.1693  3.4691          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0191                             0.3235
b -0.6789  0.0667                           
c  0.3407  0.0107  10.8487                  
d  0.0660  0.1323   0.1058  2.6986          
```

### BV — fast tonic

- size 99 (LHS-only 45), mean membership prob 0.997, noise adjacency 0.000
- medians: rate 80.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -59.3 mV, sd v 13.14 mV
- mean coefficient vector (all): a=0.07802, b=0.239, c=-66.02, d=5.871
- mean coefficient vector (LHS-only, authoritative): a=0.08052, b=0.2436, c=-67.21, d=6.536
- persistence 0.80, DBCV contribution +0.0064

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0572                            0.4115
b -0.3230  0.0654                          
c  0.5160  0.0134  8.9167                  
d  0.6869  0.2522  0.3738  3.9683          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0540                            0.3204
b -0.4796  0.0569                          
c  0.3767  0.0485  8.6365                  
d  0.7193  0.0607  0.2909  3.2883          
```

### BW — slow tonic [mixed: 25% bursting]

- size 96 (LHS-only 47), mean membership prob 0.870, noise adjacency 0.000
- medians: rate 8.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.7 mV, sd v 6.63 mV
- mean coefficient vector (all): a=0.01316, b=0.1584, c=-60.67, d=6.408
- mean coefficient vector (LHS-only, authoritative): a=0.009852, b=0.1839, c=-55.4, d=6.147
- persistence 1.00, DBCV contribution +0.0020  **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 73 parameter-space components, 2 of them holding >= 5% of the cluster, largest 6%
  component means:
    - n=6: a=0.01348, b=0.1297, c=-64.93, d=8.71
    - n=5: a=0.006941, b=0.2432, c=-47.23, d=5.768

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0090                             0.2515
b -0.5746  0.0586                           
c -0.1461  0.5322  11.8379                  
d  0.2031  0.0957   0.0007  3.3453          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0044                             0.1731
b -0.6255  0.0617                           
c -0.1360  0.6283  11.6475                  
d  0.0969  0.0972   0.0579  3.2727          
```

### BY — fast tonic

- size 94 (LHS-only 63), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 138.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -55.6 mV, sd v 15.22 mV
- mean coefficient vector (all): a=0.08439, b=0.2607, c=-67.39, d=3.668
- mean coefficient vector (LHS-only, authoritative): a=0.09245, b=0.272, c=-68.55, d=4.462
- persistence 1.00, DBCV contribution +0.0022

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0639                            0.5069
b -0.0245  0.0580                          
c  0.0102  0.0495  7.9344                  
d  0.7220  0.4738  0.0272  3.3625          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0546                            0.3789
b -0.0845  0.0521                          
c  0.0143 -0.0137  6.5822                  
d  0.7639  0.4424  0.0291  3.2138          
```

### CA — fast tonic, +accelerating, +irregular

- size 94 (LHS-only 30), mean membership prob 0.908, noise adjacency 0.000
- medians: rate 195.0 Hz, CV(ISI) 0.612, burst index 0.000, spikes/burst 1.00, adaptation -0.240, mean v -47.3 mV, sd v 14.53 mV
- mean coefficient vector (all): a=0.09793, b=0.256, c=-49.04, d=4.877
- mean coefficient vector (LHS-only, authoritative): a=0.08519, b=0.2517, c=-49.07, d=4.058
- persistence 0.20, DBCV contribution +0.0000  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 34 parameter-space components, 2 of them holding >= 5% of the cluster, largest 24%
  component means:
    - n=23: a=0.06453, b=0.2913, c=-50.8, d=3.75
    - n=23: a=0.1437, b=0.2406, c=-46.98, d=6.237

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0534                            0.3652
b -0.0317  0.0693                          
c  0.6450  0.3166  6.0534                  
d  0.8221  0.3173  0.6039  2.4672          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0493                            0.3308
b -0.1847  0.0673                          
c  0.4676  0.1703  5.0353                  
d  0.7840  0.3055  0.6575  2.2678          
```

### CB — fast tonic

- size 94 (LHS-only 68), mean membership prob 0.920, noise adjacency 0.000
- medians: rate 129.8 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -56.7 mV, sd v 15.15 mV
- mean coefficient vector (all): a=0.08734, b=0.2567, c=-68.67, d=3.92
- mean coefficient vector (LHS-only, authoritative): a=0.09483, b=0.2636, c=-68.67, d=4.75
- persistence 1.00, DBCV contribution +0.0041

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0603                            0.4531
b -0.2302  0.0610                          
c  0.1321 -0.2395  7.7338                  
d  0.6490  0.4530 -0.0325  3.3839          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0567                            0.3497
b -0.1259  0.0544                          
c -0.0378 -0.1260  6.9935                  
d  0.7117  0.4871 -0.1257  3.3899          
```

### CC — medium tonic [mixed: 21% bursting]

- size 94 (LHS-only 21), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 42.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.2 mV, sd v 11.10 mV
- mean coefficient vector (all): a=0.06492, b=0.2275, c=-60.42, d=6.574
- mean coefficient vector (LHS-only, authoritative): a=0.05049, b=0.2421, c=-59.43, d=6.917
- persistence 1.00, DBCV contribution +0.0020  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 81 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0487                             0.3901
b -0.6808  0.0848                           
c  0.3034  0.1591  12.4728                  
d  0.1957  0.2043   0.3924  3.2966          

LHS-only: insufficient-sample (n=21 < 30)
```

### CD — fast tonic

- size 93 (LHS-only 50), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 87.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -59.5 mV, sd v 13.72 mV
- mean coefficient vector (all): a=0.07936, b=0.2285, c=-69.3, d=5.134
- mean coefficient vector (LHS-only, authoritative): a=0.07813, b=0.2376, c=-69.12, d=5.438
- persistence 1.00, DBCV contribution +0.0045

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0561                            0.4103
b -0.3130  0.0623                          
c  0.2342 -0.0464  7.7936                  
d  0.5765  0.3984  0.0860  3.9495          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0525                            0.3796
b -0.2845  0.0570                          
c  0.2507 -0.0252  6.7542                  
d  0.5865  0.3738  0.1306  3.6966          
```

### CE — fast tonic

- size 89 (LHS-only 65), mean membership prob 0.983, noise adjacency 0.000
- medians: rate 163.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -47.8 mV, sd v 15.77 mV
- mean coefficient vector (all): a=0.09247, b=0.2631, c=-59.58, d=2.702
- mean coefficient vector (LHS-only, authoritative): a=0.09804, b=0.2707, c=-60.37, d=2.94
- persistence 0.00, DBCV contribution +0.0044  **UNSTABLE**

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0593                             0.4367
b -0.0400  0.0696                           
c  0.0055 -0.2050  11.2197                  
d  0.5984  0.4139  -0.0220  2.6772          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0563                             0.3922
b  0.0009  0.0651                           
c -0.1399 -0.2094  10.7646                  
d  0.5941  0.4211  -0.1926  2.6589          
```

### CF — slow tonic

- size 88 (LHS-only 36), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 16.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -66.6 mV, sd v 8.13 mV
- mean coefficient vector (all): a=0.02456, b=0.1837, c=-65.79, d=6.89
- mean coefficient vector (LHS-only, authoritative): a=0.01817, b=0.2033, c=-61.33, d=6.928
- persistence 0.60, DBCV contribution +0.0006  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 63 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0212                             0.3369
b -0.6534  0.0732                           
c -0.0850  0.2478  10.0089                  
d  0.0921  0.2069   0.0604  3.4839          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0122                             0.2687
b -0.7033  0.0655                           
c  0.3532  0.1125  10.7514                  
d  0.2004  0.0455   0.1513  3.2362          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.44 (> 0.2); trust the LHS-only matrix.

### CG — medium tonic

- size 88 (LHS-only 24), mean membership prob 0.993, noise adjacency 0.000
- medians: rate 26.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.3 mV, sd v 9.32 mV
- mean coefficient vector (all): a=0.04777, b=0.1946, c=-62.18, d=6.154
- mean coefficient vector (LHS-only, authoritative): a=0.02307, b=0.2327, c=-64.43, d=6.219
- persistence 0.60, DBCV contribution +0.0041  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 73 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0464                             0.4179
b -0.6886  0.0823                           
c  0.2176 -0.0639  10.2506                  
d -0.0008  0.2875   0.1884  3.4140          

LHS-only: insufficient-sample (n=24 < 30)
```

### CH — medium tonic

- size 87 (LHS-only 30), mean membership prob 0.964, noise adjacency 0.000
- medians: rate 27.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.1 mV, sd v 9.80 mV
- mean coefficient vector (all): a=0.04586, b=0.2035, c=-62.05, d=6.117
- mean coefficient vector (LHS-only, authoritative): a=0.03305, b=0.2228, c=-58.66, d=6.404
- persistence 1.00, DBCV contribution +0.0049  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 76 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0417                             0.4039
b -0.6009  0.0792                           
c  0.3187  0.0987  11.1921                  
d  0.1860  0.0338   0.2079  3.3442          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0200                             0.2565
b -0.4143  0.0623                           
c  0.5107  0.2230  11.8527                  
d -0.0184  0.3064   0.0787  2.9879          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.27 (> 0.2); trust the LHS-only matrix.

### CI — fast tonic

- size 87 (LHS-only 64), mean membership prob 0.930, noise adjacency 0.000
- medians: rate 240.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -45.3 mV, sd v 16.64 mV
- mean coefficient vector (all): a=0.09298, b=0.2658, c=-57.85, d=1.706
- mean coefficient vector (LHS-only, authoritative): a=0.09637, b=0.2692, c=-58.27, d=1.65
- persistence 1.00, DBCV contribution +0.0052  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 63 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0612                             0.4253
b -0.0402  0.0668                           
c  0.0493 -0.3140  12.5890                  
d  0.5647  0.3354   0.2586  1.8006          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0625                             0.4344
b -0.0537  0.0627                           
c -0.0918 -0.3197  12.2422                  
d  0.5430  0.2554   0.1190  1.5488          
```

### CJ — medium tonic

- size 84 (LHS-only 27), mean membership prob 0.998, noise adjacency 0.000
- medians: rate 43.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.4 mV, sd v 10.98 mV
- mean coefficient vector (all): a=0.06288, b=0.2162, c=-65.06, d=5.822
- mean coefficient vector (LHS-only, authoritative): a=0.06078, b=0.2237, c=-61.36, d=6.176
- persistence 0.00, DBCV contribution +0.0026  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 63 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0490                             0.4378
b -0.6525  0.0810                           
c  0.3169  0.1385  12.0308                  
d  0.4822 -0.0603   0.4051  3.4232          

LHS-only: insufficient-sample (n=27 < 30)
```

### CK — fast tonic

- size 81 (LHS-only 51), mean membership prob 0.982, noise adjacency 0.000
- medians: rate 102.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -58.2 mV, sd v 14.14 mV
- mean coefficient vector (all): a=0.09522, b=0.2399, c=-66.68, d=4.901
- mean coefficient vector (LHS-only, authoritative): a=0.09551, b=0.2559, c=-65.53, d=5.855
- persistence 1.00, DBCV contribution +0.0053  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 60 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0582                            0.3751
b -0.3522  0.0629                          
c  0.2805  0.1814  8.8003                  
d  0.4832  0.4209  0.2345  3.5944          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0524                            0.3437
b -0.5154  0.0533                          
c  0.2102  0.0979  7.4526                  
d  0.5440  0.2365  0.1461  3.3441          
```

### CL — slow bursting (2.7 spikes/burst, 2 Hz interburst), +accelerating

- size 81 (LHS-only 34), mean membership prob 0.927, noise adjacency 0.012
- medians: rate 6.0 Hz, CV(ISI) 1.211, burst index 0.625, spikes/burst 2.67, adaptation -0.992, mean v -70.9 mV, sd v 7.92 mV
- mean coefficient vector (all): a=0.006753, b=0.1171, c=-42.69, d=6.958
- mean coefficient vector (LHS-only, authoritative): a=0.00718, b=0.1236, c=-42.83, d=6.715
- persistence 0.80, DBCV contribution +0.0055

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0022                            0.1161
b -0.3788  0.0153                          
c  0.1946  0.1215  2.1832                  
d -0.0750 -0.0943 -0.1220  3.5396          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0027                            0.1372
b -0.5865  0.0182                          
c  0.1374  0.2259  1.7459                  
d -0.1053 -0.2486 -0.2204  3.3972          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.21 (> 0.2); trust the LHS-only matrix.

### CM — medium tonic

- size 79 (LHS-only 28), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 54.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -61.5 mV, sd v 11.90 mV
- mean coefficient vector (all): a=0.07526, b=0.2259, c=-63.9, d=6.117
- mean coefficient vector (LHS-only, authoritative): a=0.05952, b=0.2518, c=-63.35, d=5.96
- persistence 1.00, DBCV contribution +0.0045  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 67 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0602                             0.4377
b -0.6318  0.0786                           
c  0.4302  0.1258  11.2691                  
d  0.4019  0.1017   0.4798  3.6036          

LHS-only: insufficient-sample (n=28 < 30)
```

### CN — slow tonic

- size 79 (LHS-only 40), mean membership prob 0.972, noise adjacency 0.000
- medians: rate 8.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -67.6 mV, sd v 6.74 mV
- mean coefficient vector (all): a=0.01321, b=0.1622, c=-62.97, d=6.761
- mean coefficient vector (LHS-only, authoritative): a=0.0104, b=0.1866, c=-59.18, d=6.848
- persistence 0.20, DBCV contribution +0.0036  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 56 parameter-space components, 2 of them holding >= 5% of the cluster, largest 5%
  component means:
    - n=4: a=0.007965, b=0.1255, c=-66.54, d=2.728
    - n=4: a=0.009653, b=0.2058, c=-45.29, d=8.117

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0101                             0.2612
b -0.5835  0.0564                           
c -0.0491  0.2768  11.3409                  
d  0.2239  0.3385  -0.0293  3.5320          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0070                             0.2104
b -0.5987  0.0557                           
c -0.0980  0.2221  12.4947                  
d -0.0678  0.3650  -0.3154  3.1050          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.29 (> 0.2); trust the LHS-only matrix.

### CO — slow tonic [mixed: 22% bursting]

- size 78 (LHS-only 27), mean membership prob 0.985, noise adjacency 0.000
- medians: rate 14.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -66.0 mV, sd v 7.69 mV
- mean coefficient vector (all): a=0.03173, b=0.169, c=-63.48, d=6.329
- mean coefficient vector (LHS-only, authoritative): a=0.02814, b=0.1926, c=-58.01, d=7.377
- persistence 1.00, DBCV contribution +0.0053

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0324                             0.4093
b -0.5706  0.0734                           
c  0.1696  0.0193  12.6385                  
d  0.1238  0.1857  -0.0219  3.3131          

LHS-only: insufficient-sample (n=27 < 30)
```

### CP — medium tonic

- size 75 (LHS-only 20), mean membership prob 0.948, noise adjacency 0.000
- medians: rate 39.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -62.7 mV, sd v 10.66 mV
- mean coefficient vector (all): a=0.06348, b=0.1991, c=-61.46, d=6.069
- mean coefficient vector (LHS-only, authoritative): a=0.05018, b=0.2261, c=-56.47, d=7.266
- persistence 0.20, DBCV contribution +0.0049  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 67 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0484                             0.4299
b -0.6088  0.0740                           
c  0.1233  0.2682  11.0970                  
d  0.3309  0.0698   0.3617  3.5708          

LHS-only: insufficient-sample (n=20 < 30)
```

### CQ — fast tonic

- size 74 (LHS-only 42), mean membership prob 0.846, noise adjacency 0.000
- medians: rate 98.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -58.9 mV, sd v 13.95 mV
- mean coefficient vector (all): a=0.08944, b=0.2369, c=-66.99, d=5.254
- mean coefficient vector (LHS-only, authoritative): a=0.08977, b=0.2487, c=-66.48, d=6.394
- persistence 0.00, DBCV contribution +0.0037  **UNSTABLE**

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0633                            0.4589
b -0.4504  0.0710                          
c  0.2073 -0.0454  8.2456                  
d  0.3486  0.3400  0.2136  3.6903          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0495                            0.3528
b -0.3681  0.0612                          
c  0.1208 -0.1610  7.2301                  
d  0.4236  0.3533  0.0021  3.4304          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.21 (> 0.2); trust the LHS-only matrix.

### CR — very slow tonic

- size 74 (LHS-only 34), mean membership prob 0.978, noise adjacency 0.041
- medians: rate 2.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -68.4 mV, sd v 5.34 mV
- mean coefficient vector (all): a=0.006784, b=0.109, c=-66.3, d=7.594
- mean coefficient vector (LHS-only, authoritative): a=0.006983, b=0.1081, c=-63.11, d=7.541
- persistence 0.20, DBCV contribution +0.0001  **UNSTABLE** **MULTIMODAL** (3 components) **BELOW-RATE-FLOOR**
- **BELOW-RATE-FLOOR**: median rate 2.5 Hz is under §6's 4 Hz spectral-estimate floor (5 spikes in the 2000 ms window), so its spectrum rests on few cycles (§19 A9)
- pooled mean is **not interpretable**: 27 parameter-space components, 3 of them holding >= 5% of the cluster, largest 38%
  component means:
    - n=28: a=0.006622, b=0.1106, c=-75.98, d=10.08
    - n=6: a=0.006401, b=0.1117, c=-55.7, d=8.576
    - n=4: a=0.008849, b=0.1063, c=-58.34, d=10.24

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0016                             0.0939
b -0.4942  0.0079                           
c  0.1329  0.1515  10.9067                  
d  0.0321  0.3389  -0.3555  3.4553          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0015                             0.0895
b -0.4068  0.0070                           
c  0.1236  0.3441  10.7035                  
d  0.2470  0.0210  -0.3869  3.1308          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.32 (> 0.2); trust the LHS-only matrix.

### CS — medium tonic

- size 73 (LHS-only 31), mean membership prob 0.996, noise adjacency 0.000
- medians: rate 43.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -61.7 mV, sd v 10.93 mV
- mean coefficient vector (all): a=0.06056, b=0.2169, c=-62.61, d=5.669
- mean coefficient vector (LHS-only, authoritative): a=0.04678, b=0.233, c=-61.44, d=6.76
- persistence 1.00, DBCV contribution +0.0034  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 65 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0509                             0.4051
b -0.6483  0.0753                           
c  0.2101  0.1636  11.2297                  
d  0.1427  0.2512   0.4058  3.6452          

LHS-only (design-unbiased, authoritative):
        a       b        c       d  sd_log_a
a  0.0311                             0.2748
b -0.3942  0.0573                           
c  0.6109  0.0946  11.8798                  
d  0.1802  0.5370   0.3047  3.1429          
```

> Sampling-design flag: all-sample and LHS-only correlations differ by up to |dr| = 0.40 (> 0.2); trust the LHS-only matrix.

### CT — fast tonic

- size 72 (LHS-only 41), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 116.8 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -57.1 mV, sd v 14.62 mV
- mean coefficient vector (all): a=0.069, b=0.2394, c=-68.59, d=3.386
- mean coefficient vector (LHS-only, authoritative): a=0.08541, b=0.2535, c=-67.53, d=4.932
- persistence 0.00, DBCV contribution +0.0014  **UNSTABLE**

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0530                            0.4807
b -0.0271  0.0682                          
c  0.0565  0.0495  7.7835                  
d  0.6654  0.5405  0.1862  3.3784          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0509                            0.4326
b -0.0059  0.0603                          
c  0.0795  0.0347  7.2193                  
d  0.6899  0.5850  0.2071  3.5399          
```

### CU — medium tonic

- size 72 (LHS-only 20), mean membership prob 0.969, noise adjacency 0.000
- medians: rate 24.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -64.4 mV, sd v 9.21 mV
- mean coefficient vector (all): a=0.05063, b=0.1797, c=-62.07, d=6.324
- mean coefficient vector (LHS-only, authoritative): a=0.03254, b=0.2217, c=-58.94, d=6.091
- persistence 0.00, DBCV contribution +0.0009  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 62 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0409                             0.4302
b -0.7441  0.0767                           
c  0.2494 -0.0698  11.6990                  
d  0.2540  0.0331   0.0771  2.9988          

LHS-only: insufficient-sample (n=20 < 30)
```

### CV — fast tonic

- size 70 (LHS-only 24), mean membership prob 0.982, noise adjacency 0.000
- medians: rate 89.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -59.2 mV, sd v 13.75 mV
- mean coefficient vector (all): a=0.09427, b=0.2337, c=-67.94, d=5.13
- mean coefficient vector (LHS-only, authoritative): a=0.08077, b=0.26, c=-66.4, d=5.987
- persistence 0.40, DBCV contribution +0.0042  **UNSTABLE**

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0604                            0.4279
b -0.5629  0.0662                          
c  0.0706  0.3292  8.0269                  
d  0.4627  0.2230  0.2587  3.7061          

LHS-only: insufficient-sample (n=24 < 30)
```

### CW — medium tonic

- size 69 (LHS-only 24), mean membership prob 0.971, noise adjacency 0.000
- medians: rate 35.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -63.8 mV, sd v 10.49 mV
- mean coefficient vector (all): a=0.06096, b=0.2105, c=-61.74, d=7.112
- mean coefficient vector (LHS-only, authoritative): a=0.05101, b=0.2247, c=-58.07, d=7.047
- persistence 0.00, DBCV contribution +0.0045  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 59 parameter-space components, 0 of them holding >= 5% of the cluster, largest 4%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0545                             0.4382
b -0.7087  0.0739                           
c  0.1570  0.1474  11.7919                  
d  0.3214 -0.1092   0.2827  3.5794          

LHS-only: insufficient-sample (n=24 < 30)
```

### CY — medium tonic

- size 68 (LHS-only 22), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 48.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -61.4 mV, sd v 11.71 mV
- mean coefficient vector (all): a=0.07795, b=0.2123, c=-60.16, d=6.845
- mean coefficient vector (LHS-only, authoritative): a=0.07893, b=0.2149, c=-56.28, d=7.69
- persistence 0.80, DBCV contribution +0.0032

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0517                             0.3926
b -0.5883  0.0770                           
c  0.2590  0.1612  11.8128                  
d  0.2954  0.2803   0.2679  3.4427          

LHS-only: insufficient-sample (n=22 < 30)
```

### DA — medium tonic

- size 65 (LHS-only 17), mean membership prob 0.998, noise adjacency 0.000
- medians: rate 66.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -60.2 mV, sd v 12.75 mV
- mean coefficient vector (all): a=0.07916, b=0.2348, c=-65.35, d=5.605
- mean coefficient vector (LHS-only, authoritative): a=0.0841, b=0.2233, c=-66.34, d=6.626
- persistence 0.00, DBCV contribution +0.0040  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 57 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0557                             0.4133
b -0.4007  0.0715                           
c  0.3563  0.3730  11.1354                  
d  0.4132  0.2514   0.3607  3.4403          

LHS-only: insufficient-sample (n=17 < 30)
```

### DB — medium tonic

- size 64 (LHS-only 15), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 37.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -63.1 mV, sd v 10.55 mV
- mean coefficient vector (all): a=0.05744, b=0.2165, c=-64.69, d=6.19
- mean coefficient vector (LHS-only, authoritative): a=0.03769, b=0.2286, c=-60.59, d=5.883
- persistence 0.80, DBCV contribution +0.0031  **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 55 parameter-space components, 0 of them holding >= 5% of the cluster, largest 5%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0528                             0.4287
b -0.6767  0.0877                           
c -0.0054  0.2914  10.6626                  
d  0.3191  0.0291   0.1159  3.5994          

LHS-only: insufficient-sample (n=15 < 30)
```

### DC — slow tonic

- size 62 (LHS-only 24), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 9.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -67.9 mV, sd v 7.06 mV
- mean coefficient vector (all): a=0.01728, b=0.17, c=-64.01, d=7.183
- mean coefficient vector (LHS-only, authoritative): a=0.01038, b=0.1956, c=-59.56, d=7.276
- persistence 0.40, DBCV contribution +0.0021  **UNSTABLE** **MULTIMODAL** (0 components)
- pooled mean is **not interpretable**: 54 parameter-space components, 0 of them holding >= 5% of the cluster, largest 3%
  no single component reaches 5%, so none is summarised

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0172                             0.3280
b -0.5565  0.0691                           
c -0.1148  0.3332  10.8050                  
d  0.1933  0.3741   0.0044  3.6464          

LHS-only: insufficient-sample (n=24 < 30)
```

### DD — medium tonic

- size 62 (LHS-only 16), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 23.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation -0.000, mean v -65.4 mV, sd v 9.12 mV
- mean coefficient vector (all): a=0.03923, b=0.2038, c=-64.86, d=6.774
- mean coefficient vector (LHS-only, authoritative): a=0.0184, b=0.2525, c=-61.6, d=5.738
- persistence 0.40, DBCV contribution +0.0008  **UNSTABLE**

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0422                             0.4187
b -0.6820  0.0823                           
c -0.0318  0.0306  10.4592                  
d  0.2713 -0.0195   0.1023  3.6359          

LHS-only: insufficient-sample (n=16 < 30)
```

### DE — medium tonic

- size 60 (LHS-only 19), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 24.5 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -65.4 mV, sd v 9.50 mV
- mean coefficient vector (all): a=0.04547, b=0.203, c=-61.52, d=7.228
- mean coefficient vector (LHS-only, authoritative): a=0.02907, b=0.2338, c=-56.03, d=6.798
- persistence 0.00, DBCV contribution +0.0010  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 49 parameter-space components, 2 of them holding >= 5% of the cluster, largest 5%
  component means:
    - n=3: a=0.008017, b=0.3169, c=-61.89, d=4.997
    - n=3: a=0.0137, b=0.3405, c=-56.16, d=10.91

```
all samples (descriptive):
        a       b        c       d  sd_log_a
a  0.0444                             0.4122
b -0.6102  0.0839                           
c  0.1518  0.1335  11.8616                  
d  0.1347  0.0917   0.0983  3.3233          

LHS-only: insufficient-sample (n=19 < 30)
```

### DF — fast tonic

- size 60 (LHS-only 39), mean membership prob 1.000, noise adjacency 0.000
- medians: rate 123.0 Hz, CV(ISI) 0.000, burst index 0.000, spikes/burst 1.00, adaptation +0.000, mean v -56.7 mV, sd v 14.91 mV
- mean coefficient vector (all): a=0.08335, b=0.2472, c=-68.23, d=3.562
- mean coefficient vector (LHS-only, authoritative): a=0.08872, b=0.2601, c=-67.19, d=4.632
- persistence 0.00, DBCV contribution +0.0031  **UNSTABLE** **MULTIMODAL** (2 components)
- pooled mean is **not interpretable**: 47 parameter-space components, 2 of them holding >= 5% of the cluster, largest 7%
  component means:
    - n=4: a=0.007157, b=0.301, c=-65.86, d=0.5072
    - n=3: a=0.007447, b=0.1951, c=-77.17, d=0.1404

```
all samples (descriptive):
        a       b       c       d  sd_log_a
a  0.0628                            0.4844
b -0.2938  0.0668                          
c  0.1115 -0.0842  8.5899                  
d  0.5855  0.4200  0.1393  3.4097          

LHS-only (design-unbiased, authoritative):
        a       b       c       d  sd_log_a
a  0.0586                            0.4219
b -0.1948  0.0606                          
c  0.0451 -0.1350  8.6197                  
d  0.7628  0.3577  0.0432  3.5445          
```

## 4. Canonical type placement (S1, §13.1)

| type | a | b | c | d | cluster | p | approx_predict | note |
|---|---|---|---|---|---|---|---|---|
| RS | 0.02 | 0.2 | -65 | 8 | AB | 1.000 | AB/1.000 |  |
| IB | 0.02 | 0.2 | -55 | 4 | O | 1.000 | noise/0.000 |  |
| CH | 0.02 | 0.2 | -50 | 2 | R | 0.488 | AL/1.000 |  |
| FS | 0.1 | 0.2 | -65 | 2 | BY | 1.000 | Q/1.000 |  |
| LTS | 0.02 | 0.25 | -65 | 2 | BH | 1.000 | BH/1.000 |  |
| TC | 0.02 | 0.25 | -65 | 0.05 | AV | 0.772 | AV/0.772 | expected generic tonic (§13.1) |
| RZ | 0.1 | 0.26 | -65 | 2 | P | 1.000 | P/1.000 | expected generic tonic (§13.1) |

`approximate_predict` disagrees with the in-sample label for IB, CH, FS — it reassigns from the condensed tree's exemplars, so a point near a cluster boundary can land either side. The in-sample label is the authoritative one.

TC and RZ are expected to land in generic tonic clusters: their published behaviours need a hyperpolarising step and timed brief stimuli respectively, neither reachable at constant `I` (§13.1, §18 item 6).

## 5. Stability (S5, §13.2)

- 5 independent unseeded UMAP embeddings of 12000 points, 38.2s
- pairwise ARI: 0.789, 0.731, 0.786, 0.793, 0.756, 0.771, 0.811, 0.778, 0.773, 0.807
- mean pairwise ARI **0.780** (threshold 0.6; a run landing at 0.60 ± 0.03 should be re-run)
- UNSTABLE clusters (persistence < 0.6): A, E, G, AA, AM, AN, AQ, AV, BA, BE, BO, BS, CA, CE, CJ, CN, CP, CQ, CR, CT, CU, CV, CW, DA, DC, DD, DE, DF

## 6. Contiguity (S4, §13.3)

- single-linkage cut at the 95th percentile of the global k=5 NN distance = 0.1074 (normalised domain)
- MULTIMODAL clusters: B (113 components, largest 15%), C (90 components, largest 14%), I (72 components, largest 18%), J (147 components, largest 3%), K (109 components, largest 8%), L (131 components, largest 4%), M (109 components, largest 4%), N (110 components, largest 6%), P (69 components, largest 16%), R (119 components, largest 4%), S (110 components, largest 3%), T (122 components, largest 3%), U (121 components, largest 4%), V (108 components, largest 5%), W (100 components, largest 4%), AA (74 components, largest 6%), AB (104 components, largest 5%), AC (102 components, largest 3%), AD (102 components, largest 4%), AF (83 components, largest 5%), AH (79 components, largest 4%), AI (104 components, largest 3%), AK (74 components, largest 4%), AL (100 components, largest 4%), AM (99 components, largest 4%), AN (92 components, largest 4%), AO (91 components, largest 4%), AP (91 components, largest 3%), AR (99 components, largest 5%), AS (100 components, largest 3%), AT (85 components, largest 4%), AU (89 components, largest 3%), AV (48 components, largest 18%), AW (89 components, largest 4%), AY (92 components, largest 3%), BE (93 components, largest 3%), BG (80 components, largest 4%), BI (80 components, largest 5%), BJ (86 components, largest 4%), BK (81 components, largest 4%), BL (81 components, largest 5%), BM (92 components, largest 3%), BP (84 components, largest 4%), BQ (81 components, largest 4%), BR (60 components, largest 8%), BS (72 components, largest 5%), BT (85 components, largest 4%), BU (75 components, largest 4%), BW (73 components, largest 6%), CA (34 components, largest 24%), CC (81 components, largest 4%), CF (63 components, largest 3%), CG (73 components, largest 5%), CH (76 components, largest 3%), CI (63 components, largest 5%), CJ (63 components, largest 5%), CK (60 components, largest 5%), CM (67 components, largest 3%), CN (56 components, largest 5%), CP (67 components, largest 4%), CR (27 components, largest 38%), CS (65 components, largest 3%), CU (62 components, largest 4%), CW (59 components, largest 4%), DA (57 components, largest 5%), DB (55 components, largest 5%), DC (54 components, largest 3%), DE (49 components, largest 5%), DF (47 components, largest 7%)
- The cut is a *global* density threshold applied to each cluster's own point set, so small clusters fragment simply because their own points are sparser than the global k=5 NN scale; the largest-component fraction is the number to read alongside the flag.
- NEAR-FLOOR clusters (§19 A5): none
- BELOW-RATE-FLOOR clusters (§19 A9): CR

## 7. eom → leaf containment (§12.4)

| eom | size | leaf | containment |
|---|---|---|---|
| AC | 145 | 0 | 1.000 |
| K | 216 | 7 | 0.722 |
| CO | 78 | 1 | 1.000 |
| AL | 136 | 2 | 1.000 |
| R | 182 | 3 | 1.000 |
| Q | 186 | 4 | 1.000 |
| BY | 94 | 5 | 1.000 |
| S | 178 | 8 | 1.000 |
| AR | 127 | 9 | 1.000 |
| U | 169 | 10 | 1.000 |
| AM | 136 | 11 | 1.000 |
| BU | 99 | 12 | 1.000 |
| CF | 88 | 13 | 1.000 |
| DB | 64 | 14 | 1.000 |
| F | 247 | 30 | 0.360 |
| CT | 72 | 15 | 1.000 |
| B | 355 | 18 | 0.386 |
| D | 272 | 28 | 0.577 |
| BB | 116 | 16 | 1.000 |
| CH | 87 | 17 | 1.000 |
| O | 198 | 23 | 0.525 |
| AO | 129 | 20 | 1.000 |
| BT | 100 | 21 | 1.000 |
| AE | 142 | 24 | 1.000 |
| BN | 105 | 25 | 1.000 |
| BL | 108 | 26 | 1.000 |
| AP | 129 | 27 | 1.000 |
| AI | 139 | 32 | 1.000 |
| V | 165 | 33 | 1.000 |
| BI | 109 | 34 | 1.000 |
| BG | 113 | 35 | 1.000 |
| CQ | 74 | 36 | 1.000 |
| BJ | 109 | 37 | 1.000 |
| BK | 109 | 38 | 1.000 |
| BC | 115 | 39 | 1.000 |
| BQ | 103 | 40 | 1.000 |
| BF | 114 | 41 | 1.000 |
| CK | 81 | 42 | 1.000 |
| BW | 96 | 43 | 1.000 |
| AH | 140 | 44 | 1.000 |
| C | 327 | 50 | 0.489 |
| I | 239 | 52 | 0.540 |
| AD | 144 | 45 | 1.000 |
| CM | 79 | 46 | 1.000 |
| CN | 79 | 47 | 1.000 |
| DC | 62 | 48 | 1.000 |
| N | 200 | 54 | 0.405 |
| AF | 142 | 55 | 1.000 |
| BR | 101 | 56 | 1.000 |
| BO | 105 | 57 | 1.000 |
| CI | 87 | 58 | 1.000 |
| AY | 120 | 59 | 1.000 |
| E | 261 | 81 | 0.398 |
| Y | 152 | 60 | 1.000 |
| H | 246 | 61 | 1.000 |
| AU | 122 | 62 | 1.000 |
| T | 174 | 63 | 1.000 |
| AQ | 128 | 64 | 1.000 |
| CD | 93 | 65 | 1.000 |
| CV | 70 | 66 | 1.000 |
| CL | 81 | 67 | 1.000 |
| AA | 149 | 68 | 1.000 |
| AK | 137 | 69 | 1.000 |
| DD | 62 | 70 | 1.000 |
| AB | 149 | 71 | 1.000 |
| CG | 88 | 72 | 1.000 |
| BP | 105 | 73 | 1.000 |
| AS | 126 | 74 | 1.000 |
| J | 220 | 98 | 0.473 |
| DE | 60 | 75 | 1.000 |
| CU | 72 | 76 | 1.000 |
| CE | 89 | 77 | 1.000 |
| AV | 121 | 78 | 1.000 |
| AN | 133 | 79 | 1.000 |
| P | 191 | 82 | 1.000 |
| AT | 126 | 83 | 1.000 |
| AG | 142 | 84 | 1.000 |
| AW | 121 | 85 | 1.000 |
| CY | 68 | 86 | 1.000 |
| CW | 69 | 87 | 1.000 |
| A | 365 | -1 | 0.389 |
| AJ | 139 | 88 | 1.000 |
| BA | 119 | 89 | 1.000 |
| BV | 99 | 90 | 1.000 |
| DA | 65 | 91 | 1.000 |
| BD | 115 | 92 | 1.000 |
| G | 247 | 103 | 0.453 |
| BS | 101 | 95 | 1.000 |
| M | 210 | 96 | 1.000 |
| BH | 111 | 97 | 1.000 |
| CA | 94 | 100 | 1.000 |
| CR | 74 | 101 | 1.000 |
| CP | 75 | 104 | 1.000 |
| CB | 94 | 105 | 1.000 |
| DF | 60 | 106 | 1.000 |
| BM | 108 | 107 | 1.000 |
| W | 156 | 108 | 1.000 |
| BE | 115 | 109 | 1.000 |
| L | 215 | 110 | 1.000 |
| CS | 73 | 111 | 1.000 |
| CJ | 84 | 112 | 1.000 |
| CC | 94 | 113 | 1.000 |

## 8. Selection sweep (§12.4)

| stage | alpha | nn | mcs | ms | method | k | noise | rel. validity | score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 15 | 60 | 25 | eom | 103 | 0.063 | 0.6051 | 0.6051 |
| 1 | 0 | 15 | 30 | 25 | eom | 183 | 0.060 | 0.6003 | 0.6003 |
| 1 | 0 | 15 | 60 | 10 | eom | 118 | 0.099 | 0.5822 | 0.5822 |
| 1 | 0 | 15 | 30 | 5 | eom | 244 | 0.053 | 0.5702 | 0.5702 |
| 1 | 0 | 15 | 60 | 5 | eom | 115 | 0.109 | 0.5417 | 0.5417 |
| 1 | 0 | 15 | 30 | 10 | eom | 232 | 0.045 | 0.5414 | 0.5414 |
| 1 | 0 | 15 | 60 | 10 | leaf | 126 | 0.108 | 0.5323 | 0.5323 |
| 1 | 0 | 50 | 120 | 5 | eom | 61 | 0.020 | 0.5319 | 0.5319 |
| 1 | 0 | 30 | 60 | 5 | eom | 118 | 0.057 | 0.5250 | 0.5250 |
| 1 | 0 | 50 | 60 | 5 | eom | 115 | 0.057 | 0.5179 | 0.5179 |
| 1 | 0 | 15 | 60 | 5 | leaf | 121 | 0.121 | 0.5127 | 0.5127 |
| 1 | 0 | 15 | 30 | 5 | leaf | 261 | 0.071 | 0.5109 | 0.5109 |
| 1 | 0 | 30 | 120 | 5 | eom | 57 | 0.101 | 0.5092 | 0.5092 |
| 1 | 0 | 15 | 60 | 25 | leaf | 117 | 0.096 | 0.5036 | 0.5036 |
| 1 | 0 | 30 | 60 | 5 | leaf | 123 | 0.071 | 0.4931 | 0.4931 |
| 1 | 0 | 30 | 60 | 10 | eom | 113 | 0.049 | 0.4836 | 0.4836 |
| 1 | 0 | 50 | 120 | 10 | eom | 59 | 0.041 | 0.4777 | 0.4777 |
| 1 | 0 | 50 | 120 | 25 | eom | 54 | 0.038 | 0.4754 | 0.4754 |
| 1 | 0 | 15 | 30 | 10 | leaf | 258 | 0.064 | 0.4707 | 0.4707 |
| 1 | 0 | 50 | 120 | 5 | leaf | 66 | 0.061 | 0.4679 | 0.4679 |

(top 20 of 56 configs; full table in `sweep_table.csv`)

## 9. Acceptance criteria (§17)

| criterion | result | detail |
|---|---|---|
| convergence gate passes both checks (§7.4) | PASS | dt_int 0.0025 ms, median rel L2 0.00084; alias ratio 7.1x |
| zero samples labelled X (non-finite after refinement) | PASS | X = 0 |
| noise fraction <= 25% at the selected config | PASS | noise = 0.077 |
| >= 10 clusters at cluster_selection_method='leaf' | PASS | leaf clusters = 114 |
| mean pairwise ARI across 5 stability embeddings >= 0.6 | PASS | mean ARI = 0.780 (Monte-Carlo estimate, §13.2) |
| all 7 canonical types placed in a non-noise cluster | PASS | all placed |
| every §16 artefact present and non-empty | PASS | all present |
| samples/traces/spectra/features all have exactly N rows and identical sample_id ordering | PASS | rows = {'samples.csv': 26624, 'features_bands.csv': 26624, 'traces.csv': 26624, 'spectra_full.csv': 26624}, first/last sample_id consistent = True |
| round trip: re-reading traces.csv reproduces features_bands.csv to 1e-4 relative (mean over clustered rows, §19 A10) | PASS | mean rel L2 = 7.86e-05 over 572 clustered rows; p99 9.85e-04, max 7.23e-03 (the tail is set by bands straddling the §19 A1 floor, not by precision: it is unchanged at %.9g). The 452 quarantined rows excluded reach 1.00e+30 (§19 A7) |

## 10. Timings and measured ratios

| stage | seconds |
|---|---|
| sampling_loop | 462.0 |
| gp_fit | 234.0 |
| figures | 135.3 |
| acquire | 112.4 |
| csv_bulk | 99.0 |
| simulate | 85.0 |
| sweep_stage1 | 62.1 |
| stability | 38.2 |
| umap2_seeded | 36.9 |
| sweep_stage2 | 21.8 |
| umap10_seeded | 18.6 |
| gate | 14.9 |
| interim_hdbscan | 13.6 |
| csv_reps | 12.7 |
| representatives | 6.6 |
| roundtrip | 2.5 |
| features | 1.6 |
| csv_samples | 0.6 |
| csv_stats | 0.4 |
| hdbscan_final | 0.3 |
| contiguity | 0.1 |
| pca | 0.1 |
| **total** | **913.9** |

Measurements §12.3 and §15 require in place of their estimates:

- `umap_seeded_seconds` = 18.588
- `umap_unseeded_seconds` = 19.176
- `umap_seeded_over_unseeded` = 0.969
- `hdbscan_tree_build_seconds` = 0.225
- `hdbscan_extract_seconds` = 0.047
- `tree_over_extract` = 4.742

The single- vs multi-threaded UMAP ratio §15 assumed to be ~4x measures 0.97x here, and the tree-build-to-extraction ratio behind §12.4.1's tree reuse measures 5x.
