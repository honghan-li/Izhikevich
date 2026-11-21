# Izhikevich spiking-neuron cluster discovery

GPU-accelerated exploration of the four-dimensional coefficient space
`(a, b, c, d)` of the Izhikevich spiking-neuron model at fixed drive `I = 10`,
looking for distinguishable firing-pattern clusters.

The design is specified in [`SPEC.md`](SPEC.md) — read §1 for the objective and
§19 for the implementation's assumptions and the places where measurement
forced a change to the spec. `spikes.pdf` is the reference paper
(Izhikevich 2003).

## Run it

```bash
pixi install
pixi run run           # full run, ~25 min: ./output figures + ./data CSVs
pixi run quick         # ~4 min smoke run (n1=1024, n=512, m=6, no sweep)
pixi run test          # unit tests (§17)
```

Or directly, with flags:

```bash
pixi run python -m izhiclust.pipeline --help
```

| flag | effect |
|---|---|
| `--quick` | `n1=1024, n=512, m=6`, no sweep (§15) |
| `--no-sweep` | use the §12.4 hand-set defaults instead of the selection sweep |
| `--sweep-fast` | drop `min_samples=25` from stage 1 (36 configs, 6 tree builds) |
| `--no-stage2` | skip the spectral-tilt stage of the sweep |
| `--no-tree-reuse` | 54 full `HDBSCAN.fit` calls instead of 9 tree builds (§12.4.1) |
| `--compact-csv` | omit `traces.csv` / `spectra_full.csv` (~150 MB of output) |
| `--gzip-csv` | write the two bulk files as `.csv.gz` |
| `--csv-full-trace` | write all 25 000 trace points instead of the analysis window |
| `--ipw-stats` | add inverse-propensity-weighted correlation matrices (§14.1) |
| `--spectral-tilt A` | fix `alpha` instead of sweeping it |
| `--outdir` / `--datadir` | output locations (default `./output`, `./data`) |
| `--no-show` | never open the interactive figure window |
| `--n` / `--m` | override samples per iteration / iteration count |

## What comes out

`./output` — `clusters_umap2d.png` (2D scatter, clusters lettered, the 7
canonical types annotated), `<LETTER><k>.png` per representative (trace +
spectrum), `<LETTER>_traces.png` per cluster, `cluster_report.md`,
`sweep_table.csv`, `run_manifest.json`, `run.log`.

`./data` — `samples.csv`, `features_bands.csv`, `traces.csv`,
`spectra_full.csv`, `traces/<LETTER><k>.csv`, `cluster_stats.csv`,
`corr_<LETTER>_{all,lhs}.csv`, `run_data.npz` (re-cluster without
re-simulating). Layout and sizes: §16.2.

Start with `output/cluster_report.md`: it carries the per-cluster names, both
triangular coefficient matrices, the canonical-type placements, the stability
and contiguity results, the sweep tables and the §17 acceptance criteria.

## Code map

| file | role |
|---|---|
| `config.py` | every constant in the spec, as frozen dataclasses |
| `kernel.cu` | DP5 + reset + cascaded-boxcar output + in-kernel diagnostics |
| `simulate.py` | `RawModule` wrapper, chunking, `dt/4` refinement pass |
| `features.py` | Hann, rfft, `\|H\|` deconvolution, band-limit, hybrid banding, floor, `log1p`, L2 |
| `sampling.py` | LHS, Sobol pool, canonical seeds |
| `gp.py` | ARD-RBF GP: NLL + analytic grad, L-BFGS-B, posterior mean/var/grad |
| `acquire.py` | three-term acquisition, greedy batch with exact variance re-conditioning |
| `cluster.py` | PCA, UMAP, HDBSCAN, tree reuse, DBCV, quarantine, lettering |
| `validate.py` | canonical placement (S1), stability ARI (S5), contiguity (S4) |
| `characterize.py` | interpretable naming rules (S2) |
| `stats.py` | triangular matrices, design-unbiased and IPW variants (S3) |
| `plots.py`, `io.py` | figures; CSV/npz/manifest writing and the round-trip check |
| `pipeline.py` | orchestrator and CLI |

## Requirements

An NVIDIA GPU with CUDA 13 (developed against an RTX 500 Ada, cc 8.9, 4 GB)
and the `pixi` environment in `pixi.toml`. The pipeline aborts if the §7.4
convergence gate fails.
