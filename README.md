# Izhikevich spiking-neuron cluster discovery

GPU-accelerated exploration of the four-dimensional coefficient space
`(a, b, c, d)` of the Izhikevich spiking-neuron model, looking for
distinguishable firing-pattern clusters.

The v1 design is specified in [`SPEC.md`](SPEC.md); **[`IMPROVEMENTS.md`](IMPROVEMENTS.md)
specifies v2 and is the document the code implements** — read its §1 for the
change map, §2 for the measurements that motivated it, and §13 for the
implementation assumptions. `spikes.pdf` is the reference paper
(Izhikevich 2003).

What changed in v2, in one line each:

| | change |
|---|---|
| **C1** | the clustering input is an **encoding of the ISI sequence** (a return-map KDE, not a 128-band magnitude spectrum) |
| **C2** | spike times are captured **in-kernel at `dt_int` resolution** |
| **C3** | the drive is `I = I*(θ) + Δ` — a controlled overdrive above each sample's **numerically determined rheobase**, not a fixed `I = 10` |
| **C4** | the convergence gate is judged on **spike times**, not on the feature metric |
| **C5** | the **onset transient** is featurised rather than discarded |
| **C6** | GP active learning is retired for a single **Sobol** design at 2.5× the sample count |
| **C7/C8** | **representation-health gates**, stability tests that can fail, a **nameability merge**, acceptance criteria that can fail |
| **C9** | the batched spectral path is retired; the rFFT is host-side and on demand |

## Run it

```bash
pixi install
pixi run run           # full run, ~40 min: ./output figures + ./data CSVs
pixi run quick         # ~6 min smoke run (N = 4096, no sweep)
pixi run test          # unit tests
```

Or directly, with flags:

```bash
pixi run python -m izhiclust.pipeline --help
```

| flag | effect |
|---|---|
| `--quick` | `N = 4096`, no sweep, §8.1 gates reported instead of blocking |
| `--no-sweep` | use the hand-set defaults instead of the selection sweep |
| `--sweep-fast` | drop `min_samples=25` from stage 1 (36 configs, 6 tree builds) |
| `--no-stage2` | skip the ISI representation sweep (bandwidth and block weights) |
| `--no-tree-reuse` | 54 full `HDBSCAN.fit` calls instead of 9 tree builds |
| `--compact-csv` | omit `traces.csv` / `spike_times.csv` (~9 GB of output) |
| `--gzip-csv` | write the two bulk files as `.csv.gz` |
| `--no-health-gate` | report the §8.1 gates instead of aborting on them |
| `--outdir` / `--datadir` | output locations (default `./output`, `./data`) |
| `--no-show` | never open the interactive figure window |
| `--n` | override the sample count |

## What comes out

`./output` — `clusters_umap2d.png` (2D scatter, clusters lettered, the 7
canonical types annotated), `<LETTER><k>.png` per representative (trace +
spectrum), `<LETTER>_traces.png` per cluster, `cluster_report.md`,
`sweep_table.csv`, `run_manifest.json`, `run.log`.

`./data` — `samples.csv`, `features_isi.csv` (185 columns, §3.2),
`features_isi_onset.csv` (187), `spike_times.csv` (ragged, the raw evidence
every feature is derived from), `traces.csv`, `traces/<LETTER><k>.csv`,
`cluster_stats.csv`, `corr_<LETTER>.csv`, `subthreshold_S.csv`,
`merge_pairs.csv`, `run_data.npz` (re-cluster without re-simulating).

Start with `output/cluster_report.md`: it carries the convergence gate, the
rheobase statistics, the §8.1 representation-health gates, the per-cluster
names and triangular matrices, the canonical-type placements with their
descriptor tests, the four stability tests, the nameability merge and the §9
acceptance criteria.

## Code map

| file | role |
|---|---|
| `config.py` | every constant in both specs, as frozen dataclasses |
| `kernel.cu` | DP5 + reset + in-kernel spike-time capture and diagnostics |
| `simulate.py` | `RawModule` wrapper, chunking, `dt/4` refinement, ragged spike trains |
| `rheobase.py` | analytic `I_SN`/`I_AH` brackets, GPU bisection, excitability class |
| `isi.py` | the §3.2 blocks A–D, weighting, the §3.4 classes, the §8.1 gates |
| `features.py` | host-side, on-demand rFFT for the figures and the `S` class |
| `sampling.py` | the Sobol design and the canonical seeds |
| `cluster.py` | PCA, UMAP, HDBSCAN, tree reuse, DBCV, lettering |
| `validate.py` | canonical placement (§8.3), four stability tests (§8.2), contiguity |
| `characterize.py` | naming rules and the §8.4 nameability merge |
| `stats.py` | the 5×5 triangular matrices |
| `plots.py`, `io.py` | figures; CSV/npz/manifest writing and the round-trip check |
| `pipeline.py` | orchestrator and CLI |

## Requirements

An NVIDIA GPU with CUDA 13 (developed against an RTX 500 Ada, cc 8.9, 4 GB)
and the `pixi` environment in `pixi.toml`. The pipeline aborts if the §5
convergence gate or a §8.1 representation-health gate fails.
