"""Orchestrator for the Izhikevich cluster-discovery run (SPEC.md).

Stages, in order:
  §7.4 convergence gate -> §11 sampling/active-learning loop -> §10 quarantine
  -> §12 PCA/UMAP/HDBSCAN + staged sweep -> §13 validation -> §14 statistics
  -> §16 figures, CSVs, manifest, report.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import characterize, cluster, features, io as iox, plots, sampling, stats, validate
from .config import Config, canonical_array
from .gp import ARDGP
from .acquire import select_batch
from .simulate import Simulator, grid_for_dt, host_diagnostics


# --------------------------------------------------------------------- utils


class Log:
    def __init__(self, path: Optional[str] = None):
        self.t0 = time.time()
        self.fh = open(path, "w") if path else None

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        line = f"[{time.time() - self.t0:7.1f}s] {s}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


class Timer:
    def __init__(self):
        self.d: Dict[str, float] = {}

    def add(self, key: str, seconds: float):
        self.d[key] = self.d.get(key, 0.0) + seconds

    def section(self, key: str):
        return _Section(self, key)


class _Section:
    def __init__(self, timer: Timer, key: str):
        self.timer, self.key = timer, key

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.timer.add(self.key, time.time() - self.t0)
        return False


# ------------------------------------------------------- §7.4 convergence gate


def _rel_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.norm(A - B, axis=1) / np.maximum(np.linalg.norm(B, axis=1), 1e-30)


def _gate_features(cfg: Config, sim: Simulator, probes: np.ndarray, grid,
                   filter_on: bool = True):
    res = sim.run(probes, grid=grid, refine=False, want_device_traces=True,
                  filter_on=filter_on)
    plan = features.make_plan(cfg, grid, alpha=0.0, deconvolve=filter_on)
    F = features.band_features(features.spectra(res.traces, plan), plan)
    F = np.asarray(F.get() if hasattr(F, "get") else F)
    diag = {k: v.copy() for k, v in res.diag.items()}
    del res
    return F, diag


def convergence_gate(cfg: Config, sim: Simulator, log) -> Dict:
    """Blocking dt / aliasing checks, dt_int selection and P_floor (§7.4).

    Amendments A2/A3 (SPEC §19) refine how the two checks are evaluated:
      * probes that §10 would quarantine (no spikes) carry a meaningless
        spectral *shape* and are excluded;
      * probes whose dt=0.001-vs-0.0005 baseline already exceeds the criterion
        sit on a bifurcation surface -- no step size converges there -- and are
        excluded and reported;
      * acceptance uses the median over the remainder, and `dt_int` is selected
        as the coarsest ladder step meeting the criterion with margin;
      * the aliasing check compares filtered against naively decimated output
        on a common, much finer output grid, which is what "aliasing is
        controlled" actually means.
    """
    num = cfg.numerics
    probes = np.vstack([canonical_array(),
                        cfg.space.to_native(sampling.lhs(num.gate_n_lhs, cfg.schedule.seed + 7))])
    ladder = tuple(num.gate_dts)
    dt_base = num.gate_dt_ref / 2.0
    log(f"convergence gate: {probes.shape[0]} probes x {len(ladder) + 1} step sizes")

    feats: Dict[float, np.ndarray] = {}
    diags: Dict[float, Dict] = {}
    for dt in ladder + (dt_base,):
        g = grid_for_dt(cfg.grid, dt)
        feats[dt], diags[dt] = _gate_features(cfg, sim, probes, g)
        log(f"  dt={dt:<7g} L={g.L:<4d} n_out={g.n_out} "
            f"mean spikes={diags[dt]['n_spikes'].mean():.1f} "
            f"max local err={np.nanmax(diags[dt]['max_local_error']):.2e}")

    ref = num.gate_dt_ref
    spiking = diags[ref]["n_spikes"] > 0
    baseline = _rel_l2(feats[dt_base], feats[ref])
    sensitive = baseline > num.gate_rel_l2
    use = spiking & ~sensitive
    log(f"  probes: {int(spiking.sum())} spiking, "
        f"{int((spiking & sensitive).sum())} sensitivity-limited (excluded), "
        f"{int(use.sum())} used")

    stats_by_dt: Dict[str, Dict[str, float]] = {}
    for dt in ladder:
        r = _rel_l2(feats[dt], feats[ref])[use]
        stats_by_dt[f"{dt:g}"] = {
            "median": float(np.median(r)) if r.size else float("nan"),
            "p90": float(np.percentile(r, 90)) if r.size else float("nan"),
            "max": float(r.max()) if r.size else float("nan"),
            "n_over_criterion": int((r > num.gate_rel_l2).sum()),
            "n_used": int(r.size),
        }
        st = stats_by_dt[f"{dt:g}"]
        log(f"  dt={dt:<7g} rel L2 vs dt={ref:g}: median {st['median']:.5f} "
            f"p90 {st['p90']:.4f} max {st['max']:.4f} "
            f"({st['n_over_criterion']}/{st['n_used']} over {num.gate_rel_l2:g})")

    target = num.gate_rel_l2 * num.gate_margin
    eligible = [dt for dt in sorted(ladder, reverse=True)
                if stats_by_dt[f"{dt:g}"]["median"] <= target]
    dt_sel = eligible[0] if eligible else min(ladder)
    check_dt = stats_by_dt[f"{dt_sel:g}"]["median"] <= num.gate_rel_l2
    log(f"  selected dt_int = {dt_sel:g} ms (coarsest with median <= "
        f"{target:g}; spec default {num.gate_dt_accept:g} has median "
        f"{stats_by_dt[f'{num.gate_dt_accept:g}']['median']:.5f})")

    grid_sel = grid_for_dt(cfg.grid, dt_sel)

    # ---- aliasing check: filtered vs naive decimation, common fine reference
    L_fine = max(2, int(round(grid_sel.L / num.gate_fine_out_divisor)))
    g_fine = grid_for_dt(cfg.grid, dt_sel, dt_out=dt_sel * L_fine)
    F_fine, _ = _gate_features(cfg, sim, probes, g_fine)
    F_on = feats[dt_sel] if dt_sel in feats else _gate_features(cfg, sim, probes, grid_sel)[0]
    F_off, _ = _gate_features(cfg, sim, probes, grid_sel, filter_on=False)
    r_on = float(np.median(_rel_l2(F_on, F_fine)[use]))
    r_off = float(np.median(_rel_l2(F_off, F_fine)[use]))
    ratio = r_off / max(r_on, 1e-12)
    check_alias = ratio >= num.gate_alias_ratio
    log(f"  aliasing: vs dt_out={g_fine.dt_out:g} ms reference, filtered "
        f"{r_on:.4f} vs naive {r_off:.4f} -> {ratio:.1f}x "
        f"({'pass' if check_alias else 'FAIL'}, need "
        f">= {num.gate_alias_ratio:g}x)")

    # ---- reported for transparency: the literal dt_out-halving stability
    g_half = grid_for_dt(cfg.grid, dt_sel, dt_out=grid_sel.dt_out / 2)
    F_half, _ = _gate_features(cfg, sim, probes, g_half)
    r_half = _rel_l2(F_half, F_on)[use]
    log(f"  dt_out {grid_sel.dt_out:g} -> {g_half.dt_out:g} ms: median "
        f"{np.median(r_half):.5f} max {r_half.max():.4f}")

    # ---- P_floor from numerically silent references (§10)
    silent_probe = np.array([[a, b, -65.0, d]
                             for a in (0.005, 0.02, 0.1)
                             for b in (-0.10, -0.05, 0.0)
                             for d in (0.0, 6.0)], dtype=np.float64)
    res = sim.run(silent_probe, grid=grid_sel, want_device_traces=True)
    plan0 = features.make_plan(cfg, grid_sel, alpha=0.0)
    acp = features.ac_power(features.spectra(res.traces, plan0), plan0)
    acp = np.asarray(acp.get() if hasattr(acp, "get") else acp)
    silent = res.diag["n_spikes"] <= 0
    if silent.any():
        p_num = float(num.p_floor_multiplier * np.median(acp[silent]))
    else:
        p_num = float(num.p_floor_multiplier * acp.min())
        log("  WARNING no silent reference found; numerical floor from min ac_power")
    # physical term (§19 A5): ac_power of an in-band fluctuation of
    # p_floor_rms_mv RMS, measured through the same §9 steps
    amp = num.p_floor_rms_mv * math.sqrt(2.0)
    tt = np.arange(grid_sel.n_out) * grid_sel.dt_out * 1e-3
    probe_sin = (amp * np.sin(2 * np.pi * cfg.features.f_ref * tt)).astype(np.float32)[None, :]
    p_phys = float(np.asarray(features.ac_power(
        features.spectra(probe_sin, plan0), plan0))[0])
    p_floor = max(p_num, p_phys)
    log(f"  P_floor: numerical term {p_num:.3e} "
        f"({int(silent.sum())}/{silent.size} silent references), physical term "
        f"{p_phys:.3e} ({num.p_floor_rms_mv:g} mV RMS) -> P_floor {p_floor:.3e}")
    del res

    passed = bool(check_dt and check_alias)
    log(f"convergence gate: {'PASS' if passed else 'FAIL'}")
    return {
        "probes": int(probes.shape[0]),
        "n_spiking": int(spiking.sum()),
        "n_sensitivity_limited": int((spiking & sensitive).sum()),
        "n_used": int(use.sum()),
        "baseline_dt": dt_base,
        "baseline_median": float(np.median(baseline[spiking])),
        "stats_by_dt": stats_by_dt,
        "dt_selected": float(dt_sel),
        "dt_spec_default": float(num.gate_dt_accept),
        "check_dt": bool(check_dt),
        "alias_reference_dt_out": float(g_fine.dt_out),
        "alias_filtered_median": r_on,
        "alias_naive_median": r_off,
        "alias_ratio": float(ratio),
        "check_alias": bool(check_alias),
        "dt_out_halved_median": float(np.median(r_half)),
        "dt_out_halved_max": float(r_half.max()),
        "passed": passed,
        "p_floor": p_floor,
        "p_floor_numerical": p_num,
        "p_floor_physical": p_phys,
        "p_floor_rms_mv": num.p_floor_rms_mv,
        "silent_references": int(silent.sum()),
        "max_local_error_by_dt": {f"{dt:g}": float(np.nanmax(d["max_local_error"]))
                                  for dt, d in diags.items()},
        "mean_spikes_by_dt": {f"{dt:g}": float(d["n_spikes"].mean())
                              for dt, d in diags.items()},
    }


# ------------------------------------------------------ §11 sampling loop


class Store:
    """Accumulates everything the pipeline needs across iterations."""

    def __init__(self, cfg: Config, keep_traces: bool, keep_spectra: bool):
        self.cfg = cfg
        self.keep_traces = keep_traces
        self.keep_spectra = keep_spectra
        self.x: List[np.ndarray] = []
        self.p: List[np.ndarray] = []
        self.F: List[np.ndarray] = []
        self.mag: List[np.ndarray] = []
        self.traces: List[np.ndarray] = []
        self.acp: List[np.ndarray] = []
        self.diag: List[Dict[str, np.ndarray]] = []
        self.hist: List[np.ndarray] = []
        self.source: List[np.ndarray] = []
        self.iteration: List[np.ndarray] = []
        self.n_refined = 0

    def add(self, *, x, p, F, mag, traces, acp, diag, hist, source, iteration,
            n_refined):
        self.x.append(x)
        self.p.append(p)
        self.F.append(F)
        if self.keep_spectra:
            self.mag.append(mag)
        if self.keep_traces:
            self.traces.append(traces)
        self.acp.append(acp)
        self.diag.append(diag)
        self.hist.append(hist)
        self.source.append(np.asarray(source, dtype=object))
        self.iteration.append(np.full(x.shape[0], iteration, dtype=np.int32))
        self.n_refined += int(n_refined)

    def stacked(self):
        out = {
            "x": np.vstack(self.x),
            "p": np.vstack(self.p),
            "F": np.vstack(self.F),
            "ac_power": np.concatenate(self.acp),
            "source": np.concatenate(self.source),
            "iteration": np.concatenate(self.iteration),
            "isi_hist": np.vstack(self.hist),
        }
        keys = self.diag[0].keys()
        out["diag"] = {k: np.concatenate([d[k] for d in self.diag]) for k in keys}
        out["mag"] = np.vstack(self.mag) if self.keep_spectra else None
        out["traces"] = np.vstack(self.traces) if self.keep_traces else None
        return out


def simulate_batch(cfg: Config, sim: Simulator, x: np.ndarray, plan, log,
                   timer: Timer, keep_traces: bool, keep_spectra: bool):
    """Simulate + featurise one iteration in GPU chunks of `cfg.schedule.chunk`."""
    import cupy as cp

    p = cfg.space.to_native(x)
    n = p.shape[0]
    chunk = cfg.schedule.chunk
    Fs, mags, trs, acps, diags, hists = [], [], [], [], [], []
    n_ref = 0
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        with timer.section("simulate"):
            res = sim.run(p[sl], want_device_traces=True)
        n_ref += res.n_refined
        with timer.section("features"):
            mag = features.spectra(res.traces, plan)
            acp = features.ac_power(mag, plan)
            F = features.band_features(mag, plan)
            Fs.append(cp.asnumpy(F))
            acps.append(cp.asnumpy(acp))
            if keep_spectra:
                mags.append(cp.asnumpy(mag))
            if keep_traces:
                trs.append(cp.asnumpy(res.traces))
            diags.append(res.diag)
            hists.append(res.isi_hist)
            del mag, acp, F, res
            cp.get_default_memory_pool().free_all_blocks()
    keys = diags[0].keys()
    diag = {k: np.concatenate([d[k] for d in diags]) for k in keys}
    return {
        "p": p,
        "F": np.vstack(Fs),
        "mag": np.vstack(mags) if keep_spectra else None,
        "traces": np.vstack(trs) if keep_traces else None,
        "ac_power": np.concatenate(acps),
        "diag": diag,
        "isi_hist": np.vstack(hists),
        "n_refined": n_ref,
    }


def interim_labels(cfg: Config, P50: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Cheap labels for the acquisition entropy term: PCA(50) -> HDBSCAN, no UMAP."""
    n = P50.shape[0]
    m = int(min(cfg.schedule.interim_subsample, n))
    idx = rng.choice(n, size=m, replace=False)
    cl = cfg.clustering
    c = cluster.full_fit(np.ascontiguousarray(P50[idx], dtype=np.float32),
                         min_cluster_size=cl.min_cluster_size,
                         min_samples=cl.min_samples, method=cl.selection_method,
                         prediction_data=False)
    return idx, c.labels_


def active_learning(cfg: Config, sim: Simulator, log, timer: Timer,
                    keep_traces: bool, keep_spectra: bool) -> Tuple[Store, List[Dict]]:
    sch = cfg.schedule
    plan0 = features.make_plan(cfg, cfg.grid, alpha=0.0)
    store = Store(cfg, keep_traces, keep_spectra)
    rng = np.random.default_rng(sch.seed)
    hist: List[Dict] = []

    x, src, seed_ids = sampling.first_iteration(cfg)
    theta_prev: Optional[List[np.ndarray]] = None

    for it in range(1, sch.m + 1):
        t_it = time.time()
        if it > 1:
            n_acq, n_lhs = sch.split(sch.n_iter)
            x_lhs = sampling.lhs(n_lhs, sch.seed + 100 + it)
            with timer.section("gp_fit"):
                gps, weights, pca, theta_prev = fit_gps(cfg, store, theta_prev, log)
            with timer.section("interim_hdbscan"):
                P50 = pca.transform(np.vstack(store.F)).astype(np.float32)
                idx_lab, lab = interim_labels(cfg, P50, rng)
                x_lab = np.vstack(store.x)[idx_lab]
                log(f"  interim labels: {np.unique(lab[lab >= 0]).size} clusters, "
                    f"noise {float((lab < 0).mean()):.3f}")
            with timer.section("acquire"):
                pool = sampling.sobol_pool(sch.pool, sch.seed + 1000 + it)
                picked, adiag = select_batch(gps, weights, pool, x_lab, lab,
                                             n_acq, cfg, rng)
                x_gp = pool[picked]
            for gp in gps:
                gp.free()
            x = np.vstack([x_gp, x_lhs])
            src = np.array(["gp"] * n_acq + ["lhs"] * n_lhs, dtype=object)
            log(f"  acquisition: rank sd {adiag['mean_rank_sd']:.3f} "
                f"grad {adiag['mean_rank_grad']:.3f} "
                f"entropy {adiag['mean_rank_entropy']:.3f} "
                f"(shortlist {adiag['shortlist']})")
        else:
            adiag = {}

        out = simulate_batch(cfg, sim, x, plan0, log, timer, keep_traces, keep_spectra)
        store.add(x=x, p=out["p"], F=out["F"], mag=out["mag"], traces=out["traces"],
                  acp=out["ac_power"], diag=out["diag"], hist=out["isi_hist"],
                  source=src, iteration=it, n_refined=out["n_refined"])
        n_tot = sum(a.shape[0] for a in store.x)
        log(f"iteration {it}/{sch.m}: n={x.shape[0]} (total {n_tot}) "
            f"refined={out['n_refined']} "
            f"spikes/sample={out['diag']['n_spikes'].mean():.1f} "
            f"zero-spike={float((out['diag']['n_spikes'] <= 0).mean()):.3f} "
            f"[{time.time() - t_it:.1f}s]")
        hist.append({"iteration": it, "n": int(x.shape[0]),
                     "n_refined": int(out["n_refined"]),
                     "mean_spikes": float(out["diag"]["n_spikes"].mean()),
                     "acquisition": adiag})
    return store, hist, seed_ids


def fit_gps(cfg: Config, store: Store, theta_prev, log):
    """Six independent ARD-RBF GPs on the first K standardised PCs (§11.2)."""
    sch = cfg.schedule
    X = np.vstack(store.x)
    F = np.vstack(store.F)
    scores, pca = cluster.fit_pca(F, cfg.clustering.pca_components, seed=sch.seed)
    K = int(min(sch.k_pc, scores.shape[1]))
    Y = scores[:, :K].astype(np.float64)
    Y = (Y - Y.mean(axis=0)) / np.maximum(Y.std(axis=0), 1e-12)

    # training subset: all of the last two iterations + a uniform draw from earlier
    n = X.shape[0]
    its = np.concatenate(store.iteration)
    recent_cut = sorted(set(its.tolist()))[-2:]
    recent = np.where(np.isin(its, recent_cut))[0]
    older = np.where(~np.isin(its, recent_cut))[0]
    cap = sch.n_gp
    if recent.size >= cap:
        sel = np.random.default_rng(sch.seed).choice(recent, cap, replace=False)
    else:
        extra = min(older.size, cap - recent.size)
        rng = np.random.default_rng(sch.seed)
        sel = np.concatenate([recent, rng.choice(older, extra, replace=False)]) \
            if extra > 0 else recent
    sel = np.sort(sel)

    gps, thetas = [], []
    for k in range(K):
        gp = ARDGP(use_gpu=True, dtype=np.float32, maxiter=60)
        t0 = theta_prev[k] if theta_prev is not None and k < len(theta_prev) else None
        gp.fit(X[sel], Y[sel, k], theta0=t0)
        gps.append(gp)
        thetas.append(gp.theta)
    w = pca.explained_variance_ratio_[:K]
    w = w / w.sum()
    log(f"  GP fit on {sel.size} points; lengthscales (PC1) "
        f"{np.round(gps[0].lengthscales, 3)}; PC weights {np.round(w, 3)}")
    return gps, w, pca, thetas


# ------------------------------------------------- §12 clustering + sweep


def stage2_sweep(cfg: Config, S: Dict, active: np.ndarray, winner, log, timer):
    """Spectral tilt at the stage-1 winner's clustering parameters (§12.4)."""
    rows = []
    if S["mag"] is None:
        log("  stage 2 skipped: magnitude spectra not retained")
        return rows
    for alpha in (0.5, 1.0):
        t0 = time.time()
        plan = features.make_plan(cfg, cfg.grid, alpha=alpha)
        F = features.band_features(S["mag"][active], plan)
        P50, _, _ = cluster.fit_pca_adaptive(
            np.asarray(F), cfg.clustering.pca_components,
            cfg.clustering.pca_min_evr, seed=cfg.schedule.seed)
        E, _ = cluster.umap_embed(P50, n_components=cfg.clustering.umap_components,
                                  n_neighbors=winner.n_neighbors,
                                  min_dist=cfg.clustering.umap_min_dist,
                                  n_epochs=cfg.clustering.umap_epochs,
                                  random_state=None)
        slt, mst = cluster.build_tree(E, winner.min_samples,
                                      mcs_hint=winner.min_cluster_size)
        labels, _ = cluster.labels_from_tree(E, slt, winner.min_cluster_size,
                                             winner.method)
        sc, rv, nf, ncl = cluster.score_labelling(mst, labels, cfg)
        dt = time.time() - t0
        rows.append(cluster.SweepRow(2, alpha, winner.n_neighbors,
                                     winner.min_cluster_size, winner.min_samples,
                                     winner.method, ncl, nf, rv, sc, dt))
        log(f"  [stage2] alpha={alpha}: k={ncl} noise={nf:.3f} rv={rv:.4f} "
            f"score={sc:.4f} ({dt:.1f}s)")
        del E, slt, mst, P50, F
    return rows


def default_row(cfg: Config) -> cluster.SweepRow:
    cl = cfg.clustering
    return cluster.SweepRow(0, cfg.features.alpha, cl.umap_n_neighbors,
                            cl.min_cluster_size, cl.min_samples,
                            cl.selection_method, 0, float("nan"), float("nan"),
                            float("nan"), 0.0)


# ------------------------------------------------------------ §16 outputs


def representatives(cfg: Config, E10: np.ndarray, labels: np.ndarray,
                    probs: np.ndarray, active_idx: np.ndarray) -> Dict[int, List[int]]:
    """3 points nearest the centroid in the 10-D clustering embedding (§16.1)."""
    out: Dict[int, List[int]] = {}
    for cid in np.unique(labels[labels >= 0]):
        m = np.where(labels == cid)[0]
        c = E10[m].mean(axis=0)
        d = np.linalg.norm(E10[m] - c, axis=1)
        order = np.lexsort((-probs[m], d))
        out[int(cid)] = [int(active_idx[m[j]]) for j in order[:cfg.clustering.n_representatives]]
    return out


def write_report(path: str, text: str, log):
    with open(path, "w") as fh:
        fh.write(text)
    log(f"  wrote {os.path.basename(path)}")


# ------------------------------------------------------------------- run


def run(cfg: Config) -> Dict:
    import cupy as cp

    t_start = time.time()
    outdir = os.path.abspath(cfg.outdir)
    os.makedirs(outdir, exist_ok=True)
    log = Log(os.path.join(outdir, "run.log"))
    timer = Timer()
    backend = plots.setup_backend(cfg.show)
    log(f"izhiclust: N={cfg.schedule.n_total}, m={cfg.schedule.m}, "
        f"matplotlib backend {backend}")

    sim = Simulator(cfg)
    with timer.section("gate"):
        gate = convergence_gate(cfg, sim, log)
    if not gate["passed"]:
        log("ABORT: convergence gate failed (§7.4). If the aliasing check is the "
            "one that failed, the remedy is a longer boxcar cascade, not a "
            "smaller dt_int.")
        raise SystemExit(2)
    if gate["dt_selected"] != cfg.grid.dt_int:
        cfg = replace(cfg, grid=grid_for_dt(cfg.grid, gate["dt_selected"]))
        sim.cfg = cfg
        log(f"production grid: dt_int={cfg.grid.dt_int:g} ms, L={cfg.grid.L}, "
            f"dt_out={cfg.grid.dt_out:g} ms, n_steps={cfg.grid.n_steps}, "
            f"n_out={cfg.grid.n_out}")

    keep_traces = not cfg.compact_csv
    keep_spectra = True
    with timer.section("sampling_loop"):
        store, hist, seed_ids = active_learning(cfg, sim, log, timer,
                                                keep_traces, keep_spectra)
    S = store.stacked()
    N = S["x"].shape[0]
    hd = host_diagnostics(S["diag"], S["isi_hist"], cfg)
    log(f"simulated N={N}; refined at dt/4: {store.n_refined}")

    # ---------------------------------------------------------- quarantine
    qmask, xmask = cluster.quarantine_masks(S["diag"], S["ac_power"], gate["p_floor"])
    active = ~(qmask | xmask)
    active_idx = np.where(active)[0]
    log(f"quarantine: Z={int(qmask.sum())} (silent), X={int(xmask.sum())} "
        f"(non-finite), clustering on {int(active.sum())}")
    zero_spike = int((S['diag']['n_spikes'] <= 0).sum())
    log(f"  zero-spike samples {zero_spike}, of which "
        f"{zero_spike - int(qmask.sum())} retained as subthreshold/resonant")

    # ----------------------------------------------------------- PCA(50)
    with timer.section("pca"):
        plan0 = features.make_plan(cfg, cfg.grid, alpha=0.0)
        F0 = S["F"][active]
        P50, pca, evr_curve = cluster.fit_pca_adaptive(
            F0, cfg.clustering.pca_components, cfg.clustering.pca_min_evr,
            seed=cfg.schedule.seed)
    evr = np.cumsum(pca.explained_variance_ratio_)
    n_at_50 = min(cfg.clustering.pca_components, evr_curve.size)
    log(f"PCA: {P50.shape[1]} components, cumulative EVR {evr[-1]:.5f} "
        f"(EVR at {n_at_50} components {evr_curve[n_at_50 - 1]:.5f}; "
        f"criterion >= {cfg.clustering.pca_min_evr})")
    if evr_curve[n_at_50 - 1] < cfg.clustering.pca_min_evr:
        log(f"  §12.1/§17.10: 50 components fall short, so n_components was "
            f"raised to {P50.shape[1]} rather than relaxing the check")

    # ------------------------------------------------------------- sweep
    sweep_rows: List[cluster.SweepRow] = []
    if cfg.sweep:
        with timer.section("sweep_stage1"):
            rows1, aux = cluster.sweep_stage1(P50, cfg, log=log)
        sweep_rows += rows1
        winner = cluster.pick_winner(rows1)
        log(f"stage-1 winner: nn={winner.n_neighbors} mcs={winner.min_cluster_size} "
            f"ms={winner.min_samples} {winner.method} (score {winner.score:.4f})")
        sweep_timings = aux["timings"]
        n_unseeded = len(cfg.clustering.sweep_neighbors)
        t_unseeded = sweep_timings["umap"] / max(n_unseeded, 1)
        t_tree_mean = sweep_timings["trees"] / max(
            n_unseeded * len(cfg.clustering.sweep_fast_min_samples
                             if cfg.sweep_fast else cfg.clustering.sweep_min_samples), 1)
        t_extract_mean = sweep_timings["extract"] / max(len(rows1), 1)
        del aux
        if cfg.stage2:
            with timer.section("sweep_stage2"):
                rows2 = stage2_sweep(cfg, S, active, winner, log, timer)
            sweep_rows += rows2
            best = cluster.pick_winner([winner] + rows2)
            if best is not winner:
                log(f"stage-2 winner: alpha={best.alpha} (score {best.score:.4f})")
            winner = best
    else:
        winner = default_row(cfg)
        log(f"sweep skipped; using spec defaults nn={winner.n_neighbors} "
            f"mcs={winner.min_cluster_size} ms={winner.min_samples} {winner.method} "
            f"alpha={winner.alpha}")

    # ------------------------------------------------- final seeded fits
    alpha = winner.alpha
    if alpha != 0.0:
        plan_final = features.make_plan(cfg, cfg.grid, alpha=alpha)
        F_final_all = np.asarray(features.band_features(S["mag"], plan_final))
        S["F"] = F_final_all
        P50, pca, evr_curve = cluster.fit_pca_adaptive(
            S["F"][active], cfg.clustering.pca_components,
            cfg.clustering.pca_min_evr, seed=cfg.schedule.seed)
        evr = np.cumsum(pca.explained_variance_ratio_)
    else:
        plan_final = plan0

    with timer.section("umap10_seeded"):
        t0 = time.time()
        E10, _ = cluster.umap_embed(P50, n_components=cfg.clustering.umap_components,
                                    n_neighbors=winner.n_neighbors,
                                    min_dist=cfg.clustering.umap_min_dist,
                                    n_epochs=cfg.clustering.umap_epochs,
                                    random_state=cfg.clustering.random_state)
        t_seeded = time.time() - t0
    log(f"final seeded UMAP(10): {t_seeded:.1f}s")

    with timer.section("hdbscan_final"):
        clusterer = cluster.full_fit(E10, min_cluster_size=winner.min_cluster_size,
                                     min_samples=winner.min_samples,
                                     method=winner.method, prediction_data=True)
        labels = clusterer.labels_.copy()
        probs = clusterer.probabilities_.copy()
        mst = clusterer._min_spanning_tree
        slt = clusterer._single_linkage_tree
        other = "eom" if winner.method == "leaf" else "leaf"
        labels_other, _ = cluster.labels_from_tree(E10, slt,
                                                   winner.min_cluster_size, other)
    letters = cluster.assign_letters(labels)
    n_clusters = len(letters)
    noise_frac = float((labels < 0).mean())
    rv_total = cluster.relative_validity(mst, labels)
    rv_contrib = cluster.relative_validity_contributions(mst, labels)
    log(f"final clustering: {n_clusters} clusters ({winner.method}), "
        f"noise {noise_frac:.3f}, relative_validity {rv_total:.4f}")
    log(f"  coarse ({other}): {np.unique(labels_other[labels_other >= 0]).size} clusters")

    with timer.section("umap2_seeded"):
        E2, reducer2 = cluster.umap_embed(P50, n_components=cfg.clustering.vis_components,
                                   n_neighbors=winner.n_neighbors,
                                   min_dist=cfg.clustering.vis_min_dist,
                                   n_epochs=cfg.clustering.umap_epochs,
                                   random_state=cfg.clustering.random_state)

    # measured single- vs multi-threaded UMAP ratio and tree/extract ratio
    # (§12.3 and §15 both rest on these estimates; record the measurements)
    measured = {"umap_seeded_seconds": t_seeded}
    if cfg.sweep:
        measured.update({
            "umap_unseeded_seconds": t_unseeded,
            "umap_seeded_over_unseeded": t_seeded / max(t_unseeded, 1e-9),
            "hdbscan_tree_build_seconds": t_tree_mean,
            "hdbscan_extract_seconds": t_extract_mean,
            "tree_over_extract": t_tree_mean / max(t_extract_mean, 1e-9),
        })
        log(f"measured: seeded UMAP {t_seeded:.1f}s vs unseeded {t_unseeded:.1f}s "
            f"(ratio {measured['umap_seeded_over_unseeded']:.2f}x, §15 assumed ~4x); "
            f"tree {t_tree_mean:.1f}s vs extract+score {t_extract_mean:.2f}s "
            f"(ratio {measured['tree_over_extract']:.0f}x)")

    # ----------------------------------------------------- §13 validation
    rng = np.random.default_rng(cfg.schedule.seed)
    with timer.section("stability"):
        stab = validate.stability(P50, labels, cfg, n_neighbors=winner.n_neighbors,
                                  min_cluster_size=winner.min_cluster_size,
                                  min_samples=winner.min_samples,
                                  method=winner.method, rng=rng, log=log)
    log(f"stability: mean pairwise ARI {stab.mean_ari:.3f} over "
        f"{stab.n_fits} embeddings of {stab.n_subsample} points ({stab.seconds:.1f}s)")

    Xn_active = S["x"][active]
    with timer.section("contiguity"):
        thr = validate.global_knn_threshold(Xn_active, cfg.clustering.contiguity_knn,
                                            cfg.clustering.contiguity_percentile)
        cont = validate.contiguity(Xn_active, S["p"][active], labels, cfg, thr,
                                   rng=rng)
    n_multi = sum(1 for c in cont.values() if c.multimodal)
    log(f"contiguity: threshold {thr:.4f}, {n_multi}/{n_clusters} clusters MULTIMODAL")


    # ------------------------------------------- §13.1 canonical placement
    pos_in_active = -np.ones(N, dtype=np.int64)
    pos_in_active[active_idx] = np.arange(active_idx.size)
    seed_ids = [int(s) for s in seed_ids]
    placements = validate.place_canonical(
        clusterer, E10, seed_ids, pos_in_active, labels, probs, letters,
        (qmask | xmask)[seed_ids], S["p"][seed_ids])
    for p in placements:
        log(f"  {p.label:3s} -> {p.letter:4s} (p={p.probability:.3f}, "
            f"approx_predict {p.predicted_letter}/{p.predicted_prob:.3f})"
            + ("  [QUARANTINED]" if p.quarantined else ""))

    # -------------------------------------------------- §14 cluster stats
    from sklearn.neighbors import NearestNeighbors
    nnq = NearestNeighbors(n_neighbors=2).fit(E10)
    _, nbr = nnq.kneighbors(E10)
    nn_is_noise = labels[nbr[:, 1]] < 0

    letter_order = sorted(letters.items(), key=lambda kv: (len(kv[1]), kv[1]))
    cluster_rows: List[Dict] = []
    corr_frames: Dict[str, Tuple] = {}
    names: Dict[int, str] = {}
    unstable: Dict[int, bool] = {}
    multimodal: Dict[int, bool] = {}
    nm = cfg.naming

    diag_keys = ("rate", "cv_isi", "burst_index", "spikes_per_burst", "adapt_index")
    for cid, letter in letter_order:
        m = labels == cid
        idx_full = active_idx[m]
        P_all = S["p"][idx_full]
        src = S["source"][idx_full]
        lhs_mask = np.isin(src, ("lhs", "seed"))
        P_lhs = P_all[lhs_mask]
        med = {k: float(np.nanmedian(hd[k][idx_full])) for k in diag_keys}
        med["ac_power"] = float(np.nanmedian(S["ac_power"][idx_full]))
        med["n_spikes"] = float(np.nanmedian(S["diag"]["n_spikes"][idx_full]))
        frac_bursting = float((hd["burst_index"][idx_full] > nm.burst_index).mean())
        frac_irregular = float((hd["cv_isi"][idx_full] > nm.cv_irregular).mean())
        med["frac_bursting"] = frac_bursting
        med["mean_v"] = float(np.nanmedian(S["diag"]["mean_v"][idx_full]))
        med["sd_v"] = float(np.nanmedian(S["diag"]["sd_v"][idx_full]))
        name = characterize.name_from_medians(med, cfg, gate["p_floor"])
        names[cid] = name

        all_df = stats.triangular_frame(P_all)
        enough = P_lhs.shape[0] >= nm.min_lhs_for_unbiased
        lhs_df = stats.triangular_frame(P_lhs) if enough else None
        flag, dmax = (stats.compare_matrices(all_df, lhs_df, nm.corr_disagreement)
                      if lhs_df is not None else (False, float("nan")))
        ipw_df = None
        if cfg.ipw_stats and P_all.shape[0] >= 10:
            w = stats.ipw_weights(S["x"][idx_full])
            ipw_df = stats.weighted_triangular(P_all, w)
        corr_frames[letter] = (all_df, lhs_df, ipw_df)

        near_floor = bool(med["rate"] < nm.rate_floor and
                          med["ac_power"] < nm.near_floor_factor * gate["p_floor"])
        below_rate_floor = bool(med["rate"] < nm.rate_floor and med["n_spikes"] > 0)
        pers = stab.persistence.get(int(cid), float("nan"))
        unstable[cid] = bool(np.isfinite(pers) and pers < cfg.clustering.persistence_floor)
        co = cont[int(cid)]
        multimodal[cid] = bool(co.multimodal)

        row = {
            "letter": letter, "cluster_id": int(cid), "name": name,
            "size": int(m.sum()), "lhs_size": int(P_lhs.shape[0]),
            "noise_adjacency": float(nn_is_noise[m].mean()),
            "mean_probability": float(probs[m].mean()),
            "relative_validity_contribution": float(rv_contrib[cid])
            if cid < len(rv_contrib) else float("nan"),
            "persistence": float(pers), "unstable": unstable[cid],
            "n_param_components": int(co.n_components),
            "multimodal": multimodal[cid],
            "corr_disagreement": bool(flag), "max_abs_dr": float(dmax),
            "largest_component_fraction": float(co.largest_fraction),
            "n_param_components_total": int(co.n_components_total),
            "near_floor": near_floor,
            "below_rate_floor": below_rate_floor,
            "frac_bursting": frac_bursting,
            "frac_irregular": frac_irregular,
            "mixed": bool(0.2 <= frac_bursting <= 0.8),
            "median_n_spikes": med["n_spikes"],
            "median_ac_power_over_p_floor": float(med["ac_power"] / gate["p_floor"]),
            "lhs_matrix": "computed" if lhs_df is not None else "insufficient-sample",
        }
        for j, cn in enumerate(stats.COEFFS):
            row[f"mean_{cn}"] = float(P_all[:, j].mean())
            row[f"sd_{cn}"] = float(P_all[:, j].std(ddof=1)) if P_all.shape[0] > 1 else float("nan")
            row[f"mean_{cn}_lhs"] = float(P_lhs[:, j].mean()) if P_lhs.shape[0] else float("nan")
        row["sd_log10_a"] = float(np.std(np.log10(P_all[:, 0]), ddof=1)) \
            if P_all.shape[0] > 1 else float("nan")
        for k in diag_keys:
            row[f"median_{k}"] = med[k]
        row["median_ac_power"] = med["ac_power"]
        row["median_mean_v"] = med["mean_v"]
        row["median_sd_v"] = med["sd_v"]
        row.update(stats.corr_vector(P_all))
        cluster_rows.append(row)
        log("  " + characterize.describe(letter, med, int(m.sum()), cfg, gate["p_floor"])
            + (" [UNSTABLE]" if unstable[cid] else "")
            + (f" [MULTIMODAL {co.n_components} comp >=5% of "
               f"{co.n_components_total}, largest {co.largest_fraction:.0%}]"
               if multimodal[cid] else "")
            + (" [NEAR-FLOOR]" if near_floor else "")
            + (" [BELOW-RATE-FLOOR]" if below_rate_floor else ""))

    # --------------------------------------------- §16 representatives
    reps = representatives(cfg, E10, labels, probs, active_idx)
    rep_ids = [sid for cid, _ in letter_order for sid in reps[int(cid)]]
    rep_tags = [f"{letters[int(cid)]}{k + 1}"
                for cid, _ in letter_order for k in range(len(reps[int(cid)]))]
    with timer.section("representatives"):
        rp = S["p"][rep_ids]
        full_n = cfg.grid.n_out_total
        res_disp = sim.run(rp, filter_on=False, store_u=True, refine=True,
                           out_start=0, n_out=full_n)
        if S["mag"] is not None:
            rep_mag = S["mag"][rep_ids]
        else:
            res_f = sim.run(rp, want_device_traces=True)
            rep_mag = np.asarray(cp.asnumpy(features.spectra(res_f.traces, plan_final)))
            del res_f
    log(f"re-simulated {len(rep_ids)} representatives with the output filter disabled")

    # ------------------------------------------------------- §16.1 figures
    with timer.section("figures"):
        flags = {"unstable": unstable, "multimodal": multimodal}
        # Quarantined samples are excluded from the fits (§10); they are mapped
        # into the visualisation embedding with umap.transform so that `Z` can
        # be plotted separately, as §16.1 asks.
        E2_plot, lab_plot, q_plot = E2, labels, np.zeros(E2.shape[0], dtype=bool)
        if qmask.any():
            try:
                Pq = pca.transform(S["F"][qmask]).astype(np.float32)
                E2q = np.asarray(reducer2.transform(Pq), dtype=np.float32)
                E2_plot = np.vstack([E2, E2q])
                lab_plot = np.concatenate([labels, -np.ones(E2q.shape[0], dtype=labels.dtype)])
                q_plot = np.concatenate([q_plot, np.ones(E2q.shape[0], dtype=bool)])
            except Exception as exc:
                log(f"  WARNING could not project quarantined points: {exc}")
        placement_xy = [(p.label, p.letter, *E2[pos_in_active[p.sample_id]])
                        for p in placements if pos_in_active[p.sample_id] >= 0]
        plots.scatter_2d(os.path.join(outdir, "clusters_umap2d.png"), E2_plot,
                         lab_plot, letters, q_plot, placement_xy, flags,
                         names, show=cfg.show)
        t_full = cfg.grid.t_full
        f_hz = plan_final.freqs
        for tag, sid, k in zip(rep_tags, rep_ids, range(len(rep_ids))):
            cid = int(full_labels_lookup(labels, pos_in_active, sid))
            a, b, c, d = S["p"][sid]
            title = (f"{tag}  —  {names.get(cid, '')}   "
                     f"a={a:.4g} b={b:.4g} c={c:.4g} d={d:.4g}"
                     f"   rate {hd['rate'][sid]:.0f} Hz, sample #{sid}")
            plots.representative_figure(
                os.path.join(outdir, f"{tag}.png"), t_ms=t_full,
                v=res_disp.traces[k], f_hz=f_hz, mag=rep_mag[k], title=title,
                t_window=(cfg.grid.t_transient, cfg.grid.t_total))
        for cid, letter in letter_order:
            ks = [i for i, t in enumerate(rep_tags) if t.startswith(letter)
                  and t[len(letter):].isdigit()]
            if not ks:
                continue
            labs = [f"{rep_tags[i]}  a={S['p'][rep_ids[i]][0]:.4g} "
                    f"b={S['p'][rep_ids[i]][1]:.4g} c={S['p'][rep_ids[i]][2]:.4g} "
                    f"d={S['p'][rep_ids[i]][3]:.4g}" for i in ks]
            plots.cluster_trace_stack(
                os.path.join(outdir, f"{letter}_traces.png"), t_full,
                [res_disp.traces[i] for i in ks], labs,
                f"{letter} — {names[cid]}  (3 representatives, filter disabled)")
    log(f"figures written to {outdir}")
    return dict(cfg=cfg, log=log, timer=timer, S=S, hd=hd, gate=gate,
                store=store, hist=hist, labels=labels, probs=probs,
                letters=letters, letter_order=letter_order, names=names,
                active_idx=active_idx, active=active, qmask=qmask, xmask=xmask,
                E10=E10, E2=E2, P50=P50, pca=pca, clusterer=clusterer,
                labels_other=labels_other, other=other, winner=winner,
                sweep_rows=sweep_rows, stab=stab, cont=cont,
                placements=placements, cluster_rows=cluster_rows,
                corr_frames=corr_frames, reps=reps, rep_ids=rep_ids,
                rep_tags=rep_tags, res_disp=res_disp, rep_mag=rep_mag,
                rv_total=rv_total, rv_contrib=rv_contrib, noise_frac=noise_frac,
                n_clusters=n_clusters, unstable=unstable, multimodal=multimodal,
                plan_final=plan_final, plan0=plan0, outdir=outdir,
                t_start=t_start, evr=evr, evr_curve=evr_curve, sim=sim,
                pos_in_active=pos_in_active,
                seed_ids=seed_ids, backend=backend, t_seeded=t_seeded,
                measured=measured, N=N)


def full_labels_lookup(labels, pos_in_active, sample_id):
    j = pos_in_active[sample_id]
    return labels[j] if j >= 0 else -1


def _count_lines(path: str) -> int:
    n = 0
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rb") as fh:
        while True:
            buf = fh.read(1 << 22)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def _first_last_sample_id(path: str) -> Tuple[Optional[int], Optional[int]]:
    """(first, last) value of the `sample_id` column, located from the header."""
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    first = last = ""
    with opener(path, "rt") as fh:
        header = next(fh).rstrip("\n").split(",")
        try:
            col = header.index("sample_id")
        except ValueError:
            return None, None
        for line in fh:
            if line.strip():
                if not first:
                    first = line
                last = line
    try:
        return (int(first.split(",")[col]), int(last.split(",")[col]))
    except Exception:
        return None, None


def write_outputs(ctx: Dict) -> Dict:
    cfg: Config = ctx["cfg"]
    log = ctx["log"]
    timer: Timer = ctx["timer"]
    S = ctx["S"]
    hd = ctx["hd"]
    outdir = ctx["outdir"]
    N = S["x"].shape[0]
    active_idx = ctx["active_idx"]
    labels, probs = ctx["labels"], ctx["probs"]
    letters = ctx["letters"]
    winner = ctx["winner"]

    paths = iox.ensure_dirs(cfg)
    datadir = paths["data"]
    sample_ids = np.arange(N, dtype=np.int64)

    full_lab = -np.ones(N, dtype=np.int64)
    full_lab[active_idx] = labels
    full_other = -np.ones(N, dtype=np.int64)
    full_other[active_idx] = ctx["labels_other"]
    full_prob = np.zeros(N)
    full_prob[active_idx] = probs
    letter_col = np.array(["noise"] * N, dtype=object)
    for cid, L in letters.items():
        letter_col[active_idx[labels == cid]] = L
    letter_col[ctx["qmask"]] = "Z"
    letter_col[ctx["xmask"]] = "X"
    E10_full = np.full((N, ctx["E10"].shape[1]), np.nan, dtype=np.float32)
    E10_full[active_idx] = ctx["E10"]
    E2_full = np.full((N, 2), np.nan, dtype=np.float32)
    E2_full[active_idx] = ctx["E2"]

    if winner.method == "leaf":
        lab_leaf, lab_eom = full_lab, full_other
    else:
        lab_leaf, lab_eom = full_other, full_lab

    # -------------------------------------------------------- samples.csv
    with timer.section("csv_samples"):
        import pandas as pd
        d = {
            "iteration": S["iteration"], "sample_id": sample_ids,
            "source": S["source"].astype(str),
            "a": S["p"][:, 0], "b": S["p"][:, 1], "c": S["p"][:, 2],
            "d": S["p"][:, 3], "I": np.full(N, cfg.model.I),
            "n_spikes": S["diag"]["n_spikes"].astype(np.int64),
            "rate": hd["rate"], "cv_isi": hd["cv_isi"],
            "burst_index": hd["burst_index"],
            "spikes_per_burst": hd["spikes_per_burst"],
            "adapt_index": hd["adapt_index"],
            "mean_v": S["diag"]["mean_v"], "sd_v": S["diag"]["sd_v"],
            "ac_power": S["ac_power"],
            "max_local_error": S["diag"]["max_local_error"],
            "finite": S["diag"]["finite"].astype(np.int8),
            "quarantine": ctx["qmask"].astype(np.int8),
            "label_leaf": lab_leaf, "label_eom": lab_eom,
            "letter": letter_col.astype(str), "membership_prob": full_prob,
        }
        for j in range(E10_full.shape[1]):
            d[f"umap10_{j}"] = E10_full[:, j]
        for j in range(2):
            d[f"umap2_{j}"] = E2_full[:, j]
        df = pd.DataFrame(d)
        p_samples = os.path.join(datadir, "samples.csv")
        df.to_csv(p_samples, index=False, float_format="%.6g")
        log(f"  wrote samples.csv ({df.shape[0]} x {df.shape[1]})")

        p_bands = os.path.join(datadir, "features_bands.csv")
        iox.write_matrix_csv(p_bands, S["F"], sample_ids,
                             iox.band_columns(ctx["plan_final"].band_centres),
                             "%.6g", log=log)

    # -------------------------------------------------------- bulk files
    bulk = {}
    if not cfg.compact_csv:
        with timer.section("csv_bulk"):
            t_cols = iox.trace_columns(cfg.grid.t_window)
            p_traces = os.path.join(datadir, "traces.csv")
            if cfg.csv_full_trace:
                log("  --csv-full-trace: re-simulating full 2500 ms traces for the CSV")
                full = []
                for s in range(0, N, cfg.schedule.chunk):
                    sl = slice(s, min(s + cfg.schedule.chunk, N))
                    r = sim_run_full(ctx["sim"], cfg, S["p"][sl])
                    full.append(r)
                M = np.vstack(full)
                t_cols = iox.trace_columns(cfg.grid.t_full)
            else:
                M = S["traces"]
            bulk["traces"] = iox.write_matrix_csv(
                p_traces, M, sample_ids, t_cols, iox.TRACE_FMT,
                gzip=cfg.gzip_csv, log=log)
            del M
            p_spec = os.path.join(datadir, "spectra_full.csv")
            bulk["spectra"] = iox.write_matrix_csv(
                p_spec, S["mag"], sample_ids,
                iox.freq_columns(ctx["plan0"].freqs), iox.SPECTRA_FMT,
                gzip=cfg.gzip_csv, log=log)
    else:
        log("  --compact-csv: traces.csv and spectra_full.csv omitted")

    # ------------------------------------------- representative traces
    with timer.section("csv_reps"):
        import pandas as pd
        res_disp = ctx["res_disp"]
        t_full = cfg.grid.t_full
        for k, tag in enumerate(ctx["rep_tags"]):
            pd.DataFrame({"t_ms": t_full, "v": res_disp.traces[k],
                          "u": res_disp.u_traces[k]}).to_csv(
                os.path.join(paths["traces"], f"{tag}.csv"), index=False,
                float_format="%.6g")
        log(f"  wrote {len(ctx['rep_tags'])} representative trace CSVs")

    # ------------------------------------------------ stats / corr / npz
    with timer.section("csv_stats"):
        import pandas as pd
        pd.DataFrame(ctx["cluster_rows"]).to_csv(
            os.path.join(datadir, "cluster_stats.csv"), index=False,
            float_format="%.6g")
        for letter, (all_df, lhs_df, ipw_df) in ctx["corr_frames"].items():
            all_df.to_csv(os.path.join(datadir, f"corr_{letter}_all.csv"),
                          float_format="%.6g")
            if lhs_df is not None:
                lhs_df.to_csv(os.path.join(datadir, f"corr_{letter}_lhs.csv"),
                              float_format="%.6g")
            if ipw_df is not None:
                ipw_df.to_csv(os.path.join(datadir, f"corr_{letter}_ipw.csv"),
                              float_format="%.6g")
        np.savez_compressed(
            os.path.join(datadir, "run_data.npz"),
            params=S["p"].astype(np.float32), x_unit=S["x"].astype(np.float32),
            features=S["F"].astype(np.float32),
            band_centres=ctx["plan_final"].band_centres.astype(np.float32),
            ac_power=S["ac_power"].astype(np.float32),
            isi_hist=S["isi_hist"].astype(np.int32),
            umap10=E10_full, umap2=E2_full,
            label_leaf=lab_leaf.astype(np.int32),
            label_eom=lab_eom.astype(np.int32),
            letter=letter_col.astype(str), membership_prob=full_prob.astype(np.float32),
            iteration=S["iteration"], source=S["source"].astype(str),
            quarantine=ctx["qmask"], nonfinite=ctx["xmask"],
            **{f"diag_{k}": v.astype(np.float32) for k, v in S["diag"].items()},
            **{f"hd_{k}": np.asarray(v, dtype=np.float32) for k, v in hd.items()})
        rows = [dataclasses.asdict(r) for r in ctx["sweep_rows"]]
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(outdir, "sweep_table.csv"),
                                      index=False, float_format="%.6g")
        log("  wrote cluster_stats.csv, corr_*.csv, run_data.npz, sweep_table.csv")

    # ------------------------------------------------------ round trip
    rt = {"rows": 0, "rel_l2_max": float("nan")}
    if not cfg.compact_csv and not cfg.csv_full_trace:
        with timer.section("roundtrip"):
            p_traces = os.path.join(datadir, "traces.csv" + (".gz" if cfg.gzip_csv else ""))
            rt = iox.roundtrip_check(p_traces, S["F"], ctx["plan_final"], cfg,
                                     exclude=ctx["qmask"] | ctx["xmask"], log=log)

    return {"paths": paths, "bulk": bulk, "roundtrip": rt,
            "samples_csv": os.path.join(datadir, "samples.csv"),
            "letter_col": letter_col, "lab_leaf": lab_leaf, "lab_eom": lab_eom}


def sim_run_full(sim, cfg, params):
    r = sim.run(params, filter_on=True, out_start=0, n_out=cfg.grid.n_out_total)
    return r.traces


def acceptance(ctx: Dict, outs: Dict) -> List[Dict]:
    cfg: Config = ctx["cfg"]
    S = ctx["S"]
    N = S["x"].shape[0]
    datadir = outs["paths"]["data"]
    checks: List[Dict] = []

    def add(name, ok, detail):
        checks.append({"criterion": name, "pass": bool(ok), "detail": detail})

    g = ctx["gate"]
    add("convergence gate passes both checks (§7.4)", g["passed"],
        f"dt_int {g['dt_selected']:g} ms, median rel L2 "
        f"{g['stats_by_dt'][f'{g['dt_selected']:g}']['median']:.5f}; alias "
        f"ratio {g['alias_ratio']:.1f}x")
    nx = int(ctx["xmask"].sum())
    add("zero samples labelled X (non-finite after refinement)", nx == 0, f"X = {nx}")
    add("noise fraction <= 25% at the selected config", ctx["noise_frac"] <= 0.25,
        f"noise = {ctx['noise_frac']:.3f}")
    n_leaf = int(np.unique(outs["lab_leaf"][outs["lab_leaf"] >= 0]).size)
    add(">= 10 clusters at cluster_selection_method='leaf'", n_leaf >= 10,
        f"leaf clusters = {n_leaf}")
    add("mean pairwise ARI across 5 stability embeddings >= 0.6",
        ctx["stab"].mean_ari >= 0.6,
        f"mean ARI = {ctx['stab'].mean_ari:.3f} (Monte-Carlo estimate, §13.2)")
    placed = [p for p in ctx["placements"]]
    bad = [p.label for p in placed if p.cluster_id < 0]
    add("all 7 canonical types placed in a non-noise cluster", not bad,
        "all placed" if not bad else f"unplaced: {', '.join(bad)}")

    want = ["samples.csv", "features_bands.csv", "cluster_stats.csv", "run_data.npz"]
    if not cfg.compact_csv:
        suff = ".gz" if cfg.gzip_csv else ""
        want += ["traces.csv" + suff, "spectra_full.csv" + suff]
    missing = [f for f in want
               if not (os.path.exists(os.path.join(datadir, f))
                       and os.path.getsize(os.path.join(datadir, f)) > 0)]
    fig = ["clusters_umap2d.png", "cluster_report.md", "run_manifest.json"]
    missing += [f for f in fig
                if not (os.path.exists(os.path.join(ctx["outdir"], f))
                        and os.path.getsize(os.path.join(ctx["outdir"], f)) > 0)]
    add("every §16 artefact present and non-empty", not missing,
        "all present" if not missing else f"missing: {missing}")

    row_files = ["samples.csv", "features_bands.csv"]
    if not cfg.compact_csv:
        suff = ".gz" if cfg.gzip_csv else ""
        row_files += ["traces.csv" + suff, "spectra_full.csv" + suff]
    rows = {}
    for f in row_files:
        pth = os.path.join(datadir, f)
        if not os.path.exists(pth):
            continue
        rows[f] = _count_lines(pth) - 1
    ok_rows = all(v == N for v in rows.values())
    ids_ok = True
    for f in row_files:
        if f not in rows:
            continue
        fl = _first_last_sample_id(os.path.join(datadir, f))
        if fl != (0, N - 1):
            ids_ok = False
    add("samples/traces/spectra/features all have exactly N rows and identical "
        "sample_id ordering", ok_rows and ids_ok,
        f"rows = {rows}, first/last sample_id consistent = {ids_ok}")

    rt = outs["roundtrip"]
    if not np.isfinite(rt.get("rel_l2_max", np.nan)):
        reason = ("--compact-csv omits traces.csv" if cfg.compact_csv else
                  "--csv-full-trace breaks the row-for-row correspondence"
                  if cfg.csv_full_trace else "no rows available")
        checks.append({"criterion": "round trip: re-reading traces.csv "
                       "reproduces features_bands.csv to 1e-4 relative",
                       "pass": None, "detail": f"not applicable ({reason})"})
        return checks
    add("round trip: re-reading traces.csv reproduces features_bands.csv "
        "to 1e-4 relative (mean over clustered rows, §19 A10)",
        rt["rel_l2_mean"] <= 1e-4,
        f"mean rel L2 = {rt.get('rel_l2_mean', float('nan')):.2e} over "
        f"{rt.get('rows', 0)} clustered rows; p99 "
        f"{rt.get('rel_l2_p99', float('nan')):.2e}, max "
        f"{rt.get('rel_l2_max', float('nan')):.2e} (the tail is set by bands "
        f"straddling the §19 A1 floor, not by precision: it is unchanged at "
        f"%.9g). The {rt.get('rows_excluded', 0)} quarantined rows excluded "
        f"reach {rt.get('excluded_rel_l2_max', float('nan')):.2e} (§19 A7)")
    return checks


def build_report(ctx: Dict, outs: Dict, checks: List[Dict], manifest: Dict) -> str:
    cfg: Config = ctx["cfg"]
    S = ctx["S"]
    N = S["x"].shape[0]
    w = ctx["winner"]
    L: List[str] = []
    A = L.append
    A("# Izhikevich spiking-neuron cluster discovery — run report\n")
    A(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- Samples `N` = {N} ({cfg.schedule.m} iterations), drive `I` = {cfg.model.I}")
    A(f"- Selected config: `n_neighbors={w.n_neighbors}`, "
      f"`min_cluster_size={w.min_cluster_size}`, `min_samples={w.min_samples}`, "
      f"`cluster_selection_method='{w.method}'`, spectral tilt `alpha={w.alpha}`")
    A(f"- Clusters: **{ctx['n_clusters']}** at `{w.method}`, "
      f"{int(np.unique(ctx['labels_other'][ctx['labels_other'] >= 0]).size)} at "
      f"`{ctx['other']}`; noise {ctx['noise_frac']:.1%}; "
      f"`relative_validity_` {ctx['rv_total']:.4f}")
    A(f"- Quarantine `Z` = {int(ctx['qmask'].sum())}, non-finite `X` = "
      f"{int(ctx['xmask'].sum())}, `P_floor` = {ctx['gate']['p_floor']:.3e}")
    A(f"- Mean pairwise ARI over {ctx['stab'].n_fits} embeddings: "
      f"**{ctx['stab'].mean_ari:.3f}**")
    A(f"- PCA: {ctx['P50'].shape[1]} components retaining "
      f"{ctx['evr'][-1]:.4f} of the variance (at 50 components: "
      f"{ctx['evr_curve'][min(50, ctx['evr_curve'].size) - 1]:.4f}; §19 A4)\n")

    A("## 1. Convergence gate (§7.4, amended per §19 A2/A3)\n")
    g = ctx["gate"]
    A(f"Probe set: {g['probes']} ({g['n_spiking']} spiking, "
      f"{g['n_sensitivity_limited']} sensitivity-limited and excluded, "
      f"{g['n_used']} used).  Sensitivity baseline (dt="
      f"{g['baseline_dt']:g} vs {cfg.numerics.gate_dt_ref:g} ms) median "
      f"{g['baseline_median']:.2e}.\n")
    A("| dt (ms) | median rel L2 | p90 | max | probes > 1% | max local error | mean spikes |")
    A("|---|---|---|---|---|---|---|")
    for k, v in g["stats_by_dt"].items():
        A(f"| {k} | {v['median']:.5f} | {v['p90']:.4f} | {v['max']:.4f} | "
          f"{v['n_over_criterion']}/{v['n_used']} | "
          f"{g['max_local_error_by_dt'][k]:.3g} | {g['mean_spikes_by_dt'][k]:.1f} |")
    A("")
    A(f"- selected `dt_int` = **{g['dt_selected']:g} ms** (coarsest ladder step "
      f"with median <= {cfg.numerics.gate_rel_l2 * cfg.numerics.gate_margin:g}; "
      f"the spec default {g['dt_spec_default']:g} ms has median "
      f"{g['stats_by_dt'][f'{g['dt_spec_default']:g}']['median']:.5f})")
    A(f"- dt check: **{'PASS' if g['check_dt'] else 'FAIL'}**")
    A(f"- aliasing: against a `dt_out = {g['alias_reference_dt_out']:g}` ms "
      f"reference, filtered output is {g['alias_filtered_median']:.4f} away "
      f"and naive decimation {g['alias_naive_median']:.4f}, i.e. the §7.2 "
      f"filter buys **{g['alias_ratio']:.1f}x** "
      f"(**{'PASS' if g['check_alias'] else 'FAIL'}**, need "
      f">= {cfg.numerics.gate_alias_ratio:g}x)")
    A(f"- literal `dt_out`-halving stability (reported, not gating): median "
      f"{g['dt_out_halved_median']:.5f}, max {g['dt_out_halved_max']:.4f}")
    A(f"- `P_floor` = {g['p_floor']:.4e} = max(numerical "
      f"{g['p_floor_numerical']:.3e} from {g['silent_references']} silent "
      f"references, physical {g['p_floor_physical']:.3e} for "
      f"{g['p_floor_rms_mv']:g} mV RMS)\n")

    A("## 2. Quarantine `Z` and non-finite `X` (§10)\n")
    q = ctx["qmask"]
    x = ctx["xmask"]
    nz = int(q.sum())
    A(f"- `Z` (no spikes **and** in-band AC power below `P_floor` = "
      f"{ctx['gate']['p_floor']:.3e}): **{nz}** samples "
      f"({nz / max(N, 1):.1%}), held out before PCA/UMAP")
    A(f"- `X` (non-finite after the §7.3 refinement pass): **{int(x.sum())}** "
      f"— this is an error-budget metric and should be zero")
    zero = int((S["diag"]["n_spikes"] <= 0).sum())
    A(f"- zero-spike samples: {zero}; of those, {zero - nz} are **retained** "
      f"(AC power above `P_floor`) rather than quarantined, which is the "
      f"subthreshold/resonant case §10 exists to preserve")
    if nz:
        Pq = S["p"][q]
        A("")
        A("Quarantined parameter region (mean ± SD, and range):\n")
        A("| coefficient | mean | SD | min | max |")
        A("|---|---|---|---|---|")
        for j, cn in enumerate(stats.COEFFS):
            col = Pq[:, j]
            A(f"| {cn} | {col.mean():.4g} | {col.std(ddof=1):.4g} | "
              f"{col.min():.4g} | {col.max():.4g} |")
        A("")
        b_hi = float(cfg.model.b_bifurcation)
        frac_below = float((Pq[:, 1] < b_hi).mean())
        A(f"{frac_below:.1%} of quarantined samples have `b < "
          f"{b_hi:.4f}`, the value below which a rest state exists at "
          f"`I = {cfg.model.I:g}` (§2) — i.e. the quarantine is essentially "
          f"the sub-bifurcation half of the box, as §2 predicts.\n")

    A("## 3. Clusters\n")
    A("Names follow §14.2; flags follow §13.2 (UNSTABLE) and §13.3 (MULTIMODAL).\n")
    for row in ctx["cluster_rows"]:
        cid = row["cluster_id"]
        letter = row["letter"]
        flagtxt = []
        if row["unstable"]:
            flagtxt.append("**UNSTABLE**")
        if row["multimodal"]:
            flagtxt.append(f"**MULTIMODAL** ({row['n_param_components']} components)")
        if row["near_floor"]:
            flagtxt.append("**NEAR-FLOOR**")
        if row["below_rate_floor"]:
            flagtxt.append("**BELOW-RATE-FLOOR**")
        A(f"### {letter} — {row['name']}\n")
        A(f"- size {row['size']} (LHS-only {row['lhs_size']}), "
          f"mean membership prob {row['mean_probability']:.3f}, "
          f"noise adjacency {row['noise_adjacency']:.3f}")
        A(f"- medians: rate {row['median_rate']:.1f} Hz, CV(ISI) "
          f"{row['median_cv_isi']:.3f}, burst index {row['median_burst_index']:.3f}, "
          f"spikes/burst {row['median_spikes_per_burst']:.2f}, adaptation "
          f"{row['median_adapt_index']:+.3f}, mean v {row['median_mean_v']:.1f} mV, "
          f"sd v {row['median_sd_v']:.2f} mV")
        A(f"- mean coefficient vector (all): a={row['mean_a']:.4g}, "
          f"b={row['mean_b']:.4g}, c={row['mean_c']:.4g}, d={row['mean_d']:.4g}")
        A(f"- mean coefficient vector (LHS-only, authoritative): "
          f"a={row['mean_a_lhs']:.4g}, b={row['mean_b_lhs']:.4g}, "
          f"c={row['mean_c_lhs']:.4g}, d={row['mean_d_lhs']:.4g}")
        A(f"- persistence {row['persistence']:.2f}, DBCV contribution "
          f"{row['relative_validity_contribution']:+.4f}"
          + (("  " + " ".join(flagtxt)) if flagtxt else ""))
        if row["below_rate_floor"]:
            A(f"- **BELOW-RATE-FLOOR**: median rate {row['median_rate']:.1f} Hz "
              f"is under §6's 4 Hz spectral-estimate floor "
              f"({row['median_n_spikes']:.0f} spikes in the 2000 ms window), so "
              f"its spectrum rests on few cycles (§19 A9)")
        if row["near_floor"]:
            A(f"- **NEAR-FLOOR**: median `ac_power` is only "
              f"{row['median_ac_power_over_p_floor']:.1f}x `P_floor`, so this "
              f"cluster's spectral shape may be numerical rather than physical "
              f"(§19 A5)")
        if row["multimodal"]:
            co = ctx["cont"][cid]
            A(f"- pooled mean is **not interpretable**: {co.n_components_total} "
              f"parameter-space components, {co.n_components} of them holding "
              f">= 5% of the cluster, largest "
              f"{row['largest_component_fraction']:.0%}")
            if co.sizes:
                A("  component means:")
                for sz, mn in zip(co.sizes, co.means):
                    A(f"    - n={sz}: a={mn[0]:.4g}, b={mn[1]:.4g}, "
                      f"c={mn[2]:.4g}, d={mn[3]:.4g}")
            else:
                A("  no single component reaches 5%, so none is summarised")
        all_df, lhs_df, ipw_df = ctx["corr_frames"][letter]
        A("\n```")
        A("all samples (descriptive):")
        A(stats.render_triangular(all_df))
        A("")
        if lhs_df is not None:
            A("LHS-only (design-unbiased, authoritative):")
            A(stats.render_triangular(lhs_df))
        else:
            A(f"LHS-only: insufficient-sample (n={row['lhs_size']} < "
              f"{cfg.naming.min_lhs_for_unbiased})")
        if ipw_df is not None:
            A("")
            A("inverse-propensity weighted (--ipw-stats):")
            A(stats.render_triangular(ipw_df))
        A("```\n")
        if row["corr_disagreement"]:
            A(f"> Sampling-design flag: all-sample and LHS-only correlations differ "
              f"by up to |dr| = {row['max_abs_dr']:.2f} (> "
              f"{cfg.naming.corr_disagreement}); trust the LHS-only matrix.\n")

    A("## 4. Canonical type placement (S1, §13.1)\n")
    A("| type | a | b | c | d | cluster | p | approx_predict | note |")
    A("|---|---|---|---|---|---|---|---|---|")
    for p in ctx["placements"]:
        note = "quarantined" if p.quarantined else (
            "expected generic tonic (§13.1)" if p.label in ("TC", "RZ") else "")
        A(f"| {p.label} | {p.params[0]:g} | {p.params[1]:g} | {p.params[2]:g} | "
          f"{p.params[3]:g} | {p.letter} | {p.probability:.3f} | "
          f"{p.predicted_letter}/{p.predicted_prob:.3f} | {note} |")
    A("")
    dis = [p.label for p in ctx["placements"]
           if p.letter != p.predicted_letter]
    if dis:
        A(f"`approximate_predict` disagrees with the in-sample label for "
          f"{', '.join(dis)} — it reassigns from the condensed tree's "
          f"exemplars, so a point near a cluster boundary can land either "
          f"side. The in-sample label is the authoritative one.\n")
    A("TC and RZ are expected to land in generic tonic clusters: their published "
      "behaviours need a hyperpolarising step and timed brief stimuli "
      "respectively, neither reachable at constant `I` (§13.1, §18 item 6).\n")

    A("## 5. Stability (S5, §13.2)\n")
    st = ctx["stab"]
    A(f"- {st.n_fits} independent unseeded UMAP embeddings of "
      f"{st.n_subsample} points, {st.seconds:.1f}s")
    A(f"- pairwise ARI: " + ", ".join(f"{v:.3f}" for v in st.pairwise_ari))
    A(f"- mean pairwise ARI **{st.mean_ari:.3f}** "
      f"(threshold 0.6; a run landing at 0.60 ± 0.03 should be re-run)")
    uns = [r['letter'] for r in ctx['cluster_rows'] if r['unstable']]
    A(f"- UNSTABLE clusters (persistence < {cfg.clustering.persistence_floor}): "
      + (", ".join(uns) if uns else "none") + "\n")

    A("## 6. Contiguity (S4, §13.3)\n")
    thr = next(iter(ctx["cont"].values())).threshold if ctx["cont"] else float("nan")
    A(f"- single-linkage cut at the {cfg.clustering.contiguity_percentile:g}th "
      f"percentile of the global k={cfg.clustering.contiguity_knn} NN distance "
      f"= {thr:.4f} (normalised domain)")
    mm = [f"{r['letter']} ({r['n_param_components_total']} components, largest "
          f"{r['largest_component_fraction']:.0%})"
          for r in ctx["cluster_rows"] if r["multimodal"]]
    A(f"- MULTIMODAL clusters: " + (", ".join(mm) if mm else "none"))
    A("- The cut is a *global* density threshold applied to each cluster's own "
      "point set, so small clusters fragment simply because their own points are "
      "sparser than the global k=5 NN scale; the largest-component fraction is "
      "the number to read alongside the flag.")
    nf = [r["letter"] for r in ctx["cluster_rows"] if r["near_floor"]]
    A(f"- NEAR-FLOOR clusters (§19 A5): " + (", ".join(nf) if nf else "none"))
    bf = [r["letter"] for r in ctx["cluster_rows"] if r["below_rate_floor"]]
    A(f"- BELOW-RATE-FLOOR clusters (§19 A9): "
      + (", ".join(bf) if bf else "none") + "\n")

    A(f"## 7. {w.method} → {ctx['other']} containment (§12.4)\n")
    A(f"| {w.method} | size | {ctx['other']} | containment |")
    A("|---|---|---|---|")
    for r in cluster.containment_table(ctx["labels"], ctx["labels_other"]):
        A(f"| {ctx['letters'].get(r['leaf'], r['leaf'])} | {r['size']} | "
          f"{r['eom']} | {r['containment']:.3f} |")
    A("")

    if ctx["sweep_rows"]:
        A("## 8. Selection sweep (§12.4)\n")
        A("| stage | alpha | nn | mcs | ms | method | k | noise | rel. validity | score |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(ctx["sweep_rows"], key=lambda r: -r.score)[:20]:
            A(f"| {r.stage} | {r.alpha:g} | {r.n_neighbors} | {r.min_cluster_size} "
              f"| {r.min_samples} | {r.method} | {r.n_clusters} | "
              f"{r.noise_fraction:.3f} | {r.relative_validity:.4f} | {r.score:.4f} |")
        A("\n(top 20 of "
          f"{len(ctx['sweep_rows'])} configs; full table in `sweep_table.csv`)\n")

    A("## 9. Acceptance criteria (§17)\n")
    A("| criterion | result | detail |")
    A("|---|---|---|")
    for c in checks:
        verdict = "n/a" if c["pass"] is None else ("PASS" if c["pass"] else "FAIL")
        A(f"| {c['criterion']} | {verdict} | {c['detail']} |")
    A("")

    A("## 10. Timings and measured ratios\n")
    A("| stage | seconds |")
    A("|---|---|")
    for k, v in sorted(manifest["timings"].items(), key=lambda kv: -kv[1]):
        A(f"| {k} | {v:.1f} |")
    A(f"| **total** | **{manifest['total_seconds']:.1f}** |")
    A("")
    m = manifest.get("measured", {})
    if m:
        A("Measurements §12.3 and §15 require in place of their estimates:\n")
        for k, v in m.items():
            A(f"- `{k}` = {v:.3f}")
        if "umap_seeded_over_unseeded" in m:
            A(f"\nThe single- vs multi-threaded UMAP ratio §15 assumed to be ~4x "
              f"measures {m['umap_seeded_over_unseeded']:.2f}x here, and the "
              f"tree-build-to-extraction ratio behind §12.4.1's tree reuse "
              f"measures {m.get('tree_over_extract', float('nan')):.0f}x.")
        A("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="izhiclust",
        description="Izhikevich spiking-neuron cluster discovery (see SPEC.md)")
    ap.add_argument("--quick", action="store_true",
                    help="n1=1024, n=512, m=6, --no-sweep (~5 min smoke run)")
    ap.add_argument("--no-sweep", dest="sweep", action="store_false")
    ap.add_argument("--sweep-fast", action="store_true",
                    help="drop min_samples=25 from stage 1")
    ap.add_argument("--no-stage2", dest="stage2", action="store_false")
    ap.add_argument("--no-tree-reuse", dest="tree_reuse", action="store_false")
    ap.add_argument("--compact-csv", action="store_true")
    ap.add_argument("--gzip-csv", action="store_true")
    ap.add_argument("--csv-full-trace", action="store_true")
    ap.add_argument("--ipw-stats", action="store_true")
    ap.add_argument("--spectral-tilt", type=float, default=None,
                    help="fix alpha (0, 0.5 or 1) instead of sweeping it")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--n", type=int, default=None, help="override n per iteration")
    ap.add_argument("--m", type=int, default=None, help="override iteration count")
    a = ap.parse_args(argv)

    cfg = Config(sweep=a.sweep, sweep_fast=a.sweep_fast, stage2=a.stage2,
                 tree_reuse=a.tree_reuse, compact_csv=a.compact_csv,
                 gzip_csv=a.gzip_csv, csv_full_trace=a.csv_full_trace,
                 ipw_stats=a.ipw_stats, outdir=a.outdir, datadir=a.datadir,
                 show=a.show)
    if a.quick:
        cfg = cfg.quickened()
    if a.spectral_tilt is not None:
        cfg = replace(cfg, features=replace(cfg.features, alpha=a.spectral_tilt),
                      stage2=False)
    if a.n is not None:
        cfg = replace(cfg, schedule=replace(cfg.schedule, n_iter=a.n))
    if a.m is not None:
        cfg = replace(cfg, schedule=replace(cfg.schedule, m=a.m))

    ctx = run(cfg)
    cfg = ctx["cfg"]            # the gate may have refined dt_int (§19 A2)
    log = ctx["log"]
    outs = write_outputs(ctx)

    manifest = {
        "spec": "SPEC.md",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": list(sys.argv),
        "config": ctx["cfg"],
        "derived": {
            "n_total": int(ctx["S"]["x"].shape[0]),
            "dt_int": cfg.grid.dt_int, "L": cfg.grid.L,
            "dt_out": cfg.grid.dt_out, "n_out": cfg.grid.n_out,
            "n_steps": cfg.grid.n_steps, "df_hz": cfg.grid.df_hz,
            "band_centres_hz": ctx["plan_final"].band_centres,
            "b_bifurcation": cfg.model.b_bifurcation,
        },
        "versions": iox.package_versions(),
        "matplotlib_backend": ctx["backend"],
        "convergence_gate": ctx["gate"],
        "iterations": ctx["hist"],
        "pca": {"n_components": int(ctx["P50"].shape[1]),
                "cumulative_explained_variance_curve": ctx["evr_curve"],
                "cumulative_at_selected": float(ctx["evr"][-1]),
                "cumulative_at_50": float(ctx["evr_curve"][
                    min(cfg.clustering.pca_components, ctx["evr_curve"].size) - 1]),
                "min_evr_criterion": cfg.clustering.pca_min_evr},
        "selected_config": dataclasses.asdict(ctx["winner"]),
        "clustering": {
            "n_clusters": ctx["n_clusters"],
            "n_clusters_other": int(np.unique(
                ctx["labels_other"][ctx["labels_other"] >= 0]).size),
            "other_method": ctx["other"],
            "noise_fraction": ctx["noise_frac"],
            "relative_validity": ctx["rv_total"],
            "letters": {str(k): v for k, v in ctx["letters"].items()},
            "quarantine_Z": int(ctx["qmask"].sum()),
            "nonfinite_X": int(ctx["xmask"].sum()),
            "n_refined_dt4": int(ctx["store"].n_refined),
        },
        "stability": dataclasses.asdict(ctx["stab"]),
        "contiguity": {ctx["letters"][k]: dataclasses.asdict(v)
                       for k, v in ctx["cont"].items() if k in ctx["letters"]},
        "canonical_placement": [dataclasses.asdict(p) for p in ctx["placements"]],
        "sweep": [dataclasses.asdict(r) for r in ctx["sweep_rows"]],
        "roundtrip": outs["roundtrip"],
        "timings": dict(ctx["timer"].d),
        "measured": ctx["measured"],
        "total_seconds": time.time() - ctx["t_start"],
    }
    # written twice: the artefact-presence criterion needs the files to exist
    rp = os.path.join(ctx["outdir"], "cluster_report.md")
    mp = os.path.join(ctx["outdir"], "run_manifest.json")
    write_report(rp, build_report(ctx, outs, [], manifest), log)
    iox.write_manifest(mp, manifest, log)
    checks = acceptance(ctx, outs)
    manifest["acceptance"] = checks
    report = build_report(ctx, outs, checks, manifest)
    write_report(rp, report, log)
    iox.write_manifest(mp, manifest, log)
    print()
    print(report)
    n_fail = sum(1 for c in checks if c["pass"] is False)
    n_na = sum(1 for c in checks if c["pass"] is None)
    log(f"acceptance: {len(checks) - n_fail - n_na}/{len(checks) - n_na} "
        f"applicable criteria pass" + (f" ({n_na} n/a)" if n_na else ""))
    log(f"total runtime {manifest['total_seconds'] / 60:.1f} min")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
