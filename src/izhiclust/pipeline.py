"""Orchestrator for the Izhikevich cluster-discovery run.

Stages, in order (IMPROVEMENTS.md section numbers):
  §5 convergence gate -> §4 rheobase + §7 Sobol design -> §3.2 ISI features
  -> §3.4 classes -> §8.1 health gates -> v1 §12 PCA/UMAP/HDBSCAN + sweep
  -> §8.4 nameability merge -> §8.2/§8.3 validation -> v1 §14 statistics
  -> figures, CSVs, manifest, report, §9 acceptance.
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

from . import (characterize, cluster, features, io as iox, isi as isim, plots,
               rheobase, sampling, stats, validate)
from .config import Config, canonical_array
from .simulate import Simulator, SpikeTrains, grid_for_dt, host_diagnostics


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


def _rel_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.norm(A - B, axis=1) / np.maximum(np.linalg.norm(B, axis=1), 1e-30)


# --------------------------------------------------------- §5 convergence gate


def _spike_time_error(ref: np.ndarray, got: np.ndarray) -> float:
    """Median |dt| over matched spikes; NaN when the counts differ."""
    if ref.size != got.size:
        return float("nan")
    if ref.size == 0:
        return 0.0
    return float(np.median(np.abs(ref.astype(np.float64) - got.astype(np.float64))))


def _isi_error(ref: np.ndarray, got: np.ndarray) -> float:
    """Median per-ISI error `|dI|` over the intervals the two runs share.

    When the spike counts differ the comparison is truncated to the common
    prefix: §5 criterion 1 tolerates a probe that differs by one spike, so
    criterion 2 has to be evaluable for such a probe.

    This is the form of §5's criterion 2 the pipeline gates on, and the
    deviation is recorded as assumption V2-A12.  Spike detection is
    end-of-step, so each detected time is systematically late by up to `dt`
    and that lateness *accumulates* along the train: the absolute spike-time
    error of the k-th spike in the window is ~k*dt/2, which measures
    accumulated phase (and the window length), not truncation error -- the
    same category of defect §2.5 identifies in the v1 gate.  The §3.2 features
    consume intervals, and a common time shift leaves them untouched, so the
    interval error is what the criterion is about.  Both numbers are reported.
    """
    n = int(min(ref.size, got.size))
    if n < 2:
        return 0.0
    a = np.diff(ref[:n].astype(np.float64))
    b = np.diff(got[:n].astype(np.float64))
    return float(np.median(np.abs(a - b)))


def _probe_features(cfg: Config, s: SpikeTrains, scales, weights):
    g = cfg.grid
    st = isim.isi_set(s, g.t_transient, g.t_steady_end)
    blocks = isim.compute_blocks(st, cfg, h=weights.h)
    return isim._l2_rows(isim.steady_vector(blocks, scales, weights)), st


def convergence_gate(cfg: Config, sim: Simulator, log) -> Dict:
    """C4 (§5): the gate is judged on spike times, not on the feature metric.

    Accept `dt_int` when, against the `dt = 0.001` ms reference over the 24
    probes: (1) the steady-window spike count is identical, (2) the median
    absolute spike-time error is < 0.05 ms, and (3) the §3.2 feature vector is
    within 1% relative L2.  Probes that fail (1) or (2) even at the
    sensitivity baseline (`dt/2` against the reference) are reported as
    *sensitive* -- the signature of a chaotic regime -- and excluded.
    """
    num = cfg.numerics
    probes = np.vstack([canonical_array(),
                        cfg.space.to_native(
                            sampling.lhs(num.gate_n_lhs, cfg.schedule.seed + 7))[:, :4]])
    rb = rheobase.find_rheobase(sim, probes, cfg, log=log)
    drive = np.where(np.isfinite(rb["i_star"]), rb["i_star"] + 5.0, cfg.drive.i_max)
    drive[:len(canonical_array())] = cfg.drive.canonical_drive
    log(f"convergence gate: {probes.shape[0]} probes x "
        f"{len(num.gate_dts) + 1} step sizes, drive I* + 5 "
        f"(canonical at I = {cfg.drive.canonical_drive:g})")

    ladder = tuple(num.gate_dts)
    dt_base = num.gate_dt_ref / 2.0
    runs: Dict[float, SpikeTrains] = {}
    diags: Dict[float, Dict] = {}
    for dt in ladder + (dt_base,):
        g = grid_for_dt(cfg.grid, dt)
        res = sim.run(probes, drive, grid=g, refine=False, store_traces=False,
                      max_spikes=cfg.isi.s_max)
        runs[dt] = res.spikes
        diags[dt] = {k: v.copy() for k, v in res.diag.items()}
        log(f"  dt={dt:<7g} mean spikes={diags[dt]['n_spikes'].mean():.1f} "
            f"max local err={np.nanmax(diags[dt]['max_local_error']):.2e}")
        del res

    ref = num.gate_dt_ref
    w = isim.Weights.default(cfg)
    g_ref = grid_for_dt(cfg.grid, ref)
    st_ref = isim.isi_set(runs[ref], cfg.grid.t_transient, cfg.grid.t_steady_end)
    blocks_ref = isim.compute_blocks(st_ref, cfg, h=w.h)
    scales = isim.fit_scales(blocks_ref, blocks_ref,
                             np.zeros((probes.shape[0], 2)),
                             np.ones(probes.shape[0], dtype=bool), cfg)
    F_ref = isim._l2_rows(isim.steady_vector(blocks_ref, scales, w))

    def window_times(s: SpikeTrains, i: int) -> np.ndarray:
        t = s.get(i)
        return t[(t >= cfg.grid.t_transient) & (t < cfg.grid.t_steady_end)]

    n_pr = probes.shape[0]
    spiking = diags[ref]["n_spikes"] > 0
    base_count = np.array([window_times(runs[dt_base], i).size ==
                           window_times(runs[ref], i).size for i in range(n_pr)])
    base_err = np.array([_isi_error(window_times(runs[ref], i),
                                    window_times(runs[dt_base], i))
                         for i in range(n_pr)])
    sensitive = ~base_count | ~(base_err < num.gate_spike_time_ms)
    use = spiking & ~sensitive
    log(f"  probes: {int(spiking.sum())} spiking, "
        f"{int((spiking & sensitive).sum())} sensitivity-limited (excluded, "
        f"chaotic regimes -- a finding, not a defect), {int(use.sum())} used")

    stats_by_dt: Dict[str, Dict[str, float]] = {}
    for dt in ladder:
        dcount = np.array([abs(window_times(runs[dt], i).size -
                               window_times(runs[ref], i).size)
                           for i in range(n_pr)])
        cnt_ok = dcount == 0
        err = np.array([_spike_time_error(window_times(runs[ref], i),
                                         window_times(runs[dt], i))
                        for i in range(n_pr)])
        ierr = np.array([_isi_error(window_times(runs[ref], i),
                                    window_times(runs[dt], i))
                         for i in range(n_pr)])
        F, _ = _probe_features(cfg, runs[dt], scales, w)
        rel = _rel_l2(F, F_ref)
        sel = use
        nu = int(sel.sum())
        stats_by_dt[f"{dt:g}"] = {
            "n_used": nu,
            "count_identical": int(cnt_ok[sel].sum()),
            "count_fraction": float(cnt_ok[sel].mean()) if nu else float("nan"),
            "max_count_diff": int(dcount[sel].max()) if nu else 0,
            "isi_fraction": float((ierr[sel] < num.gate_spike_time_ms).mean())
            if nu else float("nan"),
            "feature_fraction": float((rel[sel] <= num.gate_rel_l2).mean())
            if nu else float("nan"),
            "median_spike_time_error_ms": float(np.nanmedian(err[sel])),
            "max_spike_time_error_ms": float(np.nanmax(err[sel])) if sel.any()
            else float("nan"),
            "median_isi_error_ms": float(np.nanmedian(ierr[sel])),
            "max_isi_error_ms": float(np.nanmax(ierr[sel])) if sel.any()
            else float("nan"),
            "n_over_isi_error": int((~(ierr[sel] < num.gate_spike_time_ms)).sum()),
            "n_over_spike_time": int((~(err[sel] < num.gate_spike_time_ms)).sum()),
            "feature_rel_l2_median": float(np.median(rel[sel])),
            "feature_rel_l2_max": float(rel[sel].max()) if sel.any() else float("nan"),
            "n_over_feature": int((rel[sel] > num.gate_rel_l2).sum()),
            "mean_spikes": float(diags[dt]["n_spikes"].mean()),
            "max_local_error": float(np.nanmax(diags[dt]["max_local_error"])),
        }
        s = stats_by_dt[f"{dt:g}"]
        log(f"  dt={dt:<7g} counts {s['count_identical']}/{s['n_used']} identical "
            f"({s['count_fraction']:.0%}, worst |dn| {s['max_count_diff']}), "
            f"median |d ISI| {s['median_isi_error_ms']:.2e} ms "
            f"({s['n_over_isi_error']} over {num.gate_spike_time_ms:g}, worst "
            f"{s['max_isi_error_ms']:.2e}), "
            f"median |dt_spike| {s['median_spike_time_error_ms']:.2e} ms "
            f"(accumulated, reported only), feature rel L2 median "
            f"{s['feature_rel_l2_median']:.5f} "
            f"({s['n_over_feature']} over {num.gate_rel_l2:g})")

    def acceptable(dt: float) -> bool:
        """§5 as amended by §5.1: fractional, with a bounded worst case."""
        s = stats_by_dt[f"{dt:g}"]
        f = num.gate_pass_fraction
        return bool(
            s["n_used"]
            and s["count_fraction"] >= f
            and s["max_count_diff"] <= num.gate_max_count_diff
            and s["isi_fraction"] >= f
            and s["max_isi_error_ms"] <= num.gate_isi_error_cap_ms
            and s["feature_fraction"] >= f)

    eligible = [dt for dt in sorted(ladder, reverse=True) if acceptable(dt)]
    dt_sel = eligible[0] if eligible else min(ladder)
    passed = bool(eligible)
    log(f"  selected dt_int = {dt_sel:g} ms "
        f"({'all three checks pass' if passed else 'NO ladder step passes'})")

    # ---- V_floor (§3.4): sd(v) replaces the ac_power-based P_floor
    silent_probe = np.array([[a, b, -65.0, d]
                             for a in (0.005, 0.02, 0.1)
                             for b in (-0.10, -0.05, 0.0)
                             for d in (0.0, 6.0)], dtype=np.float64)
    grid_sel = grid_for_dt(cfg.grid, dt_sel)
    res = sim.run(silent_probe, 0.0, grid=grid_sel, store_traces=False)
    silent = res.diag["n_spikes_all"] <= 0
    sd = res.diag["sd_v"]
    if silent.any():
        v_num = float(num.v_floor_multiplier * np.median(sd[silent]))
    else:
        v_num = float(num.v_floor_multiplier * sd.min())
        log("  WARNING no silent reference found; numerical floor from min sd(v)")
    v_phys = float(num.v_floor_rms_mv)
    v_floor = max(v_num, v_phys)
    log(f"  V_floor: numerical term {v_num:.3e} mV "
        f"({int(silent.sum())}/{silent.size} silent references at I = 0), "
        f"physical term {v_phys:.3e} mV -> V_floor {v_floor:.3e} mV")
    del res

    log(f"convergence gate: {'PASS' if passed else 'FAIL'}")
    return {
        "probes": int(n_pr), "n_spiking": int(spiking.sum()),
        "n_sensitivity_limited": int((spiking & sensitive).sum()),
        "sensitive_probes": [int(i) for i in np.where(spiking & sensitive)[0]],
        "n_used": int(use.sum()),
        "baseline_dt": dt_base,
        "baseline_median_isi_error_ms": float(np.nanmedian(base_err[spiking])),
        "stats_by_dt": stats_by_dt,
        "dt_selected": float(dt_sel),
        "dt_spec_default": float(num.gate_dt_accept),
        "criteria": {"pass_fraction": num.gate_pass_fraction,
                     "max_count_diff": num.gate_max_count_diff,
                     "isi_error_ms": num.gate_spike_time_ms,
                     "isi_error_cap_ms": num.gate_isi_error_cap_ms,
                     "feature_rel_l2": num.gate_rel_l2},
        "passed": passed,
        "v_floor": v_floor, "v_floor_numerical": v_num,
        "v_floor_physical": v_phys,
        "silent_references": int(silent.sum()),
        "probe_drive": drive.tolist(),
    }


# ------------------------------------------------------ §4 + §7 simulation


def simulate_all(cfg: Config, sim: Simulator, x: np.ndarray, src: np.ndarray,
                 seed_rows: Sequence[int], log, timer: Timer,
                 keep_traces: bool) -> Dict:
    """Rheobase search then one production integration per sample."""
    p = cfg.space.to_native(x)                   # (N, 5): a,b,c,d,Delta
    N = p.shape[0]
    chunk = cfg.schedule.chunk
    n_out = cfg.grid.n_out
    traces = np.empty((N, n_out), dtype=np.float32) if keep_traces else None
    drive = np.zeros(N)
    rb = {k: np.zeros(N) for k in ("i_star", "i_sn", "i_ah", "onset_rate")}
    rb["exc_class"] = np.zeros(N, dtype=np.int8)
    rb["status"] = np.zeros(N, dtype=np.int8)
    diags: List[Dict] = []
    hists: List[np.ndarray] = []
    spike_parts: List[SpikeTrains] = []
    n_refined = 0
    seed_set = set(int(s) for s in seed_rows)

    for s0 in range(0, N, chunk):
        sl = slice(s0, min(s0 + chunk, N))
        par = p[sl, :4]
        with timer.section("rheobase"):
            r = rheobase.find_rheobase(sim, par, cfg)
        rows = np.array([i - s0 for i in sorted(seed_set & set(range(sl.start, sl.stop)))],
                        dtype=np.int64)
        I = rheobase.production_drive(r["i_star"], p[sl, 4], cfg, rows)
        for k in ("i_star", "i_sn", "i_ah", "onset_rate", "exc_class", "status"):
            rb[k][sl] = r[k]
        drive[sl] = I
        with timer.section("simulate"):
            res = sim.run(par, I, store_traces=keep_traces,
                          max_spikes=cfg.isi.s_max)
        n_refined += res.n_refined
        if keep_traces:
            traces[sl] = res.traces
        diags.append({k: v.copy() for k, v in res.diag.items()})
        hists.append(res.isi_hist)
        spike_parts.append(res.spikes)
        del res
        if (s0 // chunk) % 8 == 0 or sl.stop == N:
            log(f"  simulated {sl.stop}/{N} "
                f"(mean spikes/sample {diags[-1]['n_spikes_all'].mean():.1f}, "
                f"zero-spike {float((diags[-1]['n_spikes_all'] <= 0).mean()):.3f})")

    keys = diags[0].keys()
    diag = {k: np.concatenate([d[k] for d in diags]) for k in keys}
    spikes = SpikeTrains.concat(spike_parts)
    return {"x": x, "p": p, "drive": drive, "source": np.asarray(src, dtype=object),
            "traces": traces, "diag": diag, "isi_hist": np.vstack(hists),
            "spikes": spikes, "n_refined": n_refined, **rb}


# ------------------------------------------------------- §3.2 featurisation


class FeatureBuilder:
    """Blocks A-D for both windows, cached per KDE bandwidth (§3.2)."""

    def __init__(self, cfg: Config, spikes: SpikeTrains, log, timer: Timer):
        self.cfg = cfg
        self.spikes = spikes
        self.log = log
        self.timer = timer
        g = cfg.grid
        with timer.section("isi_sets"):
            self.steady_set = isim.isi_set(spikes, g.t_transient, g.t_steady_end)
            self.onset_set = isim.isi_set(spikes, 0.0, g.t_transient)
            self.shift_set = isim.isi_set(
                spikes, g.t_transient + g.window_shift,
                g.t_steady_end + g.window_shift)
        self.desc = isim.descriptors(self.steady_set, g.window_seconds, cfg)
        self.extras = isim.onset_extras(self.onset_set, spikes, cfg,
                                        self.desc["mean_isi"])
        self._cache: Dict[float, Tuple] = {}

    def blocks(self, h: float):
        if h not in self._cache:
            with self.timer.section("features"):
                st = isim.compute_blocks(self.steady_set, self.cfg, h=h)
                on = isim.compute_blocks(self.onset_set, self.cfg, h=h)
            self._cache[h] = (st, on)
            self.log(f"  ISI blocks at h={h:g}: "
                     f"{st.A.shape[1]}+{st.B.shape[1]}+{st.C.shape[1]}+1 dims, "
                     f"{int(st.usable.sum())} samples with >= 2 steady ISIs")
        return self._cache[h]

    def shifted_blocks(self, h: float):
        with self.timer.section("features"):
            return isim.compute_blocks(self.shift_set, self.cfg, h=h)

    def assemble(self, w: isim.Weights, mask: np.ndarray, scales=None):
        st, on = self.blocks(w.h)
        sc = scales or isim.fit_scales(st, on, self.extras, mask, self.cfg)
        F, F_steady, F_onset = isim.assemble(st, on, self.extras, sc, w)
        return F, F_steady, F_onset, sc


def _pca(F: np.ndarray, cfg: Config):
    scores, pca, curve = cluster.fit_pca_adaptive(
        F, 2, cfg.clustering.pca_min_evr, seed=cfg.schedule.seed)
    return scores, pca, curve


def stage2_grid(cfg: Config) -> List[isim.Weights]:
    """§3.3/§6: the ISI representation sweep, coordinate-wise around defaults.

    `w_A = 0` and `w_onset = 0` are in the grid so each block's contribution is
    measured rather than assumed (§12 risk table).
    """
    base = isim.Weights.default(cfg)
    out: List[isim.Weights] = []
    for h in cfg.isi.h_choices:
        if h != base.h:
            out.append(replace(base, h=h))
    for v in (0.0, 1.0):
        if v != base.w_a:
            out.append(replace(base, w_a=v))
    for v in (0.0, 1.0):
        if v != base.w_c:
            out.append(replace(base, w_c=v))
    for v in (0.1, 1.0):
        if v != base.w_d:
            out.append(replace(base, w_d=v))
    for v in (0.0, 1.0):
        if v != base.w_onset:
            out.append(replace(base, w_onset=v))
    return out


def stage2_sweep(cfg: Config, fb: FeatureBuilder, mask: np.ndarray, winner,
                 log, timer, sub: Optional[np.ndarray] = None
                 ) -> Tuple[List[cluster.SweepRow], Dict]:
    """Score every §3.3 representation at stage 1's clustering parameters."""
    rows: List[cluster.SweepRow] = []
    best: Dict = {}
    for w in stage2_grid(cfg):
        t0 = time.time()
        F, _, _, _ = fb.assemble(w, mask)
        Fa = F[mask]
        if sub is not None:
            Fa = np.ascontiguousarray(Fa[sub])
        P, _, curve = _pca(Fa, cfg)
        E, _ = cluster.umap_embed(P, n_components=cfg.clustering.umap_components,
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
        rows.append(cluster.SweepRow(
            2, winner.n_neighbors, winner.min_cluster_size, winner.min_samples,
            winner.method, ncl, nf, rv, sc, dt, h=w.h, w_a=w.w_a, w_b=w.w_b,
            w_c=w.w_c, w_d=w.w_d, w_onset=w.w_onset,
            pca_components=P.shape[1], label=w.tag()))
        log(f"  [stage2] {w.tag()}: k={ncl} noise={nf:.3f} rv={rv:.4f} "
            f"score={sc:.4f} pca={P.shape[1]} ({dt:.1f}s)")
        if not best or sc > best["score"]:
            best = {"score": sc, "weights": w}
        del E, slt, mst, P, F
    return rows, best


def default_row(cfg: Config) -> cluster.SweepRow:
    cl = cfg.clustering
    w = isim.Weights.default(cfg)
    return cluster.SweepRow(0, cl.umap_n_neighbors, cl.min_cluster_size,
                            cl.min_samples, cl.selection_method, 0,
                            float("nan"), float("nan"), float("nan"), 0.0,
                            h=w.h, w_a=w.w_a, w_b=w.w_b, w_c=w.w_c, w_d=w.w_d,
                            w_onset=w.w_onset, label="defaults")


# ------------------------------------------------------------ outputs helpers


def representatives(cfg: Config, E10: np.ndarray, labels: np.ndarray,
                    probs: np.ndarray, active_idx: np.ndarray) -> Dict[int, List[int]]:
    """3 points nearest the centroid in the 10-D clustering embedding."""
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


def full_labels_lookup(labels, pos_in_active, sample_id):
    j = pos_in_active[sample_id]
    return labels[j] if j >= 0 else -1


# ------------------------------------------------------------------- run


def run(cfg: Config) -> Dict:
    import cupy as cp

    t_start = time.time()
    outdir = os.path.abspath(cfg.outdir)
    os.makedirs(outdir, exist_ok=True)
    log = Log(os.path.join(outdir, "run.log"))
    timer = Timer()
    backend = plots.setup_backend(cfg.show)
    log(f"izhiclust v2: N={cfg.schedule.n_total}, Sobol design over "
        f"{cfg.space.n_dim} axes, matplotlib backend {backend}")

    sim = Simulator(cfg)
    with timer.section("gate"):
        gate = convergence_gate(cfg, sim, log)
    if not gate["passed"]:
        log("ABORT: convergence gate failed (§5).")
        raise SystemExit(2)
    if gate["dt_selected"] != cfg.grid.dt_int:
        cfg = replace(cfg, grid=grid_for_dt(cfg.grid, gate["dt_selected"]))
        sim.cfg = cfg
        log(f"production grid: dt_int={cfg.grid.dt_int:g} ms, L={cfg.grid.L}, "
            f"dt_out={cfg.grid.dt_out:g} ms, n_steps={cfg.grid.n_steps}")

    # ------------------------------------------------- §7 design + §4 drive
    x, src, seed_ids = sampling.design(cfg)
    keep_traces = not cfg.compact_csv
    with timer.section("sampling_loop"):
        S = simulate_all(cfg, sim, x, src, seed_ids, log, timer, keep_traces)
    N = S["x"].shape[0]
    log(f"simulated N={N}; refined at dt/4: {S['n_refined']}; "
        f"overflow flags set: {int((S['diag']['overflow'] > 0.5).sum())}")
    log(f"  drive: median I = {np.median(S['drive']):.2f}, median I* = "
        f"{np.nanmedian(S['i_star']):.2f}, no-rheobase "
        f"{int((S['status'] == rheobase.NO_RHEOBASE).sum())}, bistable-bracket "
        f"{int((S['status'] == rheobase.BRACKET_LOW).sum())}")
    hd = host_diagnostics(S["diag"], S["isi_hist"], cfg)

    # --------------------------------------------------- §3.2 featurisation
    fb = FeatureBuilder(cfg, S["spikes"], log, timer)
    desc = fb.desc
    cls = isim.classify(S["diag"], desc["n_isi"], gate["v_floor"], cfg)
    active = cls.active
    active_idx = np.where(active)[0]
    log(f"classes (§3.4): " + ", ".join(f"{k}={v}" for k, v in cls.counts().items()))

    w_default = isim.Weights.default(cfg)
    F, F_steady, F_onset, scales = fb.assemble(w_default, active)
    log(f"features: {F.shape[1]} dims "
        f"({cfg.isi.n_steady} steady + {cfg.isi.n_onset} onset), "
        f"weights {w_default.tag()}")

    # -------------------------------------------- §8.1 health gates (blocking)
    with timer.section("health_gates"):
        P, pca, evr_curve = _pca(F[active], cfg)
        health = isim.health_gates(F[active], desc["rate"][active],
                                   desc["cv_isi"][active],
                                   desc["burst_index"][active], cfg, evr_curve,
                                   rng=np.random.default_rng(cfg.schedule.seed))
    log(f"§8.1 gates: PCA components for 99% = {health['pca_components_99']} "
        f"(<= {cfg.gates.pca_max_components}: "
        f"{'PASS' if health['pca_pass'] else 'FAIL'}); "
        f"near/far distance ratio {health.get('near_over_far', float('nan')):.3f} "
        f"(<= {cfg.gates.monotonicity_ratio}: {health.get('monotonicity_pass')}); "
        f"octave {health.get('octave_mean_distance', float('nan')):.3f} vs 1.5x "
        f"{health.get('ratio15_mean_distance', float('nan')):.3f} "
        f"({health.get('octave_pass')})")
    if cfg.health_gate and not health.get("passed", False):
        log("ABORT: a §8.1 representation-health gate failed. The encoding is "
            "not fit for clustering; fix the representation, do not relax the "
            "gate (--no-health-gate runs anyway, for diagnosis).")
        raise SystemExit(3)

    # ------------------------------------------------------------- sweep
    sweep_rows: List[cluster.SweepRow] = []
    weights = w_default
    # The sweep is scored on a fixed subsample (assumption V2-A14): it compares
    # configurations, and v1 measured its own sweep at N = 26 624, so scoring
    # 54 + 10 configs on 20 000 points keeps the comparison faithful while the
    # full set is reserved for the final seeded fits and the §8.2 tests.
    n_sweep = int(min(cfg.clustering.sweep_subsample, P.shape[0]))
    sweep_sub = (np.sort(np.random.default_rng(cfg.schedule.seed).choice(
        P.shape[0], size=n_sweep, replace=False))
        if n_sweep < P.shape[0] else None)
    P_sweep = P if sweep_sub is None else np.ascontiguousarray(P[sweep_sub])
    if cfg.sweep:
        log(f"sweep scored on {P_sweep.shape[0]} of {P.shape[0]} clustered "
            f"samples")
        with timer.section("sweep_stage1"):
            rows1, aux = cluster.sweep_stage1(P_sweep, cfg, log=log)
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
                rows2, best2 = stage2_sweep(cfg, fb, active, winner, log,
                                            timer, sub=sweep_sub)
            sweep_rows += rows2
            if best2 and best2["score"] > winner.score:
                weights = best2["weights"]
                log(f"stage-2 winner: {weights.tag()} (score {best2['score']:.4f})")
                best_row = max(rows2, key=lambda r: r.score)
                winner = replace(winner, h=best_row.h, w_a=best_row.w_a,
                                 w_b=best_row.w_b, w_c=best_row.w_c,
                                 w_d=best_row.w_d, w_onset=best_row.w_onset,
                                 score=best_row.score, stage=2,
                                 label=best_row.label)
                F, F_steady, F_onset, scales = fb.assemble(weights, active)
                with timer.section("pca"):
                    P, pca, evr_curve = _pca(F[active], cfg)
                with timer.section("health_gates"):
                    health = isim.health_gates(
                        F[active], desc["rate"][active], desc["cv_isi"][active],
                        desc["burst_index"][active], cfg, evr_curve,
                        rng=np.random.default_rng(cfg.schedule.seed))
                log(f"  §8.1 gates at the selected representation: "
                    f"PCA99={health['pca_components_99']}, near/far "
                    f"{health.get('near_over_far', float('nan')):.3f}, "
                    f"octave pass {health.get('octave_pass')}")
            else:
                log("stage-2 winner: the defaults (no representation scored higher)")
    else:
        winner = default_row(cfg)
        t_unseeded = t_tree_mean = t_extract_mean = float("nan")
        log(f"sweep skipped; defaults nn={winner.n_neighbors} "
            f"mcs={winner.min_cluster_size} ms={winner.min_samples} "
            f"{winner.method}, {weights.tag()}")
    log(f"PCA: {P.shape[1]} components, cumulative EVR "
        f"{float(np.cumsum(pca.explained_variance_ratio_)[-1]):.5f}")
    evr = np.cumsum(pca.explained_variance_ratio_)

    # ------------------------------------------------- final seeded fits
    with timer.section("umap10_seeded"):
        t0 = time.time()
        E10, _ = cluster.umap_embed(P, n_components=cfg.clustering.umap_components,
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
        labels_raw = clusterer.labels_.copy()
        probs = clusterer.probabilities_.copy()
        mst = clusterer._min_spanning_tree
        slt = clusterer._single_linkage_tree
        other = "eom" if winner.method == "leaf" else "leaf"
        labels_other, _ = cluster.labels_from_tree(E10, slt,
                                                   winner.min_cluster_size, other)
    n_raw = int(np.unique(labels_raw[labels_raw >= 0]).size)

    # --------------------------------------------- §8.4 nameability merge
    with timer.section("merge"):
        desc_active = {k: v[active_idx] for k, v in desc.items()}
        labels, merge_rows, merge_sum = characterize.nameability_merge(
            labels_raw, desc_active, cfg)
    log(f"§8.4 nameability merge: {merge_sum['n_before']} -> "
        f"{merge_sum['n_after']} clusters "
        f"({merge_sum['n_merged_pairs']} indistinguishable pairs at "
        f"Bonferroni alpha {merge_sum['alpha']:.2e})")

    letters = cluster.assign_letters(labels)
    n_clusters = len(letters)
    noise_frac = float((labels < 0).mean())
    rv_total = cluster.relative_validity(mst, labels)
    rv_contrib = cluster.relative_validity_contributions(mst, labels)
    log(f"final clustering: {n_clusters} clusters ({winner.method}) after merge, "
        f"noise {noise_frac:.3f}, relative_validity {rv_total:.4f}")

    with timer.section("umap2_seeded"):
        E2, reducer2 = cluster.umap_embed(
            P, n_components=cfg.clustering.vis_components,
            n_neighbors=winner.n_neighbors, min_dist=cfg.clustering.vis_min_dist,
            n_epochs=cfg.clustering.umap_epochs,
            random_state=cfg.clustering.random_state)

    measured = {"umap_seeded_seconds": t_seeded}
    if cfg.sweep:
        measured.update({
            "umap_unseeded_seconds": t_unseeded,
            "umap_seeded_over_unseeded": t_seeded / max(t_unseeded, 1e-9),
            "hdbscan_tree_build_seconds": t_tree_mean,
            "hdbscan_extract_seconds": t_extract_mean,
            "tree_over_extract": t_tree_mean / max(t_extract_mean, 1e-9),
        })

    # ------------------------------------------------- §8.2 stability tests
    rng = np.random.default_rng(cfg.schedule.seed)
    with timer.section("perturbed_features"):
        shift = fb.shifted_blocks(weights.h)
        _, on = fb.blocks(weights.h)
        F_shift = isim._l2_rows(np.hstack([
            isim.steady_vector(shift, scales, weights),
            weights.w_onset * isim.onset_vector(on, fb.extras, scales, weights)]))
        P_window = pca.transform(F_shift[active]).astype(np.float32)
        P_dt, dt_rows = _dt_halved_features(cfg, sim, S, scales, weights, pca,
                                            active_idx, rng, log, timer)
    with timer.section("stability"):
        stab = validate.stability_tests(
            P, labels, cfg, n_neighbors=winner.n_neighbors,
            min_cluster_size=winner.min_cluster_size,
            min_samples=winner.min_samples, method=winner.method, rng=rng,
            P_window=P_window, P_dt=P_dt, dt_rows=dt_rows, log=log)

    Xn_active = S["x"][active]
    with timer.section("contiguity"):
        thr = validate.global_knn_threshold(Xn_active, cfg.clustering.contiguity_knn,
                                            cfg.clustering.contiguity_percentile)
        cont = validate.contiguity(Xn_active, S["p"][active], labels, cfg, thr,
                                   rng=rng)
    n_multi = sum(1 for c in cont.values() if c.multimodal)
    log(f"contiguity: threshold {thr:.4f}, {n_multi}/{n_clusters} MULTIMODAL")

    # ------------------------------------------- §8.3 canonical placement
    pos_in_active = -np.ones(N, dtype=np.int64)
    pos_in_active[active_idx] = np.arange(active_idx.size)
    seed_ids = [int(s) for s in seed_ids]
    held_class = []
    for s in seed_ids:
        held_class.append("Z" if cls.quiescent[s] else "S" if cls.subthreshold[s]
                          else "W" if cls.sparse[s] else "X" if cls.nonfinite[s]
                          else "")
    placements = validate.place_canonical(
        clusterer, E10, seed_ids, pos_in_active, labels, probs, letters,
        ~active[seed_ids], held_class, S["p"][seed_ids], S["drive"], desc,
        active_idx)
    for pl in placements:
        log(f"  {pl.label:3s} -> {pl.letter:4s} (p={pl.probability:.3f}), "
            f"descriptor-consistent {pl.consistent} "
            f"(within whiskers: {pl.inlier})"
            + ("  [EXEMPT §8.3]" if pl.exempt else "")
            + (f"  [held out: {pl.held_out_class}]" if pl.held_out else ""))

    # -------------------------------------------------- §14 cluster stats
    from sklearn.neighbors import NearestNeighbors
    nnq = NearestNeighbors(n_neighbors=2).fit(E10)
    _, nbr = nnq.kneighbors(E10)
    nn_is_noise = labels[nbr[:, 1]] < 0

    letter_order = sorted(letters.items(), key=lambda kv: (len(kv[1]), kv[1]))
    cluster_rows: List[Dict] = []
    corr_frames: Dict[str, object] = {}
    names: Dict[int, str] = {}
    unstable: Dict[int, bool] = {}
    multimodal: Dict[int, bool] = {}
    nm = cfg.naming
    diag_keys = ("rate", "cv_isi", "burst_index", "spikes_per_burst", "adapt_index")

    for cid, letter in letter_order:
        m = labels == cid
        idx_full = active_idx[m]
        P_all = S["p"][idx_full]
        med = {k: float(np.nanmedian(desc[k][idx_full])) for k in diag_keys}
        med["n_spikes"] = float(np.nanmedian(desc["n_spikes"][idx_full]))
        frac_bursting = float((desc["burst_index"][idx_full] > nm.burst_index).mean())
        frac_irregular = float((desc["cv_isi"][idx_full] > nm.cv_irregular).mean())
        med["frac_bursting"] = frac_bursting
        name = characterize.name_from_medians(med, cfg)
        names[cid] = name
        corr_frames[letter] = stats.triangular_frame(P_all)
        pers = stab.persistence.get(int(cid), float("nan"))
        unstable[cid] = bool(np.isfinite(pers) and pers < cfg.clustering.persistence_floor)
        co = cont[int(cid)]
        multimodal[cid] = bool(co.multimodal)

        row = {
            "letter": letter, "cluster_id": int(cid), "name": name,
            "name_key": characterize.name_key(name),
            "size": int(m.sum()),
            "noise_adjacency": float(nn_is_noise[m].mean()),
            "mean_probability": float(probs[m].mean()),
            "relative_validity_contribution": float(rv_contrib[cid])
            if cid < len(rv_contrib) else float("nan"),
            "persistence": float(pers), "unstable": unstable[cid],
            "n_param_components": int(co.n_components),
            "multimodal": multimodal[cid],
            "largest_component_fraction": float(co.largest_fraction),
            "n_param_components_total": int(co.n_components_total),
            "frac_bursting": frac_bursting,
            "frac_irregular": frac_irregular,
            "mixed": bool(0.2 <= frac_bursting <= 0.8),
            "median_n_spikes": med["n_spikes"],
            "median_n_isi": float(np.nanmedian(desc["n_isi"][idx_full])),
            "median_I": float(np.nanmedian(S["drive"][idx_full])),
            "median_I_star": float(np.nanmedian(S["i_star"][idx_full])),
            "median_delta": float(np.nanmedian(P_all[:, 4])),
            "frac_class2": float((S["exc_class"][idx_full] == 2).mean()),
            "median_onset_rate": float(np.nanmedian(S["onset_rate"][idx_full])),
        }
        for j, cn in enumerate(stats.COEFFS):
            row[f"mean_{cn}"] = float(P_all[:, j].mean())
            row[f"sd_{cn}"] = float(P_all[:, j].std(ddof=1)) if P_all.shape[0] > 1 \
                else float("nan")
        row["sd_log10_a"] = float(np.std(np.log10(P_all[:, 0]), ddof=1)) \
            if P_all.shape[0] > 1 else float("nan")
        for k in diag_keys:
            row[f"median_{k}"] = med[k]
        row.update(stats.corr_vector(P_all))
        cluster_rows.append(row)
        log("  " + characterize.describe(letter, med, int(m.sum()), cfg)
            + (" [UNSTABLE]" if unstable[cid] else "")
            + (f" [MULTIMODAL {co.n_components}/{co.n_components_total}, "
               f"largest {co.largest_fraction:.0%}]" if multimodal[cid] else ""))

    # ------------------------------------ §3.4 `S` class: resonant peaks
    s_rows = _subthreshold_report(cfg, sim, S, cls, log, timer)

    # --------------------------------------------- §16 representatives
    reps = representatives(cfg, E10, labels, probs, active_idx)
    rep_ids = [sid for cid, _ in letter_order for sid in reps[int(cid)]]
    rep_tags = [f"{letters[int(cid)]}{k + 1}"
                for cid, _ in letter_order for k in range(len(reps[int(cid)]))]
    with timer.section("representatives"):
        rp = S["p"][rep_ids]
        rd = S["drive"][rep_ids]
        full_n = cfg.grid.n_out_total
        res_disp = sim.run(rp, rd, store_u=True, refine=True, out_start=0,
                           n_out=full_n)
        rep_freqs, rep_mag = _fine_spectra(cfg, sim, rp, rd)
    log(f"re-simulated {len(rep_ids)} representatives: display trace at "
        f"dt_out={cfg.grid.dt_out:g} ms, spectrum at dt_out="
        f"{cfg.grid.dt_int:g} ms (§3.5)")

    # ------------------------------------------------------- §16.1 figures
    with timer.section("figures"):
        flags = {"unstable": unstable, "multimodal": multimodal}
        placement_xy = [(pl.label, pl.letter, *E2[pos_in_active[pl.sample_id]])
                        for pl in placements if pos_in_active[pl.sample_id] >= 0]
        cc = cls.counts()
        plots.scatter_2d(
            os.path.join(outdir, "clusters_umap2d.png"), E2, labels, letters,
            np.zeros(E2.shape[0], dtype=bool), placement_xy, flags, names,
            show=cfg.show,
            title=f"Izhikevich firing-pattern clusters (ISI encoding) — "
                  f"{n_clusters} clusters, N = {labels.size} clustered of "
                  f"{N}, I = I* + Δ",
            held_out=f"held out (§3.4): Z={cc['Z']}  S={cc['S']}  "
                     f"W={cc['W']}  X={cc['X']}")
        t_full = cfg.grid.t_full
        for k, (tag, sid) in enumerate(zip(rep_tags, rep_ids)):
            cid = int(full_labels_lookup(labels, pos_in_active, sid))
            a, b, c, d, dl = S["p"][sid]
            title = (f"{tag}  —  {names.get(cid, '')}\n"
                     f"a={a:.4g}  b={b:.4g}  c={c:.4g}  d={d:.4g}  "
                     f"Δ={dl:.3g}  I={S['drive'][sid]:.2f}  |  "
                     f"rate {desc['rate'][sid]:.0f} Hz, "
                     f"CV {desc['cv_isi'][sid]:.2f}, "
                     f"BI {desc['burst_index'][sid]:.2f}  |  sample #{sid}")
            plots.representative_figure(
                os.path.join(outdir, f"{tag}.png"), t_ms=t_full,
                v=res_disp.traces[k], f_hz=rep_freqs, mag=rep_mag[k],
                title=title,
                t_window=(cfg.grid.t_transient, cfg.grid.t_steady_end))
        for cid, letter in letter_order:
            ks = [i for i, t in enumerate(rep_tags) if t.startswith(letter)
                  and t[len(letter):].isdigit()]
            if not ks:
                continue
            labs = [f"{rep_tags[i]}  a={S['p'][rep_ids[i]][0]:.4g} "
                    f"b={S['p'][rep_ids[i]][1]:.4g} c={S['p'][rep_ids[i]][2]:.4g} "
                    f"d={S['p'][rep_ids[i]][3]:.4g} I={S['drive'][rep_ids[i]]:.2f}"
                    for i in ks]
            plots.cluster_trace_stack(
                os.path.join(outdir, f"{letter}_traces.png"), t_full,
                [res_disp.traces[i] for i in ks], labs,
                f"{letter} — {names[cid]}  (3 representatives)")
    log(f"figures written to {outdir}")

    return dict(cfg=cfg, log=log, timer=timer, S=S, hd=hd, desc=desc, gate=gate,
                labels=labels, labels_raw=labels_raw, n_raw=n_raw,
                merge_rows=merge_rows, merge_sum=merge_sum, probs=probs,
                letters=letters, letter_order=letter_order, names=names,
                active_idx=active_idx, active=active, classes=cls,
                E10=E10, E2=E2, P=P, pca=pca, clusterer=clusterer,
                labels_other=labels_other, other=other, winner=winner,
                weights=weights, scales=scales, health=health,
                F=F, F_steady=F_steady, F_onset=F_onset,
                sweep_rows=sweep_rows, stab=stab, cont=cont,
                placements=placements, cluster_rows=cluster_rows,
                corr_frames=corr_frames, reps=reps, rep_ids=rep_ids,
                rep_tags=rep_tags, res_disp=res_disp, rep_mag=rep_mag,
                rep_freqs=rep_freqs, s_rows=s_rows,
                rv_total=rv_total, rv_contrib=rv_contrib, noise_frac=noise_frac,
                n_clusters=n_clusters, unstable=unstable, multimodal=multimodal,
                outdir=outdir, t_start=t_start, evr=evr, evr_curve=evr_curve,
                sim=sim, pos_in_active=pos_in_active, seed_ids=seed_ids,
                backend=backend, t_seeded=t_seeded, measured=measured, N=N)


def _dt_halved_features(cfg: Config, sim: Simulator, S: Dict, scales, weights,
                        pca, active_idx: np.ndarray, rng, log, timer):
    """Re-simulate a subsample at `dt_int/2` and featurise it (§8.2)."""
    m = int(min(cfg.clustering.stability_subsample, active_idx.size))
    rows = np.sort(rng.choice(active_idx, size=m, replace=False))
    g = grid_for_dt(cfg.grid, cfg.grid.dt_int / 2.0)
    parts: List[SpikeTrains] = []
    with timer.section("simulate_dt_half"):
        for s0 in range(0, rows.size, cfg.schedule.chunk):
            sl = rows[s0:s0 + cfg.schedule.chunk]
            res = sim.run(S["p"][sl, :4], S["drive"][sl], grid=g,
                          store_traces=False, max_spikes=cfg.isi.s_max)
            parts.append(res.spikes)
            del res
    sp = SpikeTrains.concat(parts)
    st = isim.isi_set(sp, cfg.grid.t_transient, cfg.grid.t_steady_end)
    on = isim.isi_set(sp, 0.0, cfg.grid.t_transient)
    bs = isim.compute_blocks(st, cfg, h=weights.h)
    bo = isim.compute_blocks(on, cfg, h=weights.h)
    dsc = isim.descriptors(st, cfg.grid.window_seconds, cfg)
    ex = isim.onset_extras(on, sp, cfg, dsc["mean_isi"])
    F = isim._l2_rows(np.hstack([
        isim.steady_vector(bs, scales, weights),
        weights.w_onset * isim.onset_vector(bo, ex, scales, weights)]))
    log(f"  dt/2 re-simulation: {rows.size} samples at dt_int="
        f"{g.dt_int:g} ms for the §8.2 step-size test")
    # the stability tests index rows of the *active* subset, so the row list
    # has to be mapped out of full-sample ids
    pos = np.searchsorted(active_idx, rows)
    return pca.transform(F).astype(np.float32), pos


def _fine_spectra(cfg: Config, sim: Simulator, params: np.ndarray, drive):
    """§3.5: re-simulate at `dt_out = dt_int` and rFFT host-side."""
    g = replace(cfg.grid, L=1)
    n0 = int(round(cfg.grid.t_transient / g.dt_out)) - 1
    n_out = int(round((cfg.grid.t_steady_end - cfg.grid.t_transient) / g.dt_out))
    res = sim.run(params, drive, grid=g, store_traces=True, refine=True,
                  out_start=n0, n_out=n_out, peak_hold=False)
    freqs, mag = features.rfft_magnitude(res.traces, g.dt_out)
    del res
    return freqs, mag


def _subthreshold_report(cfg: Config, sim: Simulator, S: Dict, cls, log,
                         timer) -> List[Dict]:
    """§3.4: the `S` class is reported with its resonant peak frequency."""
    idx = np.where(cls.subthreshold)[0]
    if idx.size == 0:
        log("  S class (subthreshold/resonant): none")
        return []
    cap = int(min(idx.size, 128))
    sel = idx[:cap]
    with timer.section("subthreshold"):
        freqs, mag = _fine_spectra(cfg, sim, S["p"][sel], S["drive"][sel])
        pk = features.peak_frequency(freqs, mag)
    rows = [{"sample_id": int(i), "a": float(S["p"][i, 0]),
             "b": float(S["p"][i, 1]), "c": float(S["p"][i, 2]),
             "d": float(S["p"][i, 3]), "delta": float(S["p"][i, 4]),
             "I": float(S["drive"][i]), "sd_v": float(S["diag"]["sd_v"][i]),
             "resonant_peak_hz": float(p)} for i, p in zip(sel, pk)]
    log(f"  S class (subthreshold/resonant): {idx.size} samples, "
        f"median resonant peak {np.median(pk):.1f} Hz")
    return rows


# ------------------------------------------------------------------ outputs


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
    desc = ctx["desc"]
    outdir = ctx["outdir"]
    N = ctx["N"]
    active_idx = ctx["active_idx"]
    labels, probs = ctx["labels"], ctx["probs"]
    letters = ctx["letters"]
    cls = ctx["classes"]

    paths = iox.ensure_dirs(cfg)
    datadir = paths["data"]
    sample_ids = np.arange(N, dtype=np.int64)

    full_lab = -np.ones(N, dtype=np.int64)
    full_lab[active_idx] = labels
    full_raw = -np.ones(N, dtype=np.int64)
    full_raw[active_idx] = ctx["labels_raw"]
    full_other = -np.ones(N, dtype=np.int64)
    full_other[active_idx] = ctx["labels_other"]
    full_prob = np.zeros(N)
    full_prob[active_idx] = probs
    letter_col = np.array(["noise"] * N, dtype=object)
    for cid, L in letters.items():
        letter_col[active_idx[labels == cid]] = L
    letter_col[cls.quiescent] = "Z"
    letter_col[cls.subthreshold] = "S"
    letter_col[cls.sparse] = "W"
    letter_col[cls.nonfinite] = "X"
    E10_full = np.full((N, ctx["E10"].shape[1]), np.nan, dtype=np.float32)
    E10_full[active_idx] = ctx["E10"]
    E2_full = np.full((N, 2), np.nan, dtype=np.float32)
    E2_full[active_idx] = ctx["E2"]

    if ctx["winner"].method == "leaf":
        lab_leaf, lab_eom = full_lab, full_other
    else:
        lab_leaf, lab_eom = full_other, full_lab

    peak = np.full(N, np.nan)
    for r in ctx["s_rows"]:
        peak[r["sample_id"]] = r["resonant_peak_hz"]

    # -------------------------------------------------------- samples.csv
    with timer.section("csv_samples"):
        import pandas as pd
        d = {
            "sample_id": sample_ids, "source": S["source"].astype(str),
            "a": S["p"][:, 0], "b": S["p"][:, 1], "c": S["p"][:, 2],
            "d": S["p"][:, 3], "delta": S["p"][:, 4], "I": S["drive"],
            "I_star": S["i_star"], "I_SN": S["i_sn"], "I_AH": S["i_ah"],
            "exc_class": S["exc_class"], "onset_rate": S["onset_rate"],
            "rheobase_status": S["status"],
            "n_spikes_window": desc["n_spikes"].astype(np.int64),
            "n_spikes_total": S["diag"]["n_spikes_all"].astype(np.int64),
            "n_isi": desc["n_isi"].astype(np.int64),
            "overflow": (S["diag"]["overflow"] > 0.5).astype(np.int8),
            "rate": desc["rate"], "cv_isi": desc["cv_isi"],
            "burst_index": desc["burst_index"],
            "spikes_per_burst": desc["spikes_per_burst"],
            "adapt_index": desc["adapt_index"],
            "mean_isi": desc["mean_isi"], "median_isi": desc["median_isi"],
            "mean_v": S["diag"]["mean_v"], "sd_v": S["diag"]["sd_v"],
            "resonant_peak_hz": peak,
            "max_local_error": S["diag"]["max_local_error"],
            "finite": S["diag"]["finite"].astype(np.int8),
            "label_leaf": lab_leaf, "label_eom": lab_eom,
            "label_premerge": full_raw,
            "letter": letter_col.astype(str), "membership_prob": full_prob,
        }
        for j in range(E10_full.shape[1]):
            d[f"umap10_{j}"] = E10_full[:, j]
        for j in range(2):
            d[f"umap2_{j}"] = E2_full[:, j]
        df = pd.DataFrame(d)
        df.to_csv(os.path.join(datadir, "samples.csv"), index=False,
                  float_format="%.6g")
        log(f"  wrote samples.csv ({df.shape[0]} x {df.shape[1]})")

        iox.write_matrix_csv(os.path.join(datadir, "features_isi.csv"),
                             ctx["F_steady"], sample_ids, cfg.isi.columns(),
                             "%.6g", log=log)
        iox.write_matrix_csv(os.path.join(datadir, "features_isi_onset.csv"),
                             ctx["F_onset"], sample_ids,
                             cfg.isi.onset_columns(), "%.6g", log=log)

    # -------------------------------------------------------- bulk files
    bulk = {}
    if not cfg.compact_csv:
        with timer.section("csv_bulk"):
            bulk["spike_times"] = iox.write_spike_times_csv(
                os.path.join(datadir, "spike_times.csv"), S["spikes"],
                sample_ids, gzip=cfg.gzip_csv, log=log)
            t_cols = iox.trace_columns(cfg.grid.t_window)
            bulk["traces"] = iox.write_matrix_csv(
                os.path.join(datadir, "traces.csv"), S["traces"], sample_ids,
                t_cols, iox.TRACE_FMT, gzip=cfg.gzip_csv, log=log)
    else:
        log("  --compact-csv: traces.csv and spike_times.csv omitted")

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
        for letter, frame in ctx["corr_frames"].items():
            frame.to_csv(os.path.join(datadir, f"corr_{letter}.csv"),
                         float_format="%.6g")
        if ctx["s_rows"]:
            pd.DataFrame(ctx["s_rows"]).to_csv(
                os.path.join(datadir, "subthreshold_S.csv"), index=False,
                float_format="%.6g")
        if ctx["merge_rows"]:
            pd.DataFrame(ctx["merge_rows"]).to_csv(
                os.path.join(datadir, "merge_pairs.csv"), index=False,
                float_format="%.3g")
        sp = S["spikes"]
        np.savez_compressed(
            os.path.join(datadir, "run_data.npz"),
            params=S["p"].astype(np.float32), x_unit=S["x"].astype(np.float32),
            drive=S["drive"].astype(np.float32),
            i_star=S["i_star"].astype(np.float32),
            i_sn=S["i_sn"].astype(np.float32), i_ah=S["i_ah"].astype(np.float32),
            exc_class=S["exc_class"], rheobase_status=S["status"],
            onset_rate=S["onset_rate"].astype(np.float32),
            features_isi=ctx["F_steady"].astype(np.float32),
            features_onset=ctx["F_onset"].astype(np.float32),
            spike_times=sp.times, spike_offsets=sp.offsets,
            spike_overflow=sp.overflow,
            isi_hist=S["isi_hist"].astype(np.int32),
            umap10=E10_full, umap2=E2_full,
            label_leaf=lab_leaf.astype(np.int32),
            label_eom=lab_eom.astype(np.int32),
            label_premerge=full_raw.astype(np.int32),
            letter=letter_col.astype(str),
            membership_prob=full_prob.astype(np.float32),
            source=S["source"].astype(str),
            class_Z=cls.quiescent, class_S=cls.subthreshold,
            class_W=cls.sparse, class_X=cls.nonfinite,
            **{f"diag_{k}": v.astype(np.float32) for k, v in S["diag"].items()},
            **{f"desc_{k}": np.asarray(v, dtype=np.float32)
               for k, v in desc.items()})
        rows = [dataclasses.asdict(r) for r in ctx["sweep_rows"]]
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(outdir, "sweep_table.csv"),
                                      index=False, float_format="%.6g")
        log("  wrote cluster_stats.csv, corr_*.csv, run_data.npz, sweep_table.csv")

    # ------------------------------------------------------ round trip
    rt = {"rows": 0, "rel_l2_max": float("nan")}
    if not cfg.compact_csv:
        with timer.section("roundtrip"):
            sp_path = os.path.join(datadir, "spike_times.csv"
                                   + (".gz" if cfg.gzip_csv else ""))
            rt = iox.roundtrip_check(sp_path, ctx["F_steady"], cfg,
                                     ctx["scales"], ctx["weights"],
                                     include=ctx["active"], log=log)

    return {"paths": paths, "bulk": bulk, "roundtrip": rt,
            "letter_col": letter_col, "lab_leaf": lab_leaf, "lab_eom": lab_eom}


# ------------------------------------------------------- §9 acceptance


def acceptance(ctx: Dict, outs: Dict) -> List[Dict]:
    cfg: Config = ctx["cfg"]
    S = ctx["S"]
    desc = ctx["desc"]
    N = ctx["N"]
    datadir = outs["paths"]["data"]
    gates = cfg.gates
    checks: List[Dict] = []

    def add(name, ok, detail):
        checks.append({"criterion": name, "pass": None if ok is None else bool(ok),
                       "detail": detail})

    g = ctx["gate"]
    sd = g["stats_by_dt"][f"{g['dt_selected']:g}"]
    add("convergence gate passes all three checks (§5, fractional form §5.1)",
        g["passed"],
        f"dt_int {g['dt_selected']:g} ms: counts identical for "
        f"{sd['count_fraction']:.0%} of {sd['n_used']} used probes (worst "
        f"|Δn| {sd['max_count_diff']}), per-ISI error under 0.05 ms for "
        f"{sd['isi_fraction']:.0%} (worst {sd['max_isi_error_ms']:.2e} ms), "
        f"within 1% rel L2 for {sd['feature_fraction']:.0%}")
    nx = int(ctx["classes"].nonfinite.sum())
    nov = int((S["diag"]["overflow"] > 0.5).sum())
    add("zero samples labelled X; zero overflow flags set (§3.1)",
        nx == 0 and nov == 0, f"X = {nx}, overflow = {nov} "
        f"(max spikes recorded {int(S['diag']['n_spikes_recorded'].max())} of "
        f"S_max = {cfg.isi.s_max})")
    add(f"noise fraction <= {gates.noise_fraction_max:.0%} at the selected config",
        ctx["noise_frac"] <= gates.noise_fraction_max,
        f"noise = {ctx['noise_frac']:.3f}")
    h = ctx["health"]
    add(f"PCA components for 99% of variance <= {gates.pca_max_components} (§8.1)",
        h["pca_pass"], f"{h['pca_components_99']} components "
        f"(v1: 92 of 128)")
    mono = h.get("monotonicity_pass")
    oct_ = h.get("octave_pass")
    add("metric monotonicity and octave gates (§8.1)",
        None if mono is None or oct_ is None else (mono and oct_),
        f"near/far ratio {h.get('near_over_far', float('nan')):.3f} "
        f"(<= {gates.monotonicity_ratio}, v1 0.818); octave distance "
        f"{h.get('octave_mean_distance', float('nan')):.3f} vs 1.5x "
        f"{h.get('ratio15_mean_distance', float('nan')):.3f} "
        f"(v1 0.379 vs 0.501, inverted)")

    # fraction of clusters passing both the stability and contiguity gates
    rows = ctx["cluster_rows"]
    n_cl = len(rows)
    both = sum(1 for r in rows if not r["unstable"] and not r["multimodal"])
    frac = both / max(n_cl, 1)
    add(f"fraction of reported clusters passing both the stability and "
        f"contiguity gates >= {gates.gated_cluster_fraction:.2f}",
        frac >= gates.gated_cluster_fraction,
        f"{both}/{n_cl} = {frac:.3f} (v1: 0.216 — 22 of 102)")

    keys = sorted({r["name_key"] for r in rows})
    add(f"distinct names after the §8.4 merge >= {gates.distinct_names_min}",
        len(keys) >= gates.distinct_names_min,
        f"{len(keys)} distinct names across {n_cl} clusters (v1: 6 of 102): "
        + "; ".join(keys[:12]) + ("; ..." if len(keys) > 12 else ""))

    # bursting capture
    bi = desc["burst_index"][ctx["active_idx"]]
    burst_mask = bi > gates.burst_index_threshold
    n_burst = int(burst_mask.sum())
    shares = []
    for r in rows:
        m = ctx["labels"] == r["cluster_id"]
        share = float((burst_mask & m).sum() / max(n_burst, 1))
        shares.append((r["letter"], r["name_key"], share,
                       "bursting" in r["name_key"]))
    bursty = [s for s in shares if s[3]]
    best = max((s[2] for s in bursty), default=0.0)
    union = sum(s[2] for s in bursty)
    holders = sum(1 for s in shares if s[2] > 0)
    add(f">= 1 cluster named bursting holding >= "
        f"{gates.burst_capture_fraction:.0%} of all BI > "
        f"{gates.burst_index_threshold:g} samples",
        best >= gates.burst_capture_fraction,
        f"best single bursting-named cluster holds {best:.1%} of the "
        f"{n_burst} burst-index samples; all bursting-named clusters together "
        f"{union:.1%}; those samples are spread over {holders} clusters "
        f"(v1: 86 of 102)")

    for key, run in ctx["stab"].runs.items():
        add(f"{run.name} ARI >= {run.threshold} (§8.2)", run.passed,
            f"mean ARI {run.mean_ari:.3f} over {run.n_fits} fits of "
            f"{run.n_subsample} points — {run.description}")
    for key, thr in (("window", cfg.clustering.ari_window),
                     ("dt", cfg.clustering.ari_dt)):
        if key not in ctx["stab"].runs:
            add(f"{key} ARI >= {thr} (§8.2)", None, "not evaluated")

    bad = [p.label for p in ctx["placements"]
           if not p.exempt and not p.consistent]
    whisker_only = [p.label for p in ctx["placements"]
                    if p.label in bad and p.inlier]
    add("canonical placement by descriptor consistency (§8.3), TC+RZ exempt",
        not bad, "all consistent" if not bad else
        (f"{len(whisker_only)} of {len(bad)} are inside the Tukey whiskers "
         f"({', '.join(whisker_only)}) and so are ordinary-spread misses "
         f"rather than gross mismatches; " if whisker_only else "")
        + "inconsistent: " + ", ".join(
            f"{p.label} in {p.letter} ("
            + ", ".join(k for k, v in p.descriptors.items() if not v["inside"])
            + " outside the host IQR)"
            for p in ctx["placements"] if p.label in bad))

    want = ["samples.csv", "features_isi.csv", "features_isi_onset.csv",
            "cluster_stats.csv", "run_data.npz"]
    if not cfg.compact_csv:
        suff = ".gz" if cfg.gzip_csv else ""
        want += ["traces.csv" + suff, "spike_times.csv" + suff]
    missing = [f for f in want
               if not (os.path.exists(os.path.join(datadir, f))
                       and os.path.getsize(os.path.join(datadir, f)) > 0)]
    fig = ["clusters_umap2d.png", "cluster_report.md", "run_manifest.json"]
    missing += [f for f in fig
                if not (os.path.exists(os.path.join(ctx["outdir"], f))
                        and os.path.getsize(os.path.join(ctx["outdir"], f)) > 0)]
    add("every artefact present and non-empty", not missing,
        "all present" if not missing else f"missing: {missing}")

    row_files = ["samples.csv", "features_isi.csv", "features_isi_onset.csv"]
    if not cfg.compact_csv:
        suff = ".gz" if cfg.gzip_csv else ""
        row_files += ["traces.csv" + suff, "spike_times.csv" + suff]
    rows_n = {}
    for f in row_files:
        pth = os.path.join(datadir, f)
        if os.path.exists(pth):
            rows_n[f] = _count_lines(pth) - 1
    ok_rows = all(v == N for v in rows_n.values())
    ids_ok = all(_first_last_sample_id(os.path.join(datadir, f)) == (0, N - 1)
                 for f in row_files if f in rows_n)
    add("every row-per-sample file has exactly N rows and identical sample_id "
        "ordering", ok_rows and ids_ok,
        f"rows = {rows_n}, first/last sample_id consistent = {ids_ok}")

    rt = outs["roundtrip"]
    if not np.isfinite(rt.get("rel_l2_max", np.nan)):
        add("round trip: spike times -> features reproduces features_isi.csv "
            f"to {gates.roundtrip_tol:g}", None,
            "not applicable (--compact-csv omits spike_times.csv)")
    else:
        add("round trip: spike times -> features reproduces features_isi.csv "
            f"to {gates.roundtrip_tol:g} relative",
            rt["rel_l2_max"] <= gates.roundtrip_tol,
            f"max rel L2 {rt['rel_l2_max']:.2e} over {rt['rows']} rows "
            f"(mean {rt['rel_l2_mean']:.2e})")
    return checks


# ------------------------------------------------------------------ report


def build_report(ctx: Dict, outs: Dict, checks: List[Dict], manifest: Dict) -> str:
    cfg: Config = ctx["cfg"]
    S = ctx["S"]
    desc = ctx["desc"]
    N = ctx["N"]
    w = ctx["winner"]
    wt = ctx["weights"]
    cls = ctx["classes"]
    L: List[str] = []
    A = L.append
    A("# Izhikevich spiking-neuron cluster discovery — v2 run report\n")
    A("Implements `IMPROVEMENTS.md` (C1–C9) on top of `SPEC.md`.\n")
    A(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- Samples `N` = {N} (single Sobol design over "
      f"`(a, b, c, d, Δ)`), drive `I = I*(θ) + Δ` (§4)")
    A(f"- Encoding: ISI sequence, {ctx['F'].shape[1]} dims "
      f"({cfg.isi.n_steady} steady + {cfg.isi.n_onset} onset), "
      f"`{wt.tag()}`")
    A(f"- Selected clustering config: `n_neighbors={w.n_neighbors}`, "
      f"`min_cluster_size={w.min_cluster_size}`, `min_samples={w.min_samples}`, "
      f"`cluster_selection_method='{w.method}'`")
    A(f"- Clusters: **{ctx['n_clusters']}** after the §8.4 merge "
      f"({ctx['n_raw']} before), "
      f"{int(np.unique(ctx['labels_other'][ctx['labels_other'] >= 0]).size)} at "
      f"`{ctx['other']}`; noise {ctx['noise_frac']:.1%}; "
      f"`relative_validity_` {ctx['rv_total']:.4f}")
    A(f"- Held out (§3.4): `Z` = {int(cls.quiescent.sum())}, `S` = "
      f"{int(cls.subthreshold.sum())}, `W` = {int(cls.sparse.sum())}, `X` = "
      f"{int(cls.nonfinite.sum())}; clustered {int(cls.active.sum())} "
      f"({cls.active.mean():.1%} of N — v1 clustered 54.9%)")
    A(f"- PCA: {ctx['P'].shape[1]} components, "
      f"{ctx['evr'][-1]:.4f} of the variance; 99% reached at "
      f"**{ctx['health']['pca_components_99']}** components (v1: 92 of 128)\n")

    A("## 1. Convergence gate (§5 — C4)\n")
    g = ctx["gate"]
    A(f"Judged on spike times, not on the feature metric.  Probe set: "
      f"{g['probes']} ({g['n_spiking']} spiking, "
      f"{g['n_sensitivity_limited']} sensitivity-limited and excluded as "
      f"chaotic, {g['n_used']} used).  Sensitivity baseline (`dt="
      f"{g['baseline_dt']:g}` vs `{cfg.numerics.gate_dt_ref:g}` ms) median "
      f"interval error {g['baseline_median_isi_error_ms']:.2e} ms.\n")
    sel = g["stats_by_dt"][f"{g['dt_selected']:g}"]
    cr = g["criteria"]
    A("| dt (ms) | counts identical | worst \\|Δn\\| | median per-ISI err "
      "(ms) | probes within 0.05 ms | worst per-ISI err | probes within 1% "
      "rel L2 | median \\|Δt_spike\\| (accumulated, ungated) | accepted |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k, v in g["stats_by_dt"].items():
        ok = (v["count_fraction"] >= cr["pass_fraction"]
              and v["max_count_diff"] <= cr["max_count_diff"]
              and v["isi_fraction"] >= cr["pass_fraction"]
              and v["max_isi_error_ms"] <= cr["isi_error_cap_ms"]
              and v["feature_fraction"] >= cr["pass_fraction"])
        A(f"| {k} | {v['count_identical']}/{v['n_used']} "
          f"({v['count_fraction']:.0%}) | {v['max_count_diff']} | "
          f"{v['median_isi_error_ms']:.2e} | {v['isi_fraction']:.0%} | "
          f"{v['max_isi_error_ms']:.2e} | {v['feature_fraction']:.0%} | "
          f"{v['median_spike_time_error_ms']:.2e} | "
          f"{'**yes**' if ok else 'no'}"
          + (" (selected)" if float(k) == g["dt_selected"] else "") + " |")
    A("")
    A(f"- selected `dt_int` = **{g['dt_selected']:g} ms**, the coarsest ladder "
      f"step meeting all three §5 checks in their §5.1 fractional form: "
      f"{sel['count_fraction']:.0%} of used probes have an identical spike "
      f"count and none differs by more than {sel['max_count_diff']} "
      f"(cap {cr['max_count_diff']}); {sel['isi_fraction']:.0%} have a median "
      f"per-ISI error under {cr['isi_error_ms']:g} ms and the worst is "
      f"{sel['max_isi_error_ms']:.2e} ms (cap {cr['isi_error_cap_ms']:g}); "
      f"{sel['feature_fraction']:.0%} are inside "
      f"{cr['feature_rel_l2']:.0%} relative L2 — each against the "
      f"{cr['pass_fraction']:.0%} threshold")
    strict = [float(k) for k, v in g["stats_by_dt"].items()
              if v["n_used"] and v["count_identical"] == v["n_used"]
              and v["n_over_isi_error"] == 0 and v["n_over_feature"] == 0]
    A(f"- the strict reading of the draft (every probe, every check) would "
      f"have selected "
      + (f"{max(strict):g} ms" if strict else "no ladder step")
      + f" instead; §5.1 amends it because an all-probes rule gets "
        f"monotonically stricter as the probe set grows, which would make the "
        f"gate's answer depend on how many probes were run (V2-A15)")
    A(f"- `V_floor` = **{g['v_floor']:.4e} mV** = max(numerical "
      f"{g['v_floor_numerical']:.3e} from {g['silent_references']} silent "
      f"references at `I = 0`, physical {g['v_floor_physical']:.3e} = "
      f"{cfg.numerics.v_floor_rms_mv:g} mV RMS) — `sd(v)` replaces v1's "
      f"`ac_power`/`P_floor` (§3.4)\n")

    A("## 2. Drive relative to rheobase (§4 — C3)\n")
    ok = np.isfinite(S["i_star"])
    A(f"- `I*` bracketed for **{int(ok.sum())}** of {N} samples "
      f"({int((S['status'] == rheobase.NO_RHEOBASE).sum())} have no rheobase "
      f"at `I <= {cfg.drive.i_max:g}`, "
      f"{int((S['status'] == rheobase.BRACKET_LOW).sum())} fire already at the "
      f"analytic bracket — bistable corners, §4.1 item 3)")
    A(f"- `I*` quartiles: {np.nanpercentile(S['i_star'], 25):.2f} / "
      f"{np.nanmedian(S['i_star']):.2f} / "
      f"{np.nanpercentile(S['i_star'], 75):.2f}; production `I` median "
      f"{np.median(S['drive']):.2f} (v1 held `I = 10` for every sample)")
    A(f"- excitability class (§4.1): Class 1 (`b <= a`) "
      f"{int((S['exc_class'] == 1).sum())}, Class 2 (`b > a`) "
      f"{int((S['exc_class'] == 2).sum())}")
    onr = S["onset_rate"][ok]
    c1 = ok & (S["exc_class"] == 1)
    c2 = ok & (S["exc_class"] == 2)
    A(f"- f–I onset shape (§4.3): firing rate measured at `I*` itself, over a "
      f"{cfg.drive.t_measure:g} ms window (so the measurement floor is "
      f"{1000.0 / cfg.drive.t_measure:g} Hz).  Median {np.median(onr):.1f} Hz "
      f"overall; **{float((onr > 20.0).mean()):.1%}** of samples start above "
      f"20 Hz, which a continuous (Class 1) onset cannot do at a bisection "
      f"tolerance of {cfg.drive.tol:g} in `I`.  Split by the analytic class: "
      f"median {np.median(S['onset_rate'][c1]) if c1.any() else float('nan'):.1f}"
      f" Hz for Class 1 (`b <= a`, no Hopf) against "
      f"{np.median(S['onset_rate'][c2]) if c2.any() else float('nan'):.1f} Hz "
      f"for Class 2 — the discrete excitability split v1 could not observe")
    A(f"- zero-spike samples: **{int((S['diag']['n_spikes_all'] <= 0).sum())}** "
      f"({float((S['diag']['n_spikes_all'] <= 0).mean()):.1%}), against "
      f"**45.1%** in v1 — the §4.3 prediction\n")

    A("## 3. Representation health (§8.1 — the gates that would have caught v1)\n")
    h = ctx["health"]
    A("| gate | threshold | v1 | v2 | result |")
    A("|---|---|---|---|---|")
    A(f"| PCA components for 99% variance | <= {cfg.gates.pca_max_components} "
      f"| 92 | {h['pca_components_99']} | "
      f"{'PASS' if h['pca_pass'] else 'FAIL'} |")
    A(f"| mean cosine distance at \\|Δ log f₀\\| ∈ [0.05, 0.10] relative to "
      f"\\|Δ log f₀\\| > 1 | <= {cfg.gates.monotonicity_ratio} | 0.818 | "
      f"{h.get('near_over_far', float('nan')):.3f} | "
      f"{'PASS' if h.get('monotonicity_pass') else 'FAIL'} |")
    A(f"| octave-pair distance >= 1.5x-pair distance | required | violated "
      f"(0.379 vs 0.501) | {h.get('octave_mean_distance', float('nan')):.3f} vs "
      f"{h.get('ratio15_mean_distance', float('nan')):.3f} | "
      f"{'PASS' if h.get('octave_pass') else 'FAIL'} |")
    A("")
    A(f"Measured on {h.get('tonic_subset', 0)} pure-tonic samples "
      f"(`CV < {cfg.gates.tonic_cv_max}`, `BI < {cfg.gates.tonic_bi_max}`), "
      f"the same subset definition §2.3 used.  The v2 version of the §2.3 "
      f"saturation table:\n")
    A("| \\|Δ log f₀\\| | pairs | mean cosine distance (v2) |")
    A("|---|---|---|")
    for r in h.get("distance_by_separation", []):
        A(f"| {r['lo']:g} – {r['hi']:g} | {r['pairs']} | "
          f"{r['mean_cosine_distance']:.4f} |")
    A("")

    A("## 4. Clusters\n")
    A("Names follow v1 §14.2 as amended by §8.4; flags follow §8.2 (UNSTABLE) "
      "and v1 §13.3 (MULTIMODAL).  Every clustered sample fires and has at "
      "least "
      f"{cfg.isi.n_isi_min} ISIs, so the v1 silent/subthreshold names are gone "
      "— those populations are the `Z`/`S`/`W` classes above.\n")
    for row in ctx["cluster_rows"]:
        cid = row["cluster_id"]
        letter = row["letter"]
        flagtxt = []
        if row["unstable"]:
            flagtxt.append("**UNSTABLE**")
        if row["multimodal"]:
            flagtxt.append(f"**MULTIMODAL** ({row['n_param_components']} components)")
        A(f"### {letter} — {row['name']}\n")
        A(f"- size {row['size']}, mean membership prob "
          f"{row['mean_probability']:.3f}, noise adjacency "
          f"{row['noise_adjacency']:.3f}")
        A(f"- medians: rate {row['median_rate']:.1f} Hz, CV(ISI) "
          f"{row['median_cv_isi']:.3f}, burst index "
          f"{row['median_burst_index']:.3f}, spikes/burst "
          f"{row['median_spikes_per_burst']:.2f}, adaptation "
          f"{row['median_adapt_index']:+.3f}, {row['median_n_isi']:.0f} ISIs")
        A(f"- drive: median `I` {row['median_I']:.2f} = `I*` "
          f"{row['median_I_star']:.2f} + `Δ` {row['median_delta']:.2f}; "
          f"{row['frac_class2']:.0%} Class 2; bursting members "
          f"{row['frac_bursting']:.0%}, irregular {row['frac_irregular']:.0%}")
        A(f"- mean coefficient vector: a={row['mean_a']:.4g}, "
          f"b={row['mean_b']:.4g}, c={row['mean_c']:.4g}, d={row['mean_d']:.4g}, "
          f"Δ={row['mean_delta']:.4g}")
        A(f"- persistence {row['persistence']:.2f}, DBCV contribution "
          f"{row['relative_validity_contribution']:+.4f}"
          + (("  " + " ".join(flagtxt)) if flagtxt else ""))
        if row["multimodal"]:
            co = ctx["cont"][cid]
            A(f"- pooled mean is **not interpretable**: "
              f"{co.n_components_total} parameter-space components, "
              f"{co.n_components} of them holding >= 5% of the cluster, "
              f"largest {row['largest_component_fraction']:.0%}")
            for sz, mn in zip(co.sizes, co.means):
                A(f"    - n={sz}: a={mn[0]:.4g}, b={mn[1]:.4g}, c={mn[2]:.4g}, "
                  f"d={mn[3]:.4g}, Δ={mn[4]:.4g}")
        A("\n```")
        A("coefficient vector: SD on the diagonal, Pearson r below it")
        A(stats.render_triangular(ctx["corr_frames"][letter]))
        A("```\n")

    if ctx["merge_rows"]:
        A("## 5. Nameability merge (§8.4)\n")
        ms = ctx["merge_sum"]
        A(f"{ms['n_before']} clusters -> **{ms['n_after']}**: "
          f"{ms['n_merged_pairs']} of {ms['n_pairs']} pairs were "
          f"indistinguishable on every descriptor at the Bonferroni-corrected "
          f"threshold α = {ms['alpha']:.2e} "
          f"({cfg.gates.merge_alpha:g} / ({ms['n_pairs']} pairs × "
          f"{len(characterize.MERGE_DESCRIPTORS)} descriptors)).\n")
        A("| merged group (pre-merge ids) |")
        A("|---|")
        for grp in ms.get("groups", []):
            A(f"| {', '.join(str(x) for x in grp)} |")
        A("\nFull pair table in `data/merge_pairs.csv`.\n")
    else:
        A("## 5. Nameability merge (§8.4)\n")
        A(f"No pair of clusters was indistinguishable on all "
          f"{len(characterize.MERGE_DESCRIPTORS)} descriptors at α = "
          f"{ctx['merge_sum']['alpha']:.2e}; nothing merged, so the "
          f"{ctx['n_clusters']} clusters are each descriptively distinct from "
          f"every other.\n")

    A("## 6. Canonical type placement (§8.3 — with teeth)\n")
    A("Each type must land in a cluster whose interquartile range contains the "
      "type's own measured `(CV, burst index, spikes/burst, rate)`.  v1's "
      "criterion ('lands in a non-noise cluster') passed while CH landed in a "
      "cluster named *slow tonic* (§2.2).\n")
    A("| type | a | b | c | d | I | cluster | p | CV | BI | spikes/burst | "
      "rate | consistent |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in ctx["placements"]:
        def cell(k):
            d = p.descriptors.get(k)
            if not d:
                return "—"
            mark = "ok" if d["inside"] else "**out**"
            return (f"{d['value']:.3g} in [{d['q25']:.3g}, {d['q75']:.3g}] "
                    f"{mark}")
        verdict = ("exempt" if p.exempt else
                   "yes" if p.consistent else
                   "**no** (within whiskers)" if p.inlier else "**no**")
        A(f"| {p.label} | {p.params[0]:g} | {p.params[1]:g} | {p.params[2]:g} "
          f"| {p.params[3]:g} | {p.drive:.2f} | {p.letter} | "
          f"{p.probability:.3f} | {cell('cv_isi')} | {cell('burst_index')} | "
          f"{cell('spikes_per_burst')} | {cell('rate')} | {verdict} |")
    A("")
    A("TC and RZ are exempt **by name** (§8.3): their published behaviours need "
      "a hyperpolarising step and a timed brief stimulus respectively, neither "
      "reachable at constant drive (§11 item 1).\n")

    A("## 7. Stability (§8.2 — tests that probe something)\n")
    A("| test | varies | mean pairwise ARI | threshold | result |")
    A("|---|---|---|---|---|")
    for key, r in ctx["stab"].runs.items():
        A(f"| {r.name} | {r.description} | {r.mean_ari:.3f} | {r.threshold} | "
          f"{'PASS' if r.passed else 'FAIL'} |")
    A("")
    uns = [r['letter'] for r in ctx['cluster_rows'] if r['unstable']]
    A(f"- UNSTABLE clusters (persistence < "
      f"{cfg.clustering.persistence_floor}): "
      + (", ".join(uns) if uns else "none"))
    A("- v1 §13.2 varied only the UMAP seed, so ARI 0.780 mostly established "
      "that UMAP is reproducible.  The window-shift and step-size tests compare "
      "perturbed against unperturbed features **at matched UMAP seeds**, so "
      "they measure the perturbation rather than seed noise.\n")

    A("## 8. Contiguity in parameter space (v1 §13.3)\n")
    thr = next(iter(ctx["cont"].values())).threshold if ctx["cont"] else float("nan")
    A(f"- single-linkage cut at the {cfg.clustering.contiguity_percentile:g}th "
      f"percentile of the global k={cfg.clustering.contiguity_knn} NN distance "
      f"= {thr:.4f}, in the **5-D** normalised design box (`Δ` is now a design "
      f"axis)")
    mm = [f"{r['letter']} ({r['n_param_components_total']} components, largest "
          f"{r['largest_component_fraction']:.0%})"
          for r in ctx["cluster_rows"] if r["multimodal"]]
    A(f"- MULTIMODAL clusters: " + (", ".join(mm) if mm else "none") + "\n")

    if ctx["s_rows"]:
        A("## 9. `S` class — subthreshold resonators (§3.4)\n")
        pk = np.array([r["resonant_peak_hz"] for r in ctx["s_rows"]])
        A(f"{int(cls.subthreshold.sum())} samples do not spike but fluctuate "
          f"above `V_floor`.  They are **not** clustered (an ISI encoding has "
          f"nothing to encode for them); each is reported with its resonant "
          f"peak frequency, from a host-side rFFT of a `dt_out = "
          f"{cfg.grid.dt_int:g}` ms re-simulation.\n")
        A(f"- resonant peak: median {np.median(pk):.1f} Hz, range "
          f"{pk.min():.1f}–{pk.max():.1f} Hz")
        A(f"- full table in `data/subthreshold_S.csv`\n")

    A(f"## 10. {w.method} → {ctx['other']} containment (v1 §12.4)\n")
    A(f"| {w.method} | size | {ctx['other']} | containment |")
    A("|---|---|---|---|")
    for r in cluster.containment_table(ctx["labels"], ctx["labels_other"]):
        A(f"| {ctx['letters'].get(r['leaf'], r['leaf'])} | {r['size']} | "
          f"{r['eom']} | {r['containment']:.3f} |")
    A("")

    if ctx["sweep_rows"]:
        A("## 11. Selection sweep (v1 §12.4, stage 2 per §3.3)\n")
        A("| stage | representation | nn | mcs | ms | method | k | noise | "
          "rel. validity | score |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(ctx["sweep_rows"], key=lambda r: -r.score)[:20]:
            A(f"| {r.stage} | {r.label} | {r.n_neighbors} | "
              f"{r.min_cluster_size} | {r.min_samples} | {r.method} | "
              f"{r.n_clusters} | {r.noise_fraction:.3f} | "
              f"{r.relative_validity:.4f} | {r.score:.4f} |")
        A(f"\n(top 20 of {len(ctx['sweep_rows'])} configs; full table in "
          f"`sweep_table.csv`)\n")

    A("## 12. Acceptance criteria (§9)\n")
    A("| criterion | result | detail |")
    A("|---|---|---|")
    for c in checks:
        verdict = "n/a" if c["pass"] is None else ("PASS" if c["pass"] else "FAIL")
        A(f"| {c['criterion']} | {verdict} | {c['detail']} |")
    A("")

    A("## 13. Timings\n")
    A("| stage | seconds |")
    A("|---|---|")
    for k, v in sorted(manifest["timings"].items(), key=lambda kv: -kv[1]):
        A(f"| {k} | {v:.1f} |")
    A(f"| **total** | **{manifest['total_seconds']:.1f}** |")
    A("")
    A("v1 spent 346.4 s (37.9%) in `gp_fit` + `acquire` against 85.0 s of "
      "integration (§2.6); C6 removes that stage entirely and spends the "
      "budget on samples and on validation instead.\n")
    return "\n".join(L)


# -------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="izhiclust",
        description="Izhikevich spiking-neuron cluster discovery "
                    "(SPEC.md + IMPROVEMENTS.md)")
    ap.add_argument("--quick", action="store_true",
                    help="N=4096, no sweep, non-blocking health gates")
    ap.add_argument("--no-sweep", dest="sweep", action="store_false")
    ap.add_argument("--sweep-fast", action="store_true",
                    help="drop min_samples=25 from stage 1")
    ap.add_argument("--no-stage2", dest="stage2", action="store_false",
                    help="skip the ISI representation sweep")
    ap.add_argument("--no-tree-reuse", dest="tree_reuse", action="store_false")
    ap.add_argument("--compact-csv", action="store_true",
                    help="omit traces.csv and spike_times.csv")
    ap.add_argument("--gzip-csv", action="store_true")
    ap.add_argument("--no-health-gate", dest="health_gate", action="store_false",
                    help="report the §8.1 gates instead of aborting on them")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--n", type=int, default=None, help="override N")
    a = ap.parse_args(argv)

    cfg = Config(sweep=a.sweep, sweep_fast=a.sweep_fast, stage2=a.stage2,
                 tree_reuse=a.tree_reuse, compact_csv=a.compact_csv,
                 gzip_csv=a.gzip_csv, health_gate=a.health_gate,
                 outdir=a.outdir, datadir=a.datadir, show=a.show)
    if a.quick:
        cfg = cfg.quickened()
    if a.n is not None:
        cfg = replace(cfg, schedule=replace(cfg.schedule, n_total=a.n))

    ctx = run(cfg)
    cfg = ctx["cfg"]
    log = ctx["log"]
    outs = write_outputs(ctx)

    manifest = {
        "spec": "SPEC.md + IMPROVEMENTS.md",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": list(sys.argv),
        "config": ctx["cfg"],
        "derived": {
            "n_total": int(ctx["N"]),
            "dt_int": cfg.grid.dt_int, "L": cfg.grid.L,
            "dt_out": cfg.grid.dt_out, "n_out": cfg.grid.n_out,
            "n_steps": cfg.grid.n_steps,
            "feature_dims": int(ctx["F"].shape[1]),
            "isi_scales": ctx["scales"].as_dict(),
            "isi_weights": dataclasses.asdict(ctx["weights"]),
        },
        "versions": iox.package_versions(),
        "matplotlib_backend": ctx["backend"],
        "convergence_gate": ctx["gate"],
        "rheobase": {
            "n_bracketed": int(np.isfinite(ctx["S"]["i_star"]).sum()),
            "n_no_rheobase": int((ctx["S"]["status"] == rheobase.NO_RHEOBASE).sum()),
            "n_bistable_bracket": int((ctx["S"]["status"] == rheobase.BRACKET_LOW).sum()),
            "i_star_median": float(np.nanmedian(ctx["S"]["i_star"])),
            "drive_median": float(np.median(ctx["S"]["drive"])),
            "class1": int((ctx["S"]["exc_class"] == 1).sum()),
            "class2": int((ctx["S"]["exc_class"] == 2).sum()),
        },
        "classes": ctx["classes"].counts(),
        "health_gates": ctx["health"],
        "pca": {"n_components": int(ctx["P"].shape[1]),
                "cumulative_explained_variance_curve": ctx["evr_curve"],
                "cumulative_at_selected": float(ctx["evr"][-1]),
                "components_for_99": ctx["health"]["pca_components_99"],
                "min_evr_criterion": cfg.clustering.pca_min_evr},
        "selected_config": dataclasses.asdict(ctx["winner"]),
        "clustering": {
            "n_clusters": ctx["n_clusters"],
            "n_clusters_premerge": ctx["n_raw"],
            "n_clusters_other": int(np.unique(
                ctx["labels_other"][ctx["labels_other"] >= 0]).size),
            "other_method": ctx["other"],
            "noise_fraction": ctx["noise_frac"],
            "relative_validity": ctx["rv_total"],
            "letters": {str(k): v for k, v in ctx["letters"].items()},
            "n_refined_dt4": int(ctx["S"]["n_refined"]),
        },
        "merge": ctx["merge_sum"],
        "stability": {k: dataclasses.asdict(v) for k, v in ctx["stab"].runs.items()},
        "persistence": ctx["stab"].persistence,
        "contiguity": {ctx["letters"][k]: dataclasses.asdict(v)
                       for k, v in ctx["cont"].items() if k in ctx["letters"]},
        "canonical_placement": [dataclasses.asdict(p) for p in ctx["placements"]],
        "subthreshold_S": ctx["s_rows"][:64],
        "sweep": [dataclasses.asdict(r) for r in ctx["sweep_rows"]],
        "roundtrip": outs["roundtrip"],
        "timings": dict(ctx["timer"].d),
        "measured": ctx["measured"],
        "total_seconds": time.time() - ctx["t_start"],
    }
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
