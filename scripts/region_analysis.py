#!/usr/bin/env python
"""Region analysis of the ISI embedding -- reproduces output/region_report.md.

Cuts the 2-D UMAP into the nine hand-picked regions (the "neuron" with its cell
body, three dendrites, axon and axon terminals; the "blob" with its great,
middle and end lobes), measures the dynamical descriptors that distinguish them,
and writes the figures the report cites.

    pixi run python scripts/region_analysis.py

Outputs
    data/regions.csv                 per-sample region + descriptors
    output/regions_map.png
    output/regions_return_maps.png
    output/regions_phase.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"

LO, HI = 500.0, 2500.0          # steady-state analysis window (ms)
PERIOD_RTOL, PERIOD_MAX = 0.02, 16
GRID_H = 0.30                   # UMAP units, for the connected-component cut
SOMA_BOX = (-6.35, -4.85, -1.60, 0.45)
AD_SEP = 8.70                   # x - y cut between the middle and end lobes

TRUNK = ["AR", "BB", "AB", "AA", "Y", "BO", "AD", "BI", "BM", "BV"]
TERM = ["BA", "BR", "CB", "CC", "AP", "AQ", "CA", "BJ", "BY"]

ORDER = ["great blob", "middle blob", "end blob", "dendrite T", "axon",
         "axon terminals", "dendrite C", "dendrite B (onion)", "cell body"]
COL = {"cell body": "#111111", "dendrite C": "#e8710a", "dendrite B (onion)": "#00a3a3",
       "dendrite T": "#8c564b", "axon": "#1f77b4", "axon terminals": "#2ca02c",
       "middle blob": "#7b3fa0", "end blob": "#c2185b", "great blob": "#777777",
       "neuron (other noise)": "#d8d8d8"}
SEEDS = {24544: "RS", 63776: "IB", 20072: "CH", 61538: "FS",
         63400: "LTS", 56800: "TC", 19410: "RZ"}


# ------------------------------------------------------------------ descriptors

def orbit_period(isi: np.ndarray) -> int:
    """Smallest p with isi[i] ~= isi[i+p] for all i; 0 if the orbit never closes."""
    n = isi.size
    if n < 6:
        return 0
    tol = PERIOD_RTOL * isi.mean()
    for p in range(1, PERIOD_MAX + 1):
        if n - p < 3:
            break
        if np.abs(isi[p:] - isi[:-p]).max() <= tol:
            return p
    return 0


def spectral_flatness(t: np.ndarray, dt: float = 0.5) -> float:
    """Flatness of the delta spike train over 2-500 Hz; 1 = white, 0 = a line."""
    nb = int((HI - LO) / dt)
    x = np.zeros(nb)
    np.add.at(x, np.clip(((t - LO) / dt).astype(int), 0, nb - 1), 1.0)
    x -= x.mean()
    f = np.fft.rfftfreq(nb, dt * 1e-3)
    P = np.maximum(np.abs(np.fft.rfft(x * np.hanning(nb))) ** 2, 1e-20)[(f >= 2) & (f <= 500)]
    return float(np.exp(np.log(P).mean()) / P.mean())


def descriptors() -> pd.DataFrame:
    """One row per sample, from the exact spike times."""
    rows = []
    with open(DATA / "spike_times.csv", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for rec in reader:
            sid, payload = int(rec[0]), rec[2]
            t = np.fromstring(payload, sep=" ") if payload else np.empty(0)
            steady = t[(t >= LO) & (t < HI)]
            onset = t[t < LO]
            row = dict(sample_id=sid, n_isi_win=max(steady.size - 1, 0),
                       n_spk_100ms=int((t < 100).sum()))
            if steady.size > 1:
                isi = np.diff(steady)
                mean = isi.mean()
                row.update(mean_isi_steady=mean, cv=isi.std() / mean,
                           period=orbit_period(isi),
                           log_range=np.log10(isi.max() / isi.min()),
                           flatness=spectral_flatness(steady))
                lam = np.clip(np.log10(isi / mean), -1.75, 1.0)
                pr = np.histogram(lam, bins=32, range=(-1.75, 1.0))[0]
                pr = pr[pr > 0] / isi.size
                row["log_isi_entropy"] = float(-(pr * np.log(pr)).sum())
                if onset.size > 1:
                    row["log_first_over_steady"] = np.log10(np.diff(onset)[0] / mean)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ regions

def assign(df: pd.DataFrame) -> pd.DataFrame:
    """Split the embedding into super-structures, then into the nine regions."""
    has = df.umap2_0.notna().values
    x, y = df.umap2_0.values[has], df.umap2_1.values[has]
    ix = ((x - x.min()) / GRID_H).astype(int)
    iy = ((y - y.min()) / GRID_H).astype(int)
    grid = np.zeros((ix.max() + 1, iy.max() + 1), bool)
    grid[ix, iy] = True
    comp = ndimage.label(grid, structure=np.ones((3, 3)))[0][ix, iy]

    df["side"] = "held-out (W)"
    df.loc[has, "side"] = np.where(comp == np.bincount(comp).argmax(), "blob", "neuron")

    letter, reg = df.letter.values, pd.Series("unassigned", index=df.index)
    nb = (df.side == "neuron").values
    reg[nb & (letter == "C")] = "dendrite C"
    reg[nb & (letter == "B")] = "dendrite B (onion)"
    reg[nb & np.isin(letter, ["T", "BG"])] = "dendrite T"
    reg[nb & np.isin(letter, TRUNK)] = "axon"
    reg[nb & np.isin(letter, TERM)] = "axon terminals"
    x0, x1, y0, y1 = SOMA_BOX
    reg[nb & (letter == "noise") & df.umap2_0.between(x0, x1) & df.umap2_1.between(y0, y1)] = "cell body"
    rest = nb & (reg == "unassigned").values
    reg[rest & (df.umap2_1 > 14).values] = "axon terminals"
    reg[(reg == "unassigned").values & nb] = "neuron (other noise)"

    bb = (df.side == "blob").values
    end = ((df.umap2_0 - df.umap2_1) > AD_SEP).values
    reg[bb & (letter == "A") & ~end] = "middle blob"
    reg[bb & (letter == "A") & end] = "end blob"
    reg[bb & (letter != "A")] = "great blob"
    df["region"] = reg.values
    return df


# ------------------------------------------------------------------ figures

def fig_map(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["umap2_0"])
    fig, ax = plt.subplots(figsize=(15, 11), dpi=130)
    for r in ORDER + ["neuron (other noise)"]:
        g = d[d.region == r]
        ax.scatter(g.umap2_0, g.umap2_1, s=1.4, c=COL[r], lw=0)
    ann = {"cell body": (-5.3, 2.6), "dendrite C": (-11.0, -3.0),
           "dendrite B (onion)": (-3.4, -8.6), "dendrite T": (-3.0, 1.2),
           "axon": (-7.9, 6.0), "axon terminals": (-4.0, 21.4),
           "middle blob": (-0.6, -2.0), "end blob": (8.6, -12.5), "great blob": (16.0, 6.0)}
    for r, pos in ann.items():
        ax.annotate(f"{r}\n(n={(d.region == r).sum():,})", pos, fontsize=11, weight="bold",
                    color=COL[r], ha="center",
                    bbox=dict(fc="white", ec=COL[r], alpha=.85, boxstyle="round,pad=0.3"))
    for sid, lbl in SEEDS.items():
        row = df[df.sample_id == sid].iloc[0]
        ax.plot(row.umap2_0, row.umap2_1, "*", ms=16, mfc="yellow", mec="k", mew=1.2)
        ax.annotate(lbl, (row.umap2_0, row.umap2_1), xytext=(6, 6),
                    textcoords="offset points", fontsize=10, weight="bold")
    ax.set(xlabel="UMAP-2D 1", ylabel="UMAP-2D 2",
           title=f"Hand-picked regions of the ISI embedding (N = {len(d):,} clustered + noise)")
    ax.grid(alpha=.15)
    fig.savefig(OUT / "regions_map.png", bbox_inches="tight")
    plt.close(fig)


def fig_return_maps(df: pd.DataFrame) -> None:
    """One representative per region: raster plus the lambda_n return map (block B)."""
    modal = {"great blob": (1,), "middle blob": (1,), "end blob": (1,), "dendrite T": (2,),
             "axon": (2,), "axon terminals": (2,), "dendrite C": (3,),
             "dendrite B (onion)": (4, 5, 6), "cell body": (0,)}
    reps = {}
    for r in ORDER:
        g = df[(df.region == r) & df.log_range.notna() & (df.n_isi_win > 25)]
        g = g[g.period.isin(modal[r])]
        v = np.c_[np.log10(g.rate), g.log_range]
        tgt = np.array([np.log10(g.rate.median()), g.log_range.median()])
        reps[r] = int(g.sample_id.values[np.argmin(((v - tgt) ** 2).sum(1))])

    trains, want = {}, set(reps.values())
    with open(DATA / "spike_times.csv", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for rec in reader:
            sid = int(rec[0])
            if sid in want:
                t = np.fromstring(rec[2], sep=" ")
                trains[sid] = t[(t >= LO) & (t < HI)]
                if len(trains) == len(want):
                    break

    fig, axs = plt.subplots(3, 6, figsize=(21, 10), dpi=140,
                            gridspec_kw=dict(width_ratios=[2.1, 1] * 3))
    for k, r in enumerate(ORDER):
        row, col = divmod(k, 3)
        axr, axm = axs[row, col * 2], axs[row, col * 2 + 1]
        sid = reps[r]
        isi = np.diff(trains[sid])
        s = df[df.sample_id == sid].iloc[0]
        span = float(np.clip(30 * isi.mean() * (s.period if s.period > 1 else 1), 120, 1200))
        axr.vlines(trains[sid][trains[sid] < LO + span] - LO, 0, 1, color=COL[r], lw=1.0)
        axr.set(xlim=(0, span), yticks=[], xlabel="time after 500 ms (ms)")
        per = "aperiodic" if s.period == 0 else f"period {int(s.period)}"
        axr.set_title(f"{r}   #{sid}\n{s.rate:.0f} Hz · CV {s.cv:.2f} · {per} · "
                      rf"$I_\mathrm{{max}}/I_\mathrm{{min}}$ = {10 ** s.log_range:.1f}", fontsize=9)
        axr.tick_params(labelsize=7)
        lam = np.log10(isi / isi.mean())
        axm.plot(lam[:-1], lam[1:], ".", ms=4, color=COL[r], alpha=.75)
        axm.plot([-1.85, 1.1], [-1.85, 1.1], "k--", lw=.6)
        axm.set(xlim=(-1.85, 1.1), ylim=(-1.85, 1.1), xlabel=r"$\lambda_n$",
                ylabel=r"$\lambda_{n+1}$")
        axm.set_aspect("equal")
        axm.tick_params(labelsize=7)
        axm.grid(alpha=.2)
    fig.suptitle(r"Steady-state spike trains and ISI return maps "
                 r"($\lambda_n=\log_{10}(I_n/\bar I)$ — feature block B)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "regions_return_maps.png", bbox_inches="tight")
    plt.close(fig)


def fig_phase(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.4), dpi=130)
    sub = df[df.side.isin(["neuron", "blob"])].sample(30000, random_state=0)
    nb = sub.side == "neuron"
    axs[0].scatter(sub.c[~nb], sub.d[~nb], s=2, c="#9aa0a6", lw=0, label="blob (period-1 tonic)")
    axs[0].scatter(sub.c[nb], sub.d[nb], s=2, c="#d62728", lw=0, label="neuron (period ≥ 2 / chaotic)")
    axs[0].axhline(1.5, color="k", ls="--", lw=1)
    axs[0].axvline(-52, color="k", ls="--", lw=1)
    axs[0].set(xlabel="c (mV)", ylabel="d", title="(a) where the two super-structures live in (c, d)")
    axs[0].legend(markerscale=6, loc="upper left")

    nn = df[(df.side == "neuron") & (df.period > 0)].copy()
    nn["beta"] = nn.d / (nn.c + 62)
    nn["bb"] = pd.cut(nn.beta, [0, .1, .2, .3, .4, .6, .9, 1.5])
    klass = np.select([nn.period == 2, nn.period == 3, nn.period >= 4], ["2", "3", ">=4"], "0")
    pc = pd.crosstab(nn.bb, klass, normalize="index")
    pc[[c for c in ["2", "3", ">=4"] if c in pc]].plot(
        kind="bar", stacked=True, ax=axs[1], color=["#1f77b4", "#e8710a", "#00a3a3"], width=.85)
    axs[1].set(xlabel=r"$\beta = d/(c+62)$", ylabel="fraction of samples",
               title=r"(b) orbit period vs the bifurcation parameter $\beta$")
    axs[1].legend(title="ISI orbit period", fontsize=9)
    axs[1].tick_params(axis="x", labelrotation=30, labelsize=8)

    for r in ORDER[3:]:
        g = df[df.region == r]
        axs[2].scatter(g.rate, 10 ** g.log_range, s=2, c=COL[r], lw=0, label=r)
    axs[2].set(xscale="log", yscale="log", xlabel="firing rate (Hz)",
               ylabel=r"$I_{max}/I_{min}$ within the orbit",
               title="(c) the neuron side: rate × ISI-contrast plane")
    for thr in (3.5, 32):
        axs[2].axhline(thr, color="k", ls="--", lw=.8)
    axs[2].legend(markerscale=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "regions_phase.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main

def main() -> None:
    df = pd.read_csv(DATA / "samples.csv").merge(descriptors(), on="sample_id", how="left")
    df = assign(df)
    df.to_csv(DATA / "regions.csv", index=False)

    keep = df[df.side.isin(["neuron", "blob"])]
    summary = keep.groupby("region").agg(
        n=("rate", "size"), rate=("rate", "median"), cv=("cv", "median"),
        period=("period", "median"), aperiodic=("period", lambda s: (s == 0).mean()),
        contrast=("log_range", lambda s: 10 ** s.median()),
        entropy=("log_isi_entropy", "median"), flatness=("flatness", "median"),
        a=("a", "median"), b=("b", "median"), c=("c", "median"),
        d=("d", "median"), delta=("delta", "median")).reindex(ORDER)
    print(summary.round(4).to_string())

    fig_map(df)
    fig_return_maps(df)
    fig_phase(df)
    print(f"\nwrote {DATA / 'regions.csv'} and three figures to {OUT}")


if __name__ == "__main__":
    main()
