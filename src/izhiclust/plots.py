"""Figures (SPEC.md §16.1): 2D scatter and per-representative two-panel plots."""
from __future__ import annotations

import os
import warnings
from typing import Dict, Sequence

import numpy as np


def setup_backend(prefer_interactive: bool = True) -> str:
    """Interactive backend when a display exists, else Agg with a warning."""
    import matplotlib
    if prefer_interactive and os.environ.get("DISPLAY"):
        for cand in ("QtAgg", "TkAgg", "GTK3Agg"):
            try:
                matplotlib.use(cand, force=True)
                return cand
            except Exception:
                continue
        warnings.warn("no interactive matplotlib backend available; using Agg")
    elif prefer_interactive:
        warnings.warn("DISPLAY not set; falling back to the Agg backend")
    matplotlib.use("Agg", force=True)
    return "Agg"


def _letters_for(labels: np.ndarray, letters: Dict[int, str]) -> np.ndarray:
    return np.array([letters.get(int(l), "noise") for l in labels], dtype=object)


def scatter_2d(path: str, E2: np.ndarray, labels: np.ndarray,
               letters: Dict[int, str], quarantine: np.ndarray,
               placement_xy: Sequence, flags: Dict[str, Dict[int, bool]],
               names: Dict[int, str], show: bool = True, dpi: int = 200):
    """`placement_xy` is a sequence of (type label, cluster letter, x, y)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    noise = (labels < 0) & ~quarantine
    ax.scatter(E2[noise, 0], E2[noise, 1], s=2, c="0.82", linewidths=0,
               label=f"noise ({noise.sum()})", rasterized=True)
    if quarantine.any():
        ax.scatter(E2[quarantine, 0], E2[quarantine, 1], s=4, c="0.45",
                   marker="x", linewidths=0.4,
                   label=f"Z quarantine ({int(quarantine.sum())})", rasterized=True)

    ids = sorted((cid for cid in np.unique(labels) if cid >= 0),
                 key=lambda c: letters.get(int(c), "zz"))
    cmap = plt.get_cmap("turbo")
    # keep the labels legible as the cluster count grows
    n_ids = len(ids)
    fs = 13 if n_ids <= 30 else 10 if n_ids <= 60 else 7
    for k, cid in enumerate(ids):
        m = labels == cid
        col = cmap((k + 0.5) / max(len(ids), 1))
        ax.scatter(E2[m, 0], E2[m, 1], s=3, color=col, linewidths=0,
                   rasterized=True)
        cx, cy = E2[m, 0].mean(), E2[m, 1].mean()
        tag = letters[int(cid)]
        extra = ""
        if flags.get("unstable", {}).get(int(cid)):
            extra += "*"
        if flags.get("multimodal", {}).get(int(cid)):
            extra += "†"
        ax.annotate(tag + extra, (cx, cy), fontsize=fs, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.75,
                              ec="0.4", lw=0.5))

    for lab, letter, x, y in placement_xy:
        ax.scatter([x], [y], s=150, marker="*", facecolor="none",
                   edgecolor="black", linewidths=1.4, zorder=5)
        ax.annotate(f"{lab}→{letter}", (x, y), fontsize=9,
                    textcoords="offset points", xytext=(8, 8), zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#ffffcc",
                              alpha=0.9, ec="black", lw=0.4))

    handles, lbls = ax.get_legend_handles_labels()
    import matplotlib.lines as mlines
    handles.append(mlines.Line2D([], [], marker="*", color="black", ls="none",
                                 markerfacecolor="none", markersize=12))
    lbls.append("canonical type (§13.1)")
    if any(flags.get("unstable", {}).values()):
        handles.append(mlines.Line2D([], [], ls="none"))
        lbls.append("* UNSTABLE (persistence < 0.6)")
    if any(flags.get("multimodal", {}).values()):
        handles.append(mlines.Line2D([], [], ls="none"))
        lbls.append("† MULTIMODAL (>1 parameter-space component)")
    ax.legend(handles, lbls, loc="upper left", fontsize=8, framealpha=0.9,
              markerscale=3)
    ax.set_xlabel("UMAP-2D 1")
    ax.set_ylabel("UMAP-2D 2")
    ax.set_title(f"Izhikevich firing-pattern clusters — {len(ids)} clusters, "
                 f"N = {labels.size} (I = 10)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    if show:
        try:
            import matplotlib.pyplot as plt2
            plt2.show(block=False)
            plt2.pause(0.5)
        except Exception:
            pass
    return fig


def _annotate_peaks(ax, f: np.ndarray, mag: np.ndarray, n: int = 5,
                    f_hi: float = 1000.0):
    from scipy.signal import find_peaks
    sel = f <= f_hi
    fs, ms = f[sel], mag[sel]
    if ms.size < 8:
        return
    pk, _ = find_peaks(ms, distance=4)
    if pk.size == 0:
        return
    top = pk[np.argsort(-ms[pk])[:n]]
    for i in top:
        ax.annotate(f"{fs[i]:.1f} Hz", (fs[i], ms[i]), fontsize=7,
                    textcoords="offset points", xytext=(3, 4))
        ax.plot([fs[i]], [ms[i]], marker="v", ms=4, color="crimson")


def representative_figure(path: str, *, t_ms: np.ndarray, v: np.ndarray,
                          f_hz: np.ndarray, mag: np.ndarray, title: str,
                          t_window: tuple, dpi: int = 200):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.0, 7.2))
    ax0.plot(t_ms, v, lw=0.4, color="#1f4e79")
    ax0.axvspan(t_window[0], t_window[1], color="#ffd27f", alpha=0.25,
                label="analysis window")
    ax0.set_xlim(t_ms[0], t_ms[-1])
    ax0.set_xlabel("t (ms)")
    ax0.set_ylabel("v (mV)")
    ax0.legend(loc="lower right", fontsize=8)

    sel = f_hz <= 500.0
    ax1.plot(f_hz[sel], mag[sel], lw=0.6, color="#1f4e79")
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel("|rFFT| (deconvolved)")
    ax1.set_xlim(0, 500)
    _annotate_peaks(ax1, f_hz, mag, n=5, f_hi=500.0)

    ins = ax1.inset_axes([0.55, 0.45, 0.42, 0.5])
    pos = f_hz > 0
    ins.loglog(f_hz[pos], np.maximum(mag[pos], 1e-12), lw=0.5, color="#7a4e1f")
    ins.set_xlim(0.5, 1000)
    ins.tick_params(labelsize=6)
    ins.set_title("log f, 0.5–1000 Hz", fontsize=7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def cluster_trace_stack(path: str, t_ms: np.ndarray, traces: Sequence[np.ndarray],
                        labels: Sequence[str], title: str, dpi: int = 200):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(traces), 1, figsize=(11.0, 2.2 * len(traces)),
                             sharex=True, sharey=True)
    if len(traces) == 1:
        axes = [axes]
    for ax, v, lab in zip(axes, traces, labels):
        ax.plot(t_ms, v, lw=0.4, color="#1f4e79")
        ax.set_ylabel("v (mV)")
        ax.text(0.995, 0.92, lab, transform=ax.transAxes, ha="right",
                va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8,
                          ec="0.6", lw=0.4))
    axes[-1].set_xlabel("t (ms)")
    axes[0].set_xlim(t_ms[0], t_ms[-1])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
