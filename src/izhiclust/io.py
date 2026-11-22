"""CSV / npz / manifest writing (v1 §16, amended by §3.5) and the round trip."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config

CHUNK_ROWS = 2048          # v1 §8: chunked append
TRACE_FMT = "%.4g"         # v1 §16.2
# `%.9g` round-trips a float32 exactly, which is what the kernel writes; the
# §9 round-trip criterion is 1e-6 relative, so the file must not quantise.
SPIKE_FMT = "%.9g"


def ensure_dirs(cfg: Config) -> Dict[str, str]:
    out = os.path.abspath(cfg.outdir)
    data = os.path.abspath(cfg.datadir)
    os.makedirs(out, exist_ok=True)
    os.makedirs(data, exist_ok=True)
    os.makedirs(os.path.join(data, "traces"), exist_ok=True)
    return {"output": out, "data": data, "traces": os.path.join(data, "traces")}


def write_matrix_csv(path: str, M: np.ndarray, sample_ids: np.ndarray,
                     col_names: Sequence[str], float_format: str,
                     gzip: bool = False, chunk: int = CHUNK_ROWS,
                     log=print) -> float:
    """Chunked writer via `np.savetxt` (v1 §19 A6: it is the faster of the two)."""
    import gzip as gziplib
    if gzip and not path.endswith(".gz"):
        path = path + ".gz"
    t0 = time.time()
    n, m = M.shape
    fmt = ["%d"] + [float_format] * m
    opener = gziplib.open if path.endswith(".gz") else open
    with opener(path, "wt", newline="\n") as fh:
        fh.write("sample_id," + ",".join(col_names) + "\n")
        for s in range(0, n, chunk):
            sl = slice(s, min(s + chunk, n))
            block = np.empty((sl.stop - sl.start, m + 1), dtype=np.float64)
            block[:, 0] = sample_ids[sl]
            block[:, 1:] = M[sl]
            np.savetxt(fh, block, fmt=fmt, delimiter=",")
    dt = time.time() - t0
    size = os.path.getsize(path) / 2 ** 30
    log(f"  wrote {os.path.basename(path)}: {n} rows, {size:.2f} GiB, {dt:.1f}s "
        f"({size * 1024 / max(dt, 1e-9):.0f} MB/s)")
    return dt


def write_spike_times_csv(path: str, spikes, sample_ids: np.ndarray,
                          gzip: bool = False, log=print) -> float:
    """§3.5 `spike_times.csv`: ragged, one row per sample.

    Columns `sample_id, n_spikes, spike_times_ms`, the third field a
    space-separated list so the file stays a valid rectangular CSV while the
    payload is ragged (assumption V2-A11).
    """
    import gzip as gziplib
    if gzip and not path.endswith(".gz"):
        path = path + ".gz"
    t0 = time.time()
    opener = gziplib.open if path.endswith(".gz") else open
    with opener(path, "wt", newline="\n") as fh:
        fh.write("sample_id,n_spikes,spike_times_ms\n")
        for i in range(spikes.n):
            t = spikes.get(i)
            fh.write(f"{int(sample_ids[i])},{t.size}," + " ".join(
                SPIKE_FMT % v for v in t) + "\n")
    dt = time.time() - t0
    size = os.path.getsize(path) / 2 ** 30
    log(f"  wrote {os.path.basename(path)}: {spikes.n} rows, {size:.2f} GiB, "
        f"{dt:.1f}s")
    return dt


def read_spike_times_csv(path: str, n_rows: Optional[int] = None):
    """Inverse of `write_spike_times_csv`; returns (ids, SpikeTrains)."""
    from .simulate import SpikeTrains
    df = pd.read_csv(path, nrows=n_rows)
    ids = df["sample_id"].to_numpy()
    parts = [np.fromstring(s, sep=" ", dtype=np.float32) if isinstance(s, str)
             else np.zeros(0, dtype=np.float32)
             for s in df["spike_times_ms"].fillna("")]
    counts = np.array([p.size for p in parts], dtype=np.int64)
    off = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    times = (np.concatenate(parts) if parts else np.zeros(0, np.float32))
    return ids, SpikeTrains(times.astype(np.float32), off,
                            np.zeros(counts.size, dtype=bool))


def trace_columns(t_ms: np.ndarray) -> List[str]:
    return [f"t={v:.1f}" for v in t_ms]


def freq_columns(f_hz: np.ndarray) -> List[str]:
    return [f"f={v:.1f}" for v in f_hz]


# ----------------------------------------------------------------- manifest


def _jsonable(o):
    if is_dataclass(o) and not isinstance(o, type):
        return {k: _jsonable(v) for k, v in asdict(o).items()}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else str(v)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def package_versions() -> Dict[str, str]:
    import importlib.metadata as md
    names = ["numpy", "scipy", "pandas", "scikit-learn", "umap-learn",
             "hdbscan", "cupy", "matplotlib", "tqdm", "pytest"]
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = "missing"
    try:
        import cupy as cp
        out["cuda_runtime"] = str(cp.cuda.runtime.runtimeGetVersion())
        out["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        out["compute_capability"] = cp.cuda.Device(0).compute_capability
    except Exception:
        pass
    try:
        out["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return out


def write_manifest(path: str, payload: Dict, log=print):
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=False)
    log(f"  wrote {os.path.basename(path)}")


# ------------------------------------------------------------- round trip


def roundtrip_check(spike_path: str, F_ref: np.ndarray, cfg: Config, scales,
                    weights, n_rows: int = 1024,
                    include: Optional[np.ndarray] = None, log=print) -> Dict:
    """§9: spike times -> features must reproduce `features_isi.csv` to 1e-6.

    Re-reads `spike_times.csv`, recomputes the §3.2 steady-state blocks with
    the recorded dataset-wide scales and block weights, and compares with the
    feature matrix that was written.
    """
    from . import isi as isim
    if not os.path.exists(spike_path):
        return {"rows": 0, "rel_l2_max": float("nan")}
    ids, spikes = read_spike_times_csv(spike_path, n_rows=n_rows)
    g = cfg.grid
    s = isim.isi_set(spikes, g.t_transient, g.t_steady_end)
    blocks = isim.compute_blocks(s, cfg, h=weights.h)
    F = isim._l2_rows(isim.steady_vector(blocks, scales, weights))
    ref = F_ref[ids]
    nref = np.linalg.norm(ref, axis=1)
    keep = nref > 0
    if include is not None:
        keep &= np.asarray(include)[ids]
    allrel = np.linalg.norm(F - ref, axis=1) / np.maximum(nref, 1e-30)
    rel = allrel[keep]
    other = allrel[~keep]
    res = {"rows": int(keep.sum()), "rows_read": int(len(ids)),
           "rows_excluded": int((~keep).sum()),
           "rel_l2_max": float(rel.max()) if rel.size else float("nan"),
           "rel_l2_mean": float(rel.mean()) if rel.size else float("nan"),
           "rel_l2_p99": float(np.percentile(rel, 99)) if rel.size else float("nan"),
           "excluded_rel_l2_max": float(other.max()) if other.size else float("nan")}
    log(f"  round trip over {res['rows']} clustered rows: max rel L2 "
        f"{res['rel_l2_max']:.2e}, mean {res['rel_l2_mean']:.2e} "
        f"({res['rows_excluded']} held-out rows excluded)")
    return res
