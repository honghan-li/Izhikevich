"""CSV / npz / manifest writing (SPEC.md §16) plus the round-trip check."""
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

CHUNK_ROWS = 2048          # §8: chunked to_csv(mode='a')
TRACE_FMT = "%.4g"         # §16.2
SPECTRA_FMT = "%.6g"       # §16.2


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
    """Chunked writer for the two bulk files (§8), via `np.savetxt`.

    Amendment A6 (SPEC §19): §8 assumed `pandas.to_csv` at 60-100 MB/s and
    `numpy.savetxt` at ~10 MB/s.  Measured here (pandas 3.1.1, numpy 2.5.5) it
    is the other way round -- 10 MB/s for `to_csv(float_format=...)` against
    55 MB/s for `savetxt` -- so the bulk writer uses `savetxt` over an open
    handle, keeping the same chunked-append structure and output format.
    """
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
            block = np.empty((sl.stop - sl.start, m + 1), dtype=np.float32)
            block[:, 0] = sample_ids[sl]
            block[:, 1:] = M[sl]
            np.savetxt(fh, block, fmt=fmt, delimiter=",")
    dt = time.time() - t0
    size = os.path.getsize(path) / 2 ** 30
    log(f"  wrote {os.path.basename(path)}: {n} rows, {size:.2f} GiB, {dt:.1f}s "
        f"({size * 1024 / max(dt, 1e-9):.0f} MB/s)")
    return dt


def trace_columns(t_ms: np.ndarray) -> List[str]:
    return [f"t={v:.1f}" for v in t_ms]


def freq_columns(f_hz: np.ndarray) -> List[str]:
    return [f"f={v:.1f}" for v in f_hz]


def band_columns(f_hz: np.ndarray) -> List[str]:
    return [f"f={v:.4g}" for v in f_hz]


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


def roundtrip_check(trace_path: str, F_ref: np.ndarray, plan, cfg: Config,
                    n_rows: int = 1024, exclude: Optional[np.ndarray] = None,
                    log=print) -> Dict[str, float]:
    """Re-read traces.csv and recompute §9; compare with features_bands.csv.

    Evaluated over the samples that actually enter the clustering.  A sample
    §10 quarantines is numerically silent: its banded spectrum is roundoff, so
    its L2-normalised *shape* is arbitrary and the `%.4g` quantisation §16.2
    specifies for traces.csv erases it.  Those rows are reported separately
    rather than mixed into the criterion (§19 A7).
    """
    from .features import band_features, spectra
    if not os.path.exists(trace_path):
        return {"rows": 0, "rel_l2": float("nan")}
    df = pd.read_csv(trace_path, nrows=n_rows)
    ids = df["sample_id"].to_numpy()
    V = df.drop(columns=["sample_id"]).to_numpy(dtype=np.float32)
    F = np.asarray(band_features(spectra(V, plan), plan))
    ref = F_ref[ids]
    nref = np.linalg.norm(ref, axis=1)
    # Numerically silent samples have an all-zero feature vector (their
    # spectral *shape* is undefined); §10 quarantines them and they carry no
    # round-trip information.
    keep = nref > 0
    if exclude is not None:
        keep &= ~np.asarray(exclude)[ids]
    allrel = np.linalg.norm(F - ref, axis=1) / np.maximum(nref, 1e-30)
    rel = allrel[keep]
    other = allrel[~keep]
    res = {"rows": int(keep.sum()), "rows_read": int(len(ids)),
           "rows_excluded": int((~keep).sum()),
           "rel_l2_max": float(rel.max()) if rel.size else float("nan"),
           "rel_l2_mean": float(rel.mean()) if rel.size else float("nan"),
           "rel_l2_p90": float(np.percentile(rel, 90)) if rel.size else float("nan"),
           "rel_l2_p99": float(np.percentile(rel, 99)) if rel.size else float("nan"),
           "excluded_rel_l2_max": float(other.max()) if other.size else float("nan")}
    log(f"  round trip over {res['rows']} clustered rows: max rel L2 "
        f"{res['rel_l2_max']:.2e}, mean {res['rel_l2_mean']:.2e} "
        f"({res['rows_excluded']} quarantined/zero-feature rows excluded, "
        f"max there {res['excluded_rel_l2_max']:.2e})")
    return res
