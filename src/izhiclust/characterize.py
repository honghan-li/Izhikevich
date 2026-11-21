"""Interpretable per-cluster names from the firing diagnostics (SPEC.md §14.2)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import Config


def _rate_word(rate: float, nm) -> str:
    if rate < nm.rate_slow:
        return "slow"
    if rate < nm.rate_fast:
        return "medium"
    return "fast"


def name_from_medians(med: Dict[str, float], cfg: Config, p_floor: float) -> str:
    """Name one cluster from its diagnostic medians."""
    nm = cfg.naming
    rate = med.get("rate", 0.0)
    ac = med.get("ac_power", 0.0)
    bi = med.get("burst_index", 0.0)
    cv = med.get("cv_isi", 0.0)
    ad = med.get("adapt_index", 0.0)
    spb = med.get("spikes_per_burst", 1.0)

    n_spikes = med.get("n_spikes", 1.0)
    if rate < nm.rate_floor and n_spikes <= 0:
        # no spikes at all: a real spectral peak (resonance) or the floor
        core = "subthreshold/resonant" if ac > p_floor else "silent"
    elif rate < nm.rate_floor:
        # §19 A9: spiking, but below §6's 4 Hz spectral-estimate floor -- not
        # the same thing as subthreshold, which §14.2's table conflates
        core = ("very slow bursting" if bi > nm.burst_index else "very slow tonic")
    elif bi > nm.burst_index:
        ib_rate = rate / max(spb, 1.0)
        core = (f"{_rate_word(rate, nm)} bursting "
                f"({spb:.1f} spikes/burst, {ib_rate:.0f} Hz interburst)")
    else:
        core = f"{_rate_word(rate, nm)} tonic"

    # §19 A11: §14.2 names a cluster from medians, which hides a cluster that
    # mixes bursting and tonic members at the same fundamental -- the common
    # case here, since a 17 Hz burst train and a 17 Hz tonic train share a
    # fundamental and most of their comb.
    fb = med.get("frac_bursting", None)
    if fb is not None and 0.2 <= fb <= 0.8:
        core += f" [mixed: {fb:.0%} bursting]"

    tags = []
    if n_spikes > 0:
        if ad > nm.adapt:
            tags.append("+adapting")
        elif ad < -nm.adapt:
            tags.append("+accelerating")
        if cv is not None and np.isfinite(cv) and cv > nm.cv_irregular and bi <= nm.burst_index:
            tags.append("+irregular")
    return ", ".join([core] + tags)


def describe(letter: str, med: Dict[str, float], size: int, cfg: Config,
             p_floor: float) -> str:
    name = name_from_medians(med, cfg, p_floor)
    cv = med.get("cv_isi", float("nan"))
    return (f"{letter} — {name} (n={size}, rate {med.get('rate', 0.0):.0f} Hz, "
            f"CV {cv:.2f})" if np.isfinite(cv) else
            f"{letter} — {name} (n={size}, rate {med.get('rate', 0.0):.0f} Hz)")
