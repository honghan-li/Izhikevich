"""A numpy DP5 stepper built from the tableau constants parsed out of
kernel.cu, so the host-side reference and the device kernel cannot drift apart.
"""
from __future__ import annotations

import os
import re

import numpy as np

KERNEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "src", "izhiclust", "kernel.cu")


def tableau() -> dict:
    src = open(KERNEL).read()
    out = {}
    for name, expr in re.findall(r"#define\s+([ABE]\d\d?)\s+(\([^\n]*\))\n", src):
        out[name] = float(eval(expr, {"__builtins__": {}}))
    return out


def dp5_step(f, y, h, T):
    k1 = f(y)
    k2 = f(y + h * (T["A21"] * k1))
    k3 = f(y + h * (T["A31"] * k1 + T["A32"] * k2))
    k4 = f(y + h * (T["A41"] * k1 + T["A42"] * k2 + T["A43"] * k3))
    k5 = f(y + h * (T["A51"] * k1 + T["A52"] * k2 + T["A53"] * k3 + T["A54"] * k4))
    k6 = f(y + h * (T["A61"] * k1 + T["A62"] * k2 + T["A63"] * k3 + T["A64"] * k4
                    + T["A65"] * k5))
    yn = y + h * (T["B1"] * k1 + T["B3"] * k3 + T["B4"] * k4 + T["B5"] * k5
                  + T["B6"] * k6)
    k7 = f(yn)
    err = h * (T["E1"] * k1 + T["E3"] * k3 + T["E4"] * k4 + T["E5"] * k5
               + T["E6"] * k6 + T["E7"] * k7)
    return yn, err


def integrate(f, y0, h, n, T=None):
    T = T or tableau()
    y = np.asarray(y0, dtype=np.float64)
    for _ in range(n):
        y, _ = dp5_step(f, y, h, T)
    return y
