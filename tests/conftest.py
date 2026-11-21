import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a working CuPy/CUDA device")


@pytest.fixture(scope="session")
def gpu():
    try:
        import cupy as cp
        cp.arange(4).sum()
    except Exception as exc:                     # pragma: no cover
        pytest.skip(f"no usable GPU: {exc}")
    return cp


@pytest.fixture(scope="session")
def sim(gpu):
    from izhiclust.config import DEFAULT
    from izhiclust.simulate import Simulator
    return Simulator(DEFAULT)
