"""§17.6 -- GP hyperparameter recovery and the exact rank-1 variance update."""
import numpy as np
import pytest

from izhiclust.gp import ARDGP


def _draw(n=1200, d=4, ell=(0.15, 0.5, 1.0, 0.35), seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    ell = np.asarray(ell)
    D = ((X[:, None, :] - X[None, :, :]) ** 2 / ell ** 2).sum(-1)
    K = np.exp(-0.5 * D) + 1e-6 * np.eye(n)
    y = np.linalg.cholesky(K) @ rng.standard_normal(n)
    return X, (y - y.mean()) / y.std(), ell


@pytest.mark.gpu
def test_recovers_ard_lengthscales(gpu):
    X, y, ell = _draw()
    gp = ARDGP(use_gpu=True, dtype=np.float64).fit(X, y)
    rel = np.abs(gp.lengthscales - ell) / ell
    assert rel.max() < 0.20, (gp.lengthscales, ell, rel)


@pytest.mark.gpu
def test_rank1_variance_update_matches_full_recompute(gpu):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular

    X, y, _ = _draw(n=400)
    gp = ARDGP(use_gpu=True, dtype=np.float64, jitter=1e-10).fit(X, y)
    rng = np.random.default_rng(7)
    Q = rng.random((60, 4))
    A = gp.basis(Q)
    V0 = gp.signal_var - (A * A).sum(axis=0)
    s = 11
    cov = gp.kernel_cross(Q, Q[s:s + 1]).ravel() - A.T @ A[:, s]
    u = cov / cp.sqrt(V0[s])
    V1 = V0 - u * u

    n = X.shape[0]
    Xa = np.vstack([X, Q[s]])
    Kaa = gp.kernel_cross(Xa, Xa)
    idx = cp.arange(n)
    Kaa[idx, idx] += gp.noise_var + 1e-10
    L = cp.linalg.cholesky(Kaa)
    B = solve_triangular(L, gp.kernel_cross(Xa, Q), lower=True)
    Vfull = gp.signal_var - (B * B).sum(axis=0)
    assert float(cp.max(cp.abs(V1 - Vfull))) < 1e-10


@pytest.mark.gpu
def test_mean_gradient_matches_finite_difference(gpu):
    X, y, _ = _draw(n=300)
    gp = ARDGP(use_gpu=True, dtype=np.float64).fit(X, y)
    rng = np.random.default_rng(3)
    Q = rng.random((8, 4)) * 0.8 + 0.1
    _, gn = gp.mean_and_grad_norm(Q)
    gn = np.asarray(gn.get())
    h = 1e-6
    for i in range(Q.shape[0]):
        g = np.zeros(4)
        for d in range(4):
            qp, qm = Q[i].copy(), Q[i].copy()
            qp[d] += h
            qm[d] -= h
            g[d] = (float(gp.mean(qp[None, :])[0]) - float(gp.mean(qm[None, :])[0])) / (2 * h)
        assert abs(np.linalg.norm(g) - gn[i]) < 1e-4 * max(1.0, gn[i])
