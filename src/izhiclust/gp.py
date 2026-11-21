"""Hand-rolled ARD-RBF Gaussian process in CuPy (SPEC.md §11.2).

Six independent GPs (one per PCA output) with a constant (zero, because the
outputs are standardised) mean and Gaussian noise.  Hyperparameters are fitted
by L-BFGS-B on the log marginal likelihood with analytic gradients over the
six log parameters (4 lengthscales, signal variance, noise variance).

Default dtype is float32: the fit is dominated by a 3000x3000 Cholesky, which
is ~64x faster in FP32 than FP64 on this device, and the noise-variance lower
bound keeps the condition number inside FP32's reach.  Pass dtype=float64 when
exactness matters (the rank-1 update test does).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.optimize import minimize

N_HYP = 6   # 4 log lengthscales + log signal var + log noise var


def _get_xp(use_gpu: bool):
    if use_gpu:
        import cupy as cp
        return cp
    return np


def _cholesky(xp, K):
    return xp.linalg.cholesky(K)


def _solve_tri(xp, L, B, trans=0):
    if xp is np:
        from scipy.linalg import solve_triangular
        return solve_triangular(L, B, lower=True, trans=trans)
    from cupyx.scipy.linalg import solve_triangular
    return solve_triangular(L, B, lower=True, trans=trans)


@dataclass
class GPFit:
    theta: np.ndarray          # log hyperparameters
    nll: float
    n_train: int


class ARDGP:
    def __init__(self, use_gpu: bool = True, dtype=np.float32,
                 jitter: Optional[float] = None, maxiter: int = 60):
        self.xp = _get_xp(use_gpu)
        self.dtype = dtype
        single = np.dtype(dtype) == np.float32
        # FP32 needs a larger jitter and noise floor to keep the Cholesky
        # well conditioned; FP64 can go much lower.
        self.jitter = (1e-5 if single else 1e-8) if jitter is None else jitter
        self.noise_lo = 1e-4 if single else 1e-8
        self.maxiter = maxiter
        self.theta: Optional[np.ndarray] = None
        self._X = None
        self._D = None
        self._L = None
        self._alpha = None
        self.fit_info: Optional[GPFit] = None

    # ---------------------------------------------------------------- kernel
    def _sqdiff(self, X):
        """(d, n, n) pairwise squared differences per input dimension."""
        xp = self.xp
        Xd = X.T                                # (d, n)
        return (Xd[:, :, None] - Xd[:, None, :]) ** 2

    def _corr(self, D, ell):
        xp = self.xp
        inv = (1.0 / (ell ** 2)).astype(self.dtype)
        Q = xp.zeros(D.shape[1:], dtype=self.dtype)
        for d in range(D.shape[0]):
            Q += D[d] * inv[d]
        return xp.exp(-0.5 * Q)

    def _unpack(self, theta):
        ell = np.exp(theta[:4])
        sf2 = math.exp(theta[4])
        sn2 = math.exp(theta[5])
        return ell, sf2, sn2

    # ------------------------------------------------------------------- NLL
    def _nll_grad(self, theta, y):
        xp = self.xp
        n = y.shape[0]
        ell, sf2, sn2 = self._unpack(theta)
        E = self._corr(self._D, xp.asarray(ell, dtype=self.dtype))
        K = sf2 * E
        idx = xp.arange(n)
        K[idx, idx] += sn2 + self.jitter
        try:
            L = _cholesky(xp, K)
        except Exception:
            return 1e10, np.zeros(N_HYP)
        if not bool(xp.isfinite(L[0, 0])):
            return 1e10, np.zeros(N_HYP)
        yv = y[:, None]
        v = _solve_tri(xp, L, yv)
        alpha = _solve_tri(xp, L, v, trans=1)
        logdet = 2.0 * float(xp.log(xp.diag(L)).sum())
        nll = 0.5 * float((yv * alpha).sum()) + 0.5 * logdet + 0.5 * n * math.log(2 * math.pi)
        if not math.isfinite(nll):
            return 1e10, np.zeros(N_HYP)

        # W = Kinv - alpha alpha^T ; grad = 0.5 * sum(W * dK/dtheta)
        eye = xp.eye(n, dtype=self.dtype)
        Linv = _solve_tri(xp, L, eye)
        Kinv = Linv.T @ Linv
        W = Kinv - alpha @ alpha.T
        g = np.zeros(N_HYP)
        for d in range(4):
            g[d] = 0.5 * float((W * (sf2 * E * self._D[d])).sum()) / (ell[d] ** 2)
        g[4] = 0.5 * float((W * (sf2 * E)).sum())
        g[5] = 0.5 * float(xp.diag(W).sum()) * sn2
        del E, K, L, Linv, Kinv, W, v, alpha
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
        return nll, g

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray, theta0: Optional[np.ndarray] = None):
        xp = self.xp
        self._X = xp.asarray(np.ascontiguousarray(X), dtype=self.dtype)
        self._D = self._sqdiff(self._X)
        yd = xp.asarray(np.ascontiguousarray(y), dtype=self.dtype)

        if theta0 is None:
            theta0 = np.array([math.log(0.3)] * 4 + [math.log(1.0), math.log(1e-2)])
        theta0 = np.clip(np.asarray(theta0, dtype=np.float64),
                         [math.log(0.02)] * 4 + [math.log(1e-3), math.log(self.noise_lo)],
                         [math.log(20.0)] * 4 + [math.log(1e3), 0.0])
        bounds = [(math.log(0.02), math.log(20.0))] * 4 + \
                 [(math.log(1e-3), math.log(1e3)),
                  (math.log(self.noise_lo), math.log(1.0))]

        res = minimize(lambda t: self._nll_grad(t, yd), theta0, jac=True,
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.maxiter, "maxcor": 10})
        self.theta = np.asarray(res.x, dtype=np.float64)
        ell, sf2, sn2 = self._unpack(self.theta)
        E = self._corr(self._D, xp.asarray(ell, dtype=self.dtype))
        K = sf2 * E
        idx = xp.arange(y.shape[0])
        K[idx, idx] += sn2 + self.jitter
        self._L = _cholesky(xp, K)
        v = _solve_tri(xp, self._L, yd[:, None])
        self._alpha = _solve_tri(xp, self._L, v, trans=1)
        self._ell, self._sf2, self._sn2 = ell, sf2, sn2
        self.fit_info = GPFit(theta=self.theta, nll=float(res.fun), n_train=int(y.shape[0]))
        del E, K
        self._D = None
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
        return self

    # --------------------------------------------------------------- predict
    def _kstar(self, Xs):
        """k(X_train, Xs) -> (n_train, m)."""
        xp = self.xp
        inv = xp.asarray(1.0 / self._ell ** 2, dtype=self.dtype)
        Q = xp.zeros((self._X.shape[0], Xs.shape[0]), dtype=self.dtype)
        for d in range(Xs.shape[1]):
            diff = self._X[:, d][:, None] - Xs[:, d][None, :]
            Q += inv[d] * diff * diff
        return self._sf2 * xp.exp(-0.5 * Q)

    def basis(self, Xs):
        """A = L^-1 k(X, Xs); var = sf2 - colsum(A^2), cov0 = k - A^T A."""
        Ks = self._kstar(self.xp.asarray(Xs, dtype=self.dtype))
        return _solve_tri(self.xp, self._L, Ks)

    def mean(self, Xs):
        Ks = self._kstar(self.xp.asarray(Xs, dtype=self.dtype))
        return (Ks.T @ self._alpha).ravel()

    def var(self, Xs):
        A = self.basis(Xs)
        return self._sf2 - (A * A).sum(axis=0)

    def mean_and_grad_norm(self, Xs):
        """(mean, ||grad_x mu||) -- analytic for the RBF kernel."""
        xp = self.xp
        Xs = xp.asarray(Xs, dtype=self.dtype)
        Ks = self._kstar(Xs)
        al = self._alpha.ravel()
        mu = Ks.T @ al
        g2 = xp.zeros(Xs.shape[0], dtype=self.dtype)
        for d in range(Xs.shape[1]):
            diff = Xs[:, d][None, :] - self._X[:, d][:, None]      # (n, m)
            gd = -((diff * Ks) * al[:, None]).sum(axis=0) / (self._ell[d] ** 2)
            g2 += gd * gd
        return mu, xp.sqrt(g2)

    def kernel_cross(self, Xa, Xb):
        """k(Xa, Xb) with the fitted hyperparameters."""
        xp = self.xp
        Xa = xp.asarray(Xa, dtype=self.dtype)
        Xb = xp.asarray(Xb, dtype=self.dtype)
        inv = xp.asarray(1.0 / self._ell ** 2, dtype=self.dtype)
        Q = xp.zeros((Xa.shape[0], Xb.shape[0]), dtype=self.dtype)
        for d in range(Xa.shape[1]):
            diff = Xa[:, d][:, None] - Xb[:, d][None, :]
            Q += inv[d] * diff * diff
        return self._sf2 * xp.exp(-0.5 * Q)

    @property
    def lengthscales(self) -> np.ndarray:
        return np.asarray(self._ell, dtype=np.float64)

    @property
    def signal_var(self) -> float:
        return float(self._sf2)

    @property
    def noise_var(self) -> float:
        return float(self._sn2)

    def free(self):
        self._X = self._D = self._L = self._alpha = None
        if self.xp is not np:
            self.xp.get_default_memory_pool().free_all_blocks()
