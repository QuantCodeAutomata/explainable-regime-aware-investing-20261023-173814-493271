"""Closed-form 2-Wasserstein distance between multivariate Gaussians.

Implements Sec. 5.4, Eq.:
W_2^2(N(mu1,Sigma1), N(mu2,Sigma2))
    = ||mu1 - mu2||_2^2 + Tr(Sigma1 + Sigma2 - 2*(Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2}).

Context7 (cvxpy / scipy ecosystem) was checked for an off-the-shelf closed-form
Gaussian-Wasserstein distance; `POT` (Python Optimal Transport) exposes only
sample-based / discrete optimal transport solvers (e.g. `ot.emd2`), not the
closed-form Gaussian formula, so this is a direct, from-scratch implementation
of the paper's Eq. in Sec. 5.4 using `scipy.linalg.sqrtm`.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def _psd_sqrtm(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetric PSD matrix square root, clipping tiny negative eigenvalues
    that arise from numerical noise."""
    mat = 0.5 * (mat + mat.T)
    eigval, eigvec = np.linalg.eigh(mat)
    eigval = np.clip(eigval, eps, None)
    return (eigvec * np.sqrt(eigval)) @ eigvec.T


def gaussian_w2_squared(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Squared 2-Wasserstein distance between N(mu1, sigma1) and N(mu2, sigma2)."""
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    mean_term = float(np.sum((mu1 - mu2) ** 2))

    sqrt_s2 = _psd_sqrtm(sigma2)
    inner = sqrt_s2 @ sigma1 @ sqrt_s2
    inner_sqrt = _psd_sqrtm(inner)
    cov_term = np.trace(sigma1 + sigma2 - 2.0 * inner_sqrt)
    # Numerical noise can leave a tiny negative trace; the true value is >= 0.
    cov_term = max(float(np.real(cov_term)), 0.0)
    return mean_term + cov_term


def gaussian_w2(mu1, sigma1, mu2, sigma2) -> float:
    return float(np.sqrt(gaussian_w2_squared(mu1, sigma1, mu2, sigma2)))
