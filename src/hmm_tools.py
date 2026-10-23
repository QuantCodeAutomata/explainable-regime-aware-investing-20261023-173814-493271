"""Gaussian HMM fitting, forward (predictive) filtering, and model-order
complexity, implementing Algorithm 1 (Sec. 5.1-5.3) of the paper.

Context7 check: `hmmlearn.hmm.GaussianHMM` (the library named in the
experiment's implementation notes) provides EM fitting and a *smoothed*
`predict_proba`, but not the one-step-ahead *predictive* filtered
probabilities P(z_t | F_{t-1}) or the per-step predictive log-likelihood that
Algorithm 1 needs for causal model-order selection (Sec. 5.2) and daily
regime inference (Sec. 5.3). Those recursions are implemented here from
scratch on top of the fitted hmmlearn parameters (startprob_, transmat_,
means_, covars_), using the standard forward-algorithm recursion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def gaussian_log_emissions(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """log b_k(x_t) for every t, k. means: (K,D); covars: (K,D,D)."""
    T = X.shape[0]
    K = means.shape[0]
    log_b = np.zeros((T, K))
    for k in range(K):
        cov_k = covars[k] + 1e-10 * np.eye(means.shape[1])
        log_b[:, k] = multivariate_normal.logpdf(X, mean=means[k], cov=cov_k)
    return log_b


def forward_predictive_filter(
    startprob: np.ndarray, transmat: np.ndarray, log_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward algorithm returning, for every t = 0..T-1:
    - log_pred_probs[t,k] = log P(z_t=k | x_1..x_{t-1})  (predictive, prior for t=0)
    - log_filtered[t,k]   = log P(z_t=k | x_1..x_t)       (filtered)
    - pred_loglik[t]      = log P(x_t | x_1..x_{t-1})     (one-step predictive log-lik)
    """
    T, K = log_b.shape
    log_A = np.log(np.clip(transmat, 1e-300, None))
    log_pred_probs = np.zeros((T, K))
    log_filtered = np.zeros((T, K))
    pred_loglik = np.zeros(T)

    log_pred_probs[0] = np.log(np.clip(startprob, 1e-300, None))
    step0 = log_pred_probs[0] + log_b[0]
    pred_loglik[0] = logsumexp(step0)
    log_filtered[0] = step0 - pred_loglik[0]

    for t in range(1, T):
        log_pred_probs[t] = logsumexp(log_filtered[t - 1][:, None] + log_A, axis=0)
        step = log_pred_probs[t] + log_b[t]
        pred_loglik[t] = logsumexp(step)
        log_filtered[t] = step - pred_loglik[t]

    return log_pred_probs, log_filtered, pred_loglik


def hmm_complexity(k: int, d: int) -> float:
    """Complexity(K): number of free parameters of a K-state full-covariance
    Gaussian HMM in D dimensions (means + symmetric covariances + transition
    matrix rows + initial distribution), used as the AIC-style penalty term
    in the model-order selection score (Sec. 5.2)."""
    n_means = k * d
    n_cov = k * d * (d + 1) // 2
    n_trans = k * (k - 1)
    n_start = k - 1
    return float(n_means + n_cov + n_trans + n_start)


@dataclass
class _WarmState:
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray


class WarmStartHMMBank:
    """Maintains one warm-startable Gaussian HMM per candidate order K.

    Refitting an expanding-window HMM from scratch every day is wasteful
    because the window grows by very few observations between calls; warm
    starting from the previous day's EM solution (`init_params=''`) achieves
    the same expanding-history causal fit (Sec. 5.1, Algorithm 1 lines 9-10)
    at a small fraction of the compute cost, since EM then needs only a few
    iterations to re-converge.
    """

    def __init__(self, n_iter: int = 20, warm_n_iter: int = 5, tol: float = 1e-2, random_state: int = 0):
        self.n_iter = n_iter
        self.warm_n_iter = warm_n_iter
        self.tol = tol
        self.random_state = random_state
        self._state: Dict[int, _WarmState] = {}

    @staticmethod
    def _regularize_covars(covars: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Project each component covariance back onto the positive-definite
        cone by clipping its eigenvalues from below, guarding the warm-start
        EM recursion (Sec. 5.1) against the numerically singular covariances
        that can arise when a component is assigned very few points on the
        early, short expanding-history windows."""
        reg = covars.copy()
        d = covars.shape[-1]
        for k in range(covars.shape[0]):
            cov = 0.5 * (reg[k] + reg[k].T)
            eigval, eigvec = np.linalg.eigh(cov)
            eigval = np.clip(eigval, eps, None)
            reg[k] = (eigvec * eigval) @ eigvec.T
        return reg

    def _fit_from_scratch(self, k: int, X: np.ndarray) -> GaussianHMM:
        model = GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            init_params="stmc",
            params="stmc",
            min_covar=1e-6,
        )
        model.fit(X)
        return model

    def fit(self, k: int, X: np.ndarray) -> GaussianHMM:
        """Fit (or warm-refit) a K-state full-covariance Gaussian HMM on X."""
        prev = self._state.get(k)
        if prev is None:
            model = self._fit_from_scratch(k, X)
        else:
            model = GaussianHMM(
                n_components=k,
                covariance_type="full",
                n_iter=self.warm_n_iter,
                tol=self.tol,
                random_state=self.random_state,
                init_params="",
                params="stmc",
                min_covar=1e-6,
            )
            model.startprob_ = prev.startprob.copy()
            model.transmat_ = prev.transmat.copy()
            model.means_ = prev.means.copy()
            model.covars_ = self._regularize_covars(prev.covars.copy())
            try:
                model.fit(X)
            except ValueError:
                # Warm EM drifted to a singular covariance on this window;
                # recover with a fresh fit rather than propagating the state.
                model = self._fit_from_scratch(k, X)
        self._state[k] = _WarmState(
            startprob=model.startprob_.copy(),
            transmat=model.transmat_.copy(),
            means=model.means_.copy(),
            covars=self._regularize_covars(model.covars_.copy()),
        )
        return model
