"""Tests for the forward predictive filter and model-order complexity
(src.hmm_tools), implementing Sec. 5.1-5.3 of "Explainable Regime-Aware
Investing"."""
import numpy as np
import pytest

from src.hmm_tools import (
    WarmStartHMMBank,
    forward_predictive_filter,
    gaussian_log_emissions,
    hmm_complexity,
)


def test_gaussian_log_emissions_shape_and_known_value():
    means = np.array([[0.0, 0.0], [1.0, 1.0]])
    covars = np.array([np.eye(2), np.eye(2)])
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    log_b = gaussian_log_emissions(X, means, covars)
    assert log_b.shape == (2, 2)
    # density of standard bivariate normal at its own mean = 1/(2*pi)
    assert log_b[0, 0] == pytest.approx(np.log(1.0 / (2 * np.pi)), rel=1e-5)


def test_forward_filter_probabilities_sum_to_one_each_day():
    rng = np.random.default_rng(0)
    T, K = 50, 3
    startprob = np.ones(K) / K
    transmat = rng.dirichlet(np.ones(K), size=K)
    log_b = rng.normal(size=(T, K))
    log_pred, log_filtered, pred_ll = forward_predictive_filter(startprob, transmat, log_b)
    filtered = np.exp(log_filtered)
    pred = np.exp(log_pred)
    assert np.allclose(filtered.sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-8)
    assert pred_ll.shape == (T,)


def test_forward_filter_predictive_is_causal_shift_of_filtered():
    """log_pred_probs[t] must depend only on filtered[t-1] and the transition
    matrix, never on log_b[t] (strict causality, Sec. 5.1)."""
    rng = np.random.default_rng(1)
    T, K = 20, 2
    startprob = np.array([0.6, 0.4])
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    log_b = rng.normal(size=(T, K))

    log_pred_a, log_filtered_a, _ = forward_predictive_filter(startprob, transmat, log_b)

    log_b_mutated = log_b.copy()
    log_b_mutated[10:] += 100.0  # change emissions strictly after t=9
    log_pred_b, _, _ = forward_predictive_filter(startprob, transmat, log_b_mutated)

    assert np.allclose(log_pred_a[:10], log_pred_b[:10], atol=1e-8)


def test_hmm_complexity_increases_with_k_and_d():
    c_small = hmm_complexity(2, 3)
    c_large_k = hmm_complexity(5, 3)
    c_large_d = hmm_complexity(2, 6)
    assert c_large_k > c_small
    assert c_large_d > c_small


def test_warm_start_bank_fits_and_reuses_state():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 3))
    bank = WarmStartHMMBank(n_iter=5, warm_n_iter=2, random_state=0)
    model_a = bank.fit(2, X)
    assert model_a.means_.shape == (2, 3)
    assert model_a.transmat_.shape == (2, 2)
    X2 = np.vstack([X, rng.normal(size=(5, 3))])
    model_b = bank.fit(2, X2)
    assert model_b.means_.shape == (2, 3)
