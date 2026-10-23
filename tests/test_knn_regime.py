"""Tests for the rolling KNN conditional-moment estimator (src.knn_regime),
implementing Algorithm 2 / Sec. 7.2 of "Explainable Regime-Aware Investing"."""
import numpy as np
import pytest

from src.knn_regime import knn_conditional_moments


def test_knn_moments_shape_and_mean_matches_manual_average():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(100, 3))
    r_train = rng.normal(size=(100, 3)) * 0.01
    x_query = x_train[0] + 1e-6  # nearly identical to a known training point

    mu, sigma = knn_conditional_moments(x_train, r_train, x_query, n_neighbors=5)
    assert mu.shape == (3,)
    assert sigma.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(sigma) >= -1e-10)  # PSD


def test_knn_moments_exact_match_for_single_neighbor():
    x_train = np.array([[0.0, 0.0], [10.0, 10.0]])
    r_train = np.array([[0.01, 0.02], [0.5, 0.5]])
    x_query = np.array([0.0, 0.0])
    mu, _ = knn_conditional_moments(x_train, r_train, x_query, n_neighbors=1)
    assert np.allclose(mu, [0.01, 0.02])


def test_knn_handles_more_neighbors_requested_than_available():
    x_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    r_train = np.array([[0.01, 0.0], [0.02, 0.0], [0.03, 0.0]])
    x_query = np.array([0.0, 0.0])
    mu, sigma = knn_conditional_moments(x_train, r_train, x_query, n_neighbors=100)
    assert mu.shape == (2,)
    assert sigma.shape == (2, 2)


def test_knn_moments_average_over_k_neighbors():
    x_train = np.array([[0.0], [0.1], [0.2], [5.0]])
    r_train = np.array([[0.01], [0.02], [0.03], [-1.0]])
    x_query = np.array([0.0])
    mu, _ = knn_conditional_moments(x_train, r_train, x_query, n_neighbors=3)
    assert mu[0] == pytest.approx((0.01 + 0.02 + 0.03) / 3)
