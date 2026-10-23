"""Tests for the closed-form 2-Wasserstein distance between Gaussians
(src.wasserstein), implementing Sec. 5.4 of "Explainable Regime-Aware
Investing"."""
import numpy as np
import pytest

from src.wasserstein import gaussian_w2, gaussian_w2_squared


def test_distance_to_self_is_zero():
    mu = np.array([0.01, -0.02, 0.0])
    sigma = np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 0.5]])
    d2 = gaussian_w2_squared(mu, sigma, mu, sigma)
    assert d2 == pytest.approx(0.0, abs=1e-8)


def test_univariate_equal_variance_reduces_to_mean_distance():
    # For equal-covariance 1-D Gaussians, W2^2 = (mu1 - mu2)^2 exactly.
    mu1, mu2 = np.array([1.0]), np.array([4.0])
    sigma = np.array([[2.0]])
    d2 = gaussian_w2_squared(mu1, sigma, mu2, sigma)
    assert d2 == pytest.approx((1.0 - 4.0) ** 2, rel=1e-6)


def test_univariate_equal_mean_matches_variance_gap_formula():
    # For equal-mean 1-D Gaussians, W2^2 = (sigma1 - sigma2)^2.
    mu = np.array([0.0])
    s1, s2 = np.array([[4.0]]), np.array([[1.0]])
    d2 = gaussian_w2_squared(mu, s1, mu, s2)
    assert d2 == pytest.approx((2.0 - 1.0) ** 2, rel=1e-6)


def test_distance_nonnegative_and_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(10):
        d = 3
        mu1, mu2 = rng.normal(size=d), rng.normal(size=d)
        a1, a2 = rng.normal(size=(d, d)), rng.normal(size=(d, d))
        s1, s2 = a1 @ a1.T + np.eye(d), a2 @ a2.T + np.eye(d)
        d2_fwd = gaussian_w2_squared(mu1, s1, mu2, s2)
        d2_bwd = gaussian_w2_squared(mu2, s2, mu1, s1)
        assert d2_fwd >= -1e-8
        assert d2_fwd == pytest.approx(d2_bwd, rel=1e-5, abs=1e-8)


def test_gaussian_w2_is_sqrt_of_squared():
    mu1, mu2 = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    s1 = s2 = np.eye(2)
    assert gaussian_w2(mu1, s1, mu2, s2) == pytest.approx(
        np.sqrt(gaussian_w2_squared(mu1, s1, mu2, s2))
    )
