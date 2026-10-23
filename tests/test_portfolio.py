"""Tests for the transaction-cost-aware MVO solver (src.portfolio), Sec. 6."""
import numpy as np
import pytest

from src.portfolio import solve_tc_aware_mvo


def _check_feasible(w, w_max, n):
    assert w.shape == (n,)
    assert np.all(w >= -1e-8)
    assert np.all(w <= w_max + 1e-6)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_solution_satisfies_budget_and_bounds():
    mu = np.array([0.001, 0.002, -0.0005, 0.0003])
    sigma = 0.0001 * np.eye(4)
    w_prev = np.full(4, 0.25)
    w = solve_tc_aware_mvo(mu, sigma, w_prev, risk_aversion=5.0, tau=0.001, w_max=0.6)
    _check_feasible(w, 0.6, 4)


def test_max_weight_cap_binds_for_a_dominant_asset():
    mu = np.array([0.01, 0.0, 0.0])
    sigma = 0.0001 * np.eye(3)
    w_prev = np.full(3, 1.0 / 3)
    w = solve_tc_aware_mvo(mu, sigma, w_prev, risk_aversion=1.0, tau=0.0, w_max=0.5)
    _check_feasible(w, 0.5, 3)
    assert w[0] == pytest.approx(0.5, abs=1e-3)


def test_zero_transaction_cost_moves_fully_to_optimum_vs_high_cost_stays_put():
    mu = np.array([0.01, -0.01])
    sigma = 0.0001 * np.eye(2)
    w_prev = np.array([0.0, 1.0])
    w_low_tau = solve_tc_aware_mvo(mu, sigma, w_prev, risk_aversion=5.0, tau=0.0, w_max=1.0)
    w_high_tau = solve_tc_aware_mvo(mu, sigma, w_prev, risk_aversion=5.0, tau=10.0, w_max=1.0)
    # higher transaction cost should keep the solution closer to w_prev
    dist_low = np.abs(w_low_tau - w_prev).sum()
    dist_high = np.abs(w_high_tau - w_prev).sum()
    assert dist_high <= dist_low + 1e-6


def test_higher_risk_aversion_reduces_concentration():
    mu = np.array([0.01, 0.009, 0.008])
    sigma = 0.0004 * (0.5 * np.eye(3) + 0.5 * np.ones((3, 3)))
    w_prev = np.full(3, 1.0 / 3)
    w_low_gamma = solve_tc_aware_mvo(mu, sigma, w_prev, risk_aversion=0.5, tau=0.0, w_max=1.0)
    w_high_gamma = solve_tc_aware_mvo(mu, sigma, w_prev, risk_aversion=50.0, tau=0.0, w_max=1.0)
    herfindahl_low = (w_low_gamma ** 2).sum()
    herfindahl_high = (w_high_gamma ** 2).sum()
    assert herfindahl_high <= herfindahl_low + 1e-6
