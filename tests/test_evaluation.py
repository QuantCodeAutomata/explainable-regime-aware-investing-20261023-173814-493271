"""Tests for the Sec. 6 evaluation metrics (src.evaluation)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    effective_n_positions,
    max_drawdown,
    sharpe_ratio,
    summarize_performance,
    turnover_series,
)


def test_sharpe_ratio_zero_for_zero_vol():
    r = pd.Series([0.001] * 10)
    assert sharpe_ratio(r) == 0.0


def test_sharpe_ratio_known_value():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, size=5000))
    sr = sharpe_ratio(r, periods_per_year=252)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sr == pytest.approx(expected, rel=1e-9)


def test_max_drawdown_is_nonpositive_and_matches_manual_example():
    # wealth path: 1 -> 1.1 -> 0.88 -> 0.99 ; min drawdown at the trough = 0.88/1.1 - 1
    r = pd.Series([0.10, -0.20, 0.125])
    dd = max_drawdown(r)
    assert dd <= 0
    assert dd == pytest.approx(0.88 / 1.1 - 1.0, rel=1e-6)


def test_turnover_series_formula_and_bounds():
    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.2], "B": [0.5, 0.5, 0.8]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    to = turnover_series(weights)
    assert to.iloc[0] == pytest.approx(0.5)  # 0.5 * (|0.5|+|0.5|)
    assert to.iloc[1] == pytest.approx(0.0)  # unchanged weights
    assert to.iloc[2] == pytest.approx(0.3)  # 0.5 * (0.3+0.3)
    assert (to >= 0).all()


def test_effective_n_positions_bounds():
    n_assets = 5
    equal_weights = pd.DataFrame([[1.0 / n_assets] * n_assets], columns=list("ABCDE"))
    neff_equal = effective_n_positions(equal_weights)
    assert neff_equal.iloc[0] == pytest.approx(n_assets)

    concentrated = pd.DataFrame([[1.0, 0.0, 0.0, 0.0, 0.0]], columns=list("ABCDE"))
    neff_conc = effective_n_positions(concentrated)
    assert neff_conc.iloc[0] == pytest.approx(1.0)
    assert 1.0 <= neff_equal.iloc[0] <= n_assets


def test_summarize_performance_returns_all_expected_keys():
    idx = pd.date_range("2022-01-01", periods=50, freq="B")
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0003, 0.005, size=50), index=idx)
    weights = pd.DataFrame(
        rng.dirichlet(np.ones(3), size=50), index=idx, columns=["X", "Y", "Z"]
    )
    metrics = summarize_performance(returns, weights)
    expected_keys = {
        "sharpe_ratio", "annualized_return", "annualized_vol", "max_drawdown",
        "avg_daily_turnover", "turnover_q95", "avg_effective_n_positions", "n_days",
    }
    assert set(metrics.keys()) == expected_keys
    assert metrics["n_days"] == 50
    assert metrics["max_drawdown"] <= 0
