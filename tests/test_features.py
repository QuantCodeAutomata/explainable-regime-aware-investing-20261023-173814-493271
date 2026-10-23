"""Tests for shared feature construction (src.features), implementing Sec. 4
of "Explainable Regime-Aware Investing": x_t = [r_t; sigma_t; m_t]."""
import numpy as np
import pandas as pd
import pytest

from data.generate_synthetic import generate_synthetic_prices
from src.features import asset_columns, build_feature_matrix, log_returns


def test_log_returns_matches_manual_diff_of_log_prices():
    prices = generate_synthetic_prices(n_days=100, n_assets=2, seed=1)
    r = log_returns(prices)
    expected = np.log(prices).diff().dropna()
    pd.testing.assert_frame_equal(r, expected)
    assert len(r) == len(prices) - 1


def test_log_returns_rejects_nonpositive_prices():
    prices = pd.DataFrame({"A": [1.0, -2.0, 3.0]})
    with pytest.raises(ValueError):
        log_returns(prices)


def test_feature_matrix_columns_and_shape():
    prices = generate_synthetic_prices(n_days=200, n_assets=3, seed=2)
    r = log_returns(prices)
    feats = build_feature_matrix(r, vol_window=60, mean_window=20)
    expected_cols = list(r.columns) + [f"{c}_vol" for c in r.columns] + [f"{c}_mean" for c in r.columns]
    assert list(feats.columns) == expected_cols
    assert feats.shape[1] == 3 * r.shape[1]
    assert not feats.isna().any().any()
    # warm-up: rows before vol_window (the longer window) are dropped
    assert len(feats) == len(r) - 59


def test_feature_at_time_t_only_uses_returns_up_to_t_no_lookahead():
    """sigma_t / m_t at date t must be computable from r_1..r_t only: mutating
    a return strictly after t must not change the feature row at t."""
    prices = generate_synthetic_prices(n_days=200, n_assets=2, seed=3)
    r = log_returns(prices)
    feats_a = build_feature_matrix(r, vol_window=60, mean_window=20)

    r_mutated = r.copy()
    mutate_at = r_mutated.index[150]
    r_mutated.loc[mutate_at, :] = r_mutated.loc[mutate_at, :] + 10.0
    feats_b = build_feature_matrix(r_mutated, vol_window=60, mean_window=20)

    check_date = r.index[120]
    assert check_date in feats_a.index and check_date in feats_b.index
    pd.testing.assert_series_equal(feats_a.loc[check_date], feats_b.loc[check_date])


def test_rolling_windows_must_be_at_least_two():
    prices = generate_synthetic_prices(n_days=50, n_assets=2, seed=4)
    r = log_returns(prices)
    with pytest.raises(ValueError):
        build_feature_matrix(r, vol_window=1, mean_window=20)


def test_asset_columns_matches_returns_columns():
    prices = generate_synthetic_prices(n_days=50, n_assets=4, seed=5)
    r = log_returns(prices)
    assert asset_columns(r) == list(r.columns)
