"""End-to-end tests of the two shared backtests (src.backtest) on synthetic
data (data.generate_synthetic), implementing Algorithm 1 and Algorithm 2 of
"Explainable Regime-Aware Investing"."""
import numpy as np
import pandas as pd
import pytest

from data.generate_synthetic import generate_synthetic_prices
from src.backtest import run_knn_backtest, run_wasserstein_hmm_backtest
from src.config import KNNRegimeConfig, ReplicationConfig, WassersteinHMMConfig


def _small_config(prices: pd.DataFrame) -> ReplicationConfig:
    split = prices.index[int(len(prices) * 0.6)]
    return ReplicationConfig(
        tickers={c: c for c in prices.columns},
        train_test_split=str(split.date()),
        vol_window=20,
        mean_window=10,
        wasserstein_hmm=WassersteinHMMConfig(
            k_min=2, k_max=3, order_selection_freq=5, validation_length=15,
            n_templates=3, calibration_years=0.5, hmm_n_iter=5,
        ),
        knn=KNNRegimeConfig(n_neighbors=10, lookback=20),
    )


@pytest.fixture(scope="module")
def synthetic_prices():
    return generate_synthetic_prices(n_days=400, n_assets=3, seed=7)


def test_wasserstein_hmm_backtest_outputs_are_well_formed(synthetic_prices):
    config = _small_config(synthetic_prices)
    result = run_wasserstein_hmm_backtest(synthetic_prices, config)
    weights = result["weights"]
    returns = result["returns"]

    assert set(["weights", "returns", "diagnostics", "metrics"]).issubset(result.keys())
    assert len(weights) == len(returns) > 0
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
    assert (weights.values >= -1e-6).all()
    assert (weights.values <= config.wasserstein_hmm.max_weight + 1e-4).all()
    assert "sharpe_ratio" in result["metrics"]
    assert result["metrics"]["n_days"] == len(returns)


def test_knn_backtest_outputs_are_well_formed(synthetic_prices):
    config = _small_config(synthetic_prices)
    result = run_knn_backtest(synthetic_prices, config)
    weights = result["weights"]
    returns = result["returns"]

    assert len(weights) == len(returns) > 0
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
    assert (weights.values >= -1e-6).all()
    assert (weights.values <= config.knn.max_weight + 1e-4).all()
    assert result["metrics"]["n_days"] == len(returns)


def test_knn_backtest_query_uses_prior_day_features_no_lookahead(synthetic_prices):
    """Perturbing today's realized return must not change today's weight
    decision: the KNN query vector must be built from information available
    up to t-1 (Sec. 7.2's causal requirement), never from r_t itself."""
    config = _small_config(synthetic_prices)
    result_a = run_knn_backtest(synthetic_prices, config)

    mutated = synthetic_prices.copy()
    # Shock the price on the *last* OOS day only: this changes r_T but must
    # not change the weight decision made for day T (the decision only uses
    # the feature row built through day T-1).
    last_date = mutated.index[-1]
    mutated.loc[last_date] = mutated.loc[last_date] * 1.5
    result_b = run_knn_backtest(mutated, config)

    w_a = result_a["weights"].iloc[-1]
    w_b = result_b["weights"].iloc[-1]
    pd.testing.assert_series_equal(w_a, w_b, check_exact=False, atol=1e-8)


def test_backtest_raises_when_no_oos_dates(synthetic_prices):
    config = _small_config(synthetic_prices)
    config.train_test_split = str(synthetic_prices.index[-1].date())
    with pytest.raises(ValueError):
        run_knn_backtest(synthetic_prices, config)
