"""Synthetic OHLC price generator.

Not used by the headline experiments: Section 3 of the paper requires real
daily adjusted close prices from Yahoo Finance for the SPX/BOND/GOLD/OIL/USD
cross-asset universe. This module exists only to support deterministic,
network-free unit tests of the shared feature/backtest pipeline in `tests/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_prices(
    n_days: int = 800,
    n_assets: int = 5,
    asset_labels=None,
    seed: int = 42,
    start: str = "2015-01-01",
    regime_switch_every: int = 150,
) -> pd.DataFrame:
    """Generate a synthetic multi-asset price panel with regime switches.

    Produces log-return series drawn from alternating Gaussian regimes (mean
    and covariance shift every `regime_switch_every` days) so that unit tests
    can exercise the HMM/KNN regime-detection pipeline without external data.
    """
    if asset_labels is None:
        asset_labels = [f"A{i}" for i in range(n_assets)]
    assert len(asset_labels) == n_assets

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    returns = np.zeros((n_days, n_assets))
    regime_means = [
        np.full(n_assets, 0.0006),
        np.full(n_assets, -0.0008),
    ]
    base_cov = 0.0001 * (np.eye(n_assets) * 0.8 + 0.2)
    regime_covs = [base_cov, base_cov * 2.5]

    for t in range(n_days):
        regime = (t // regime_switch_every) % 2
        returns[t] = rng.multivariate_normal(regime_means[regime], regime_covs[regime])

    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=asset_labels)
