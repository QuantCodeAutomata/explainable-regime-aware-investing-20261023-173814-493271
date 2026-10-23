"""Feature construction shared by both regime-investing strategies.

Implements Sec. 4 / Sec. 5.1: the feature vector
x_t = [r_t; sigma_t; m_t] in R^{3N}, where sigma_t is a 60-day rolling
volatility per asset and m_t is a 20-day rolling mean return per asset. All
features at time t use information available up to t-1 (strict causality) --
this module computes the raw rolling statistics; the causality cut (using
x_1..x_{t-1} to decide w_t) is enforced by the backtest loop in `backtest.py`,
not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns r_t = log(P_t) - log(P_{t-1})."""
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive to take logs")
    return np.log(prices).diff().dropna()


def build_feature_matrix(
    returns: pd.DataFrame, vol_window: int = 60, mean_window: int = 20
) -> pd.DataFrame:
    """Build x_t = [r_t; sigma_t; m_t] for every asset, per Sec. 4.

    sigma_t(asset) = rolling std of returns over the trailing `vol_window`
    days ending at t (inclusive); m_t(asset) = rolling mean over the trailing
    `mean_window` days ending at t (inclusive). Rows before the longer window
    is full are dropped (warm-up), since sigma_t would otherwise be based on
    fewer than `vol_window` observations.
    """
    if vol_window < 2 or mean_window < 2:
        raise ValueError("rolling windows must be at least 2")
    vol = returns.rolling(vol_window).std()
    mean = returns.rolling(mean_window).mean()

    vol.columns = [f"{c}_vol" for c in returns.columns]
    mean.columns = [f"{c}_mean" for c in returns.columns]

    features = pd.concat([returns, vol, mean], axis=1)
    features = features.dropna(how="any")
    return features


def asset_columns(returns: pd.DataFrame) -> list:
    return list(returns.columns)
