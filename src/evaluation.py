"""Evaluation metrics (Sec. 6): out-of-sample Sharpe ratio, maximum drawdown,
daily turnover, and effective number of positions, each defined exactly as in
the methodology."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio: mean(r) / std(r) * sqrt(periods_per_year)."""
    if len(returns) < 2 or returns.std(ddof=1) < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of the cumulative wealth curve built from `returns`."""
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    """TO_t = 0.5 * ||w_t - w_{t-1}||_1 (Sec. 6)."""
    diffs = weights.diff().abs().sum(axis=1) * 0.5
    diffs.iloc[0] = 0.5 * weights.iloc[0].abs().sum()
    return diffs


def effective_n_positions(weights: pd.DataFrame) -> pd.Series:
    """N_eff,t = (sum_i w_{t,i}^2)^{-1} (Sec. 6)."""
    sq_sum = (weights ** 2).sum(axis=1)
    return 1.0 / sq_sum.replace(0, np.nan)


def summarize_performance(returns: pd.Series, weights: pd.DataFrame) -> Dict[str, float]:
    """Aggregate the Sec. 6 evaluation metrics into one dict."""
    to = turnover_series(weights)
    neff = effective_n_positions(weights)
    return {
        "sharpe_ratio": sharpe_ratio(returns),
        "annualized_return": float(returns.mean() * TRADING_DAYS_PER_YEAR),
        "annualized_vol": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "max_drawdown": max_drawdown(returns),
        "avg_daily_turnover": float(to.mean()),
        "turnover_q95": float(to.quantile(0.95)),
        "avg_effective_n_positions": float(neff.mean()),
        "n_days": int(len(returns)),
    }
