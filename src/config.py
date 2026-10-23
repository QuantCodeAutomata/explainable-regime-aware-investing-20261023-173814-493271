"""Shared configuration for the regime-aware investing replication.

Parameter provenance follows the source priority of the replication: values
explicitly given in the paper ("Explainable Regime-Aware Investing") are used
verbatim and cited by section; values the paper leaves unspecified (see its
"uncertainties" discussion around Algorithm 1 / Algorithm 2, Sec. 7) are fixed
to conservative, literature-standard defaults and recorded here with the
"ASSUMPTION" tag so that every unresolved detail is explicit and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# Sec. 3 "Data and Market Universe": SPX / BOND / GOLD / OIL / USD are given
# only as asset-class proxies, not as exact ticker symbols. We map each proxy
# to the most standard liquid Yahoo Finance ETF/future for that asset class.
# ASSUMPTION: exact tickers are not specified in the paper.
ASSET_PROXY_TICKERS: Dict[str, str] = {
    "SPX": "SPY",     # S&P 500 Index proxy (Sec. 3)
    "BOND": "AGG",    # Broad bond proxy (Sec. 3)
    "GOLD": "GLD",    # Gold (Sec. 3)
    "OIL": "USO",      # Oil (Sec. 3)
    "USD": "UUP",      # U.S. Dollar proxy (Sec. 3)
}

# Sec. 3: "daily adjusted close prices from Yahoo Finance from 2005 to 2026".
DATA_START = "2005-01-01"
DATA_END = "2026-10-23"  # replication run date; paper states data "to 2026"

# Sec. 8 reports the first out-of-sample weights in May 2023 (Table 3 / Fig.
# 4-5 portfolio-weight tables both start at "2023-05"), which fixes the
# train/test split date t0 used by Algorithm 1 and Algorithm 2.
TRAIN_TEST_SPLIT = "2023-04-30"

# Sec. 4 / 5.1: feature windows (w_sigma, w_m) are given explicitly.
VOL_WINDOW = 60     # 60-day rolling volatility
MEAN_WINDOW = 20    # 20-day rolling mean return


@dataclass
class WassersteinHMMConfig:
    """Algorithm 1 parameters (Sec. 7.1). Values not numerically specified by
    the paper are marked ASSUMPTION and are each a single literature-standard
    default; they do not change the method, only its operating point."""

    k_min: int = 2          # ASSUMPTION: smallest candidate regime count
    k_max: int = 5          # ASSUMPTION: largest candidate regime count
    order_selection_freq: int = 5     # F_K: ASSUMPTION, "weekly" per Sec. 5.2 example -> 5 trading days
    validation_length: int = 60       # |V_t|: ASSUMPTION, one quarter of trading days
    complexity_lambda: float = 1.0    # lambda_K: ASSUMPTION, AIC-style penalty (score = PredLL - Complexity(K))
    n_templates: int = 5              # G: ASSUMPTION, set equal to K_max so every component can map to a distinct template
    smoothing_rate: float = 0.1       # eta: ASSUMPTION, exponential-smoothing rate for template tracking
    risk_aversion: float = 5.0        # gamma: ASSUMPTION, standard MVO risk-aversion coefficient
    transaction_cost: float = 0.001   # tau: ASSUMPTION, 10 bps linear transaction cost
    max_weight: float = 0.6           # w_max: ASSUMPTION, per-asset weight cap
    calibration_years: float = 2.0    # ASSUMPTION: length of the "initial calibration window" (Algorithm 1, line 3)
    hmm_n_iter: int = 20
    hmm_tol: float = 1e-2
    random_state: int = 0


@dataclass
class KNNRegimeConfig:
    """Algorithm 2 parameters (Sec. 7.2). F and the triggers (delta_mu,
    delta_Sigma) appear only in the Require line and are never used in the
    algorithm steps; per the paper text and the experiment description,
    standard daily rebalancing is applied and F / triggers are not used."""

    n_neighbors: int = 60     # K: ASSUMPTION, number of nearest neighbors
    lookback: int = 60        # L: ASSUMPTION, warm-up before the OOS loop starts (>= vol window)
    risk_aversion: float = 5.0     # lambda: ASSUMPTION, same economic risk-aversion value used for the HMM strategy
    transaction_cost: float = 0.001  # tau: ASSUMPTION, 10 bps linear transaction cost
    max_weight: float = 0.6        # w_max: ASSUMPTION, per-asset weight cap


@dataclass
class ReplicationConfig:
    tickers: Dict[str, str] = field(default_factory=lambda: dict(ASSET_PROXY_TICKERS))
    data_start: str = DATA_START
    data_end: str = DATA_END
    train_test_split: str = TRAIN_TEST_SPLIT
    vol_window: int = VOL_WINDOW
    mean_window: int = MEAN_WINDOW
    wasserstein_hmm: WassersteinHMMConfig = field(default_factory=WassersteinHMMConfig)
    knn: KNNRegimeConfig = field(default_factory=KNNRegimeConfig)


DEFAULT_CONFIG = ReplicationConfig()
