"""Orchestration of the two regime-aware backtests (Sec. 5-8 of the paper):

- `run_wasserstein_hmm_backtest`: Algorithm 1 (strictly causal Wasserstein-
  template Gaussian HMM + transaction-cost-aware MVO).
- `run_knn_backtest`: Algorithm 2 (rolling KNN conditional-moment estimator +
  transaction-cost-aware MVO).

Both share one feature pipeline (`src.features`) and one portfolio solver
(`src.portfolio.solve_tc_aware_mvo`) so that any performance difference
reflects only the regime-estimation method, as the paper intends.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.config import KNNRegimeConfig, ReplicationConfig, WassersteinHMMConfig
from src.evaluation import summarize_performance
from src.features import build_feature_matrix, log_returns
from src.hmm_tools import (
    WarmStartHMMBank,
    forward_predictive_filter,
    gaussian_log_emissions,
    hmm_complexity,
)
from src.knn_regime import knn_conditional_moments
from src.portfolio import solve_tc_aware_mvo
from src.regime_templates import TemplateBank

logger = logging.getLogger(__name__)


def _prepare_features(prices: pd.DataFrame, config: ReplicationConfig):
    returns = log_returns(prices)
    features = build_feature_matrix(returns, config.vol_window, config.mean_window)
    return returns, features


def run_wasserstein_hmm_backtest(
    prices: pd.DataFrame, config: ReplicationConfig
) -> Dict[str, object]:
    """Algorithm 1: Wasserstein-template HMM + transaction-cost-aware MVO."""
    hmm_cfg: WassersteinHMMConfig = config.wasserstein_hmm
    returns, features = _prepare_features(prices, config)
    assets = list(prices.columns)
    n_assets = len(assets)

    dates = features.index
    X_all = features.values
    split_date = pd.Timestamp(config.train_test_split)
    test_dates = dates[dates > split_date]
    if len(test_dates) == 0:
        raise ValueError("no out-of-sample dates after train_test_split")

    calib_cutoff = dates.min() + pd.Timedelta(days=int(round(hmm_cfg.calibration_years * 365.25)))
    calib_mask = dates <= calib_cutoff
    calib_X = X_all[calib_mask]
    if calib_X.shape[0] < hmm_cfg.n_templates * 5:
        raise ValueError("initial calibration window too short for template initialization")

    template_bank = TemplateBank.initialize(
        calib_X, hmm_cfg.n_templates, hmm_cfg.smoothing_rate, hmm_cfg.random_state
    )
    hmm_bank = WarmStartHMMBank(
        n_iter=hmm_cfg.hmm_n_iter, warm_n_iter=5, tol=hmm_cfg.hmm_tol, random_state=hmm_cfg.random_state
    )

    date_to_pos = {d: i for i, d in enumerate(dates)}
    w_prev = np.full(n_assets, 1.0 / n_assets)
    current_K = hmm_cfg.k_min
    feature_dim = X_all.shape[1]

    weight_records, diag_records = [], []
    for day_count, t in enumerate(test_dates):
        pos = date_to_pos[t]
        H_t = X_all[:pos]
        min_history = hmm_cfg.k_max * 5
        if H_t.shape[0] < min_history:
            w_t = w_prev
            weight_records.append(w_t)
            diag_records.append({"date": t, "K_t": np.nan, "score": np.nan})
            w_prev = w_t
            continue

        if day_count % hmm_cfg.order_selection_freq == 0:
            validation_len = min(hmm_cfg.validation_length, H_t.shape[0] // 2)
            best_score, best_K, best_model, best_log_filtered = -np.inf, None, None, None
            for K in range(hmm_cfg.k_min, hmm_cfg.k_max + 1):
                model = hmm_bank.fit(K, H_t)
                log_b = gaussian_log_emissions(H_t, model.means_, model.covars_)
                _, log_filtered, pred_loglik = forward_predictive_filter(
                    model.startprob_, model.transmat_, log_b
                )
                pred_ll = float(pred_loglik[-validation_len:].sum())
                complexity = hmm_complexity(K, feature_dim)
                score = pred_ll - hmm_cfg.complexity_lambda * complexity
                if score > best_score:
                    best_score, best_K, best_model, best_log_filtered = score, K, model, log_filtered
            current_K, model, log_filtered = best_K, best_model, best_log_filtered
            selection_score = best_score
        else:
            model = hmm_bank.fit(current_K, H_t)
            log_b = gaussian_log_emissions(H_t, model.means_, model.covars_)
            _, log_filtered, _ = forward_predictive_filter(model.startprob_, model.transmat_, log_b)
            selection_score = np.nan

        filtered_last = np.exp(log_filtered[-1])
        predictive_probs = filtered_last @ model.transmat_  # p_{t,k} = P(z_t=k | F_{t-1})

        template_probs = template_bank.map_and_update(predictive_probs, model.means_, model.covars_)
        mu_full, sigma_full = template_bank.aggregate_moments(template_probs)

        # The HMM observes the augmented feature vector [r; sigma; m] (Sec. 4),
        # but the MVO objective (Sec. 6) needs only expected returns and the
        # return covariance. For a (template-mixture) Gaussian, the marginal
        # on a subset of coordinates is exactly the corresponding sub-vector /
        # sub-matrix, so we take the return block of the aggregated moments.
        mu_t = mu_full[:n_assets]
        sigma_t = sigma_full[:n_assets, :n_assets]

        w_t = solve_tc_aware_mvo(
            mu_t, sigma_t, w_prev, hmm_cfg.risk_aversion, hmm_cfg.transaction_cost, hmm_cfg.max_weight
        )
        weight_records.append(w_t)
        diag_records.append({"date": t, "K_t": current_K, "score": selection_score})
        w_prev = w_t

    weights_df = pd.DataFrame(weight_records, index=test_dates, columns=assets)
    diagnostics_df = pd.DataFrame(diag_records).set_index("date")
    portfolio_returns = (weights_df * returns.loc[test_dates]).sum(axis=1)
    portfolio_returns.name = "portfolio_return"

    metrics = summarize_performance(portfolio_returns, weights_df)
    logger.info("Wasserstein-HMM backtest: %d OOS days, metrics=%s", len(test_dates), metrics)

    return {
        "weights": weights_df,
        "returns": portfolio_returns,
        "diagnostics": diagnostics_df,
        "metrics": metrics,
    }


def run_knn_backtest(prices: pd.DataFrame, config: ReplicationConfig) -> Dict[str, object]:
    """Algorithm 2: rolling KNN conditional-moment estimator + transaction-
    cost-aware MVO. Per the paper text and the experiment description, the
    rebalance-frequency and trigger parameters named only in Algorithm 2's
    "Require" line are not used in its steps, so the loop below rebalances
    daily, exactly as described.

    Causality note (Algorithm 2, line 7: "Let x_t be the most recent feature
    vector (constructed using data up to t-1). Let X_train be all prior
    feature vectors."): because each row of the feature matrix built by
    `src.features.build_feature_matrix` for calendar day d already embeds
    information through day d itself (it contains r_d), the feature vector
    satisfying "constructed using data up to t-1" for decision day t is the
    row at t-1, i.e. `X_all[pos - 1]`, not `X_all[pos]`. Querying
    `X_all[pos]` would embed today's own realized return r_t into the KNN
    query used to size today's position -- a one-day look-ahead leak. The
    training set `X_train` is "all prior feature vectors", i.e. strictly
    before that query row: `X_all[:pos - 1]`.
    """
    knn_cfg: KNNRegimeConfig = config.knn
    returns, features = _prepare_features(prices, config)
    assets = list(prices.columns)
    n_assets = len(assets)

    dates = features.index
    X_all = features.values
    R_all = returns.loc[dates].values  # same-date return r_t aligned with feature row x_t
    split_date = pd.Timestamp(config.train_test_split)
    test_dates = dates[dates > split_date]
    if len(test_dates) == 0:
        raise ValueError("no out-of-sample dates after train_test_split")

    date_to_pos = {d: i for i, d in enumerate(dates)}
    w_prev = np.full(n_assets, 1.0 / n_assets)

    weight_records = []
    for t in test_dates:
        pos = date_to_pos[t]
        query_pos = pos - 1  # x_t: most recent feature row built from info up to t-1
        H_t_X = X_all[:query_pos]  # X_train: all feature vectors strictly prior to x_t
        H_t_R = R_all[:query_pos]
        if query_pos < 1 or H_t_X.shape[0] < max(knn_cfg.lookback, knn_cfg.n_neighbors):
            w_t = w_prev
        else:
            x_query = X_all[query_pos]
            mu_t, sigma_t = knn_conditional_moments(H_t_X, H_t_R, x_query, knn_cfg.n_neighbors)
            w_t = solve_tc_aware_mvo(
                mu_t, sigma_t, w_prev, knn_cfg.risk_aversion, knn_cfg.transaction_cost, knn_cfg.max_weight
            )
        weight_records.append(w_t)
        w_prev = w_t

    weights_df = pd.DataFrame(weight_records, index=test_dates, columns=assets)
    portfolio_returns = (weights_df * returns.loc[test_dates]).sum(axis=1)
    portfolio_returns.name = "portfolio_return"

    metrics = summarize_performance(portfolio_returns, weights_df)
    logger.info("KNN backtest: %d OOS days, metrics=%s", len(test_dates), metrics)

    return {"weights": weights_df, "returns": portfolio_returns, "metrics": metrics}
