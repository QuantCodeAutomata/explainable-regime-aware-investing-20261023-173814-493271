"""Experiment 1 -- Parametric Regime Investing: Wasserstein HMM + MVO.

Reproduces Algorithm 1 of "Explainable Regime-Aware Investing": a strictly
causal rolling Gaussian HMM with predictive model-order selection and
persistent regime identity via 2-Wasserstein template tracking, feeding a
transaction-cost-aware mean-variance optimizer (Sec. 5-6).

Run with: python -m exp.exp1_wasserstein_hmm
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import load_price_data
from src.config import DEFAULT_CONFIG
from src.backtest import run_wasserstein_hmm_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def main() -> dict:
    config = DEFAULT_CONFIG
    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info(
        "Loading price data: tickers=%s, window=%s..%s",
        config.tickers, config.data_start, config.data_end,
    )
    prices = load_price_data(config.tickers, config.data_start, config.data_end)
    prices.to_csv(RESULTS_DIR / "exp1_price_panel.csv")

    result = run_wasserstein_hmm_backtest(prices, config)

    weights_path = RESULTS_DIR / "exp1_weights.csv"
    returns_path = RESULTS_DIR / "exp1_returns.csv"
    diag_path = RESULTS_DIR / "exp1_diagnostics.csv"
    metrics_path = RESULTS_DIR / "exp1_metrics.json"

    result["weights"].to_csv(weights_path)
    result["returns"].to_csv(returns_path)
    result["diagnostics"].to_csv(diag_path)
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2, default=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    (1 + result["returns"]).cumprod().plot(ax=axes[0], title="Wasserstein-HMM strategy: cumulative wealth")
    axes[0].set_ylabel("Wealth (OOS, start=1)")

    result["weights"].plot.area(ax=axes[1], title="Portfolio weights")
    axes[1].set_ylabel("Weight")

    result["diagnostics"]["K_t"].plot(ax=axes[2], title="Selected HMM order $K_t$", marker=".", linestyle="none")
    axes[2].set_ylabel("$K_t$")
    axes[2].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_summary.png", dpi=150)
    plt.close(fig)

    logger.info("Experiment 1 metrics: %s", result["metrics"])
    return result["metrics"]


if __name__ == "__main__":
    main()
