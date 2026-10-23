"""Experiment 2 -- Non-Parametric Regime Investing: KNN + MVO.

Reproduces Algorithm 2 of "Explainable Regime-Aware Investing": a rolling
K-nearest-neighbors conditional-moment estimator with Ledoit-Wolf shrinkage,
feeding the same transaction-cost-aware mean-variance optimizer as
Experiment 1, to isolate the economic role of parametric regime inference
(Sec. 7.2 / 8).

Run with: python -m exp.exp2_knn_mvo
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
from src.backtest import run_knn_backtest

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

    result = run_knn_backtest(prices, config)

    weights_path = RESULTS_DIR / "exp2_weights.csv"
    returns_path = RESULTS_DIR / "exp2_returns.csv"
    metrics_path = RESULTS_DIR / "exp2_metrics.json"

    result["weights"].to_csv(weights_path)
    result["returns"].to_csv(returns_path)
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2, default=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    (1 + result["returns"]).cumprod().plot(ax=axes[0], title="KNN strategy: cumulative wealth")
    axes[0].set_ylabel("Wealth (OOS, start=1)")

    result["weights"].plot.area(ax=axes[1], title="Portfolio weights")
    axes[1].set_ylabel("Weight")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_summary.png", dpi=150)
    plt.close(fig)

    logger.info("Experiment 2 metrics: %s", result["metrics"])
    return result["metrics"]


if __name__ == "__main__":
    main()
