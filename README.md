# Explainable Regime-Aware Investing — Replication

Paper: **"Explainable Regime-Aware Investing"** (indexed research paper,
LlamaCloud project `556c7664-5ab9-47e4-9551-5dcdb5fa301c`, index
`dec-x6emv8amus5oa8se4q5ewz4zj3au`).

This repository reproduces two of the paper's regime-aware portfolio
experiments:

- **Experiment 1 — Parametric Regime Investing (Algorithm 1, Sec. 4-6):** a
  strictly causal rolling Gaussian Hidden Markov Model with predictive
  model-order selection, persistent regime identity via closed-form
  2-Wasserstein template tracking (Sec. 5.4), and a transaction-cost-aware
  mean-variance optimizer (MVO).
- **Experiment 2 — Non-Parametric Regime Investing (Algorithm 2, Sec. 7.2 /
  8):** a rolling K-nearest-neighbors conditional-moment estimator with
  Ledoit-Wolf covariance shrinkage, feeding the same transaction-cost-aware
  MVO, used as a baseline that isolates the economic role of the parametric
  regime model.

## Repository layout

```
src/     shared library: features, HMM tools, Wasserstein distance,
         regime-template tracking, KNN moment estimator, MVO portfolio
         construction, evaluation metrics, backtest orchestration
exp/     exp1_wasserstein_hmm.py, exp2_knn_mvo.py — run the two experiments
tests/   pytest unit tests for every src/ module
data/    data_loader.py (QuantX Dataloader market data)
results/ metrics, weights, returns, diagnostics, and plots from the final run
```

## Data

Section 3 of the paper specifies daily adjusted close prices from Yahoo
Finance, 2005-2026, for five cross-asset proxies: S&P 500 (SPX), a broad bond
proxy (BOND), Gold (GOLD), Oil (OIL), and the U.S. Dollar (USD). The paper
gives only asset-class names, not exact tickers; this replication maps them to
the standard liquid Yahoo Finance ETF for each class (SPY, AGG, GLD, USO,
UUP — see `src/config.py`).

Price history is loaded through the QuantX Dataloader `POST /fetch-ohlcvs`
route with `source="yahoo"` (the paper names Yahoo Finance itself as the data
source for this universe) and `adjust_dividends=True`. See
`data/data_loader.py` for the exact request and validation logic.

Set the following environment variables before running (see `.env.example`):

```
DATALOADER_API_KEY=...   # https://quantx.limex.pro/
MASSIVE_API_KEY=...      # https://massive.com/ (not required for this Yahoo-sourced load)
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest                    # unit tests (no network calls)
python -m exp.exp1_wasserstein_hmm  # Experiment 1
python -m exp.exp2_knn_mvo          # Experiment 2
```

or simply:

```bash
./reproduce.sh
```

All experiment outputs (`*_metrics.json`, `*_weights.csv`, `*_returns.csv`,
`*_diagnostics.csv`, `*_summary.png`) are written to `results/`.

## Results

See `results/RESULTS.md` for the full comparison against the paper's reported
outcomes, including deviations and unresolved parameters.
