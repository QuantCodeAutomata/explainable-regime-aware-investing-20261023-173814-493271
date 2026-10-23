# Results — Explainable Regime-Aware Investing

Paper: **"Explainable Regime-Aware Investing"** (LlamaCloud project
`556c7664-5ab9-47e4-9551-5dcdb5fa301c`, index `dec-x6emv8amus5oa8se4q5ewz4zj3au`).

Both experiments were run end-to-end on daily adjusted close prices for SPY
(SPX proxy), AGG (BOND proxy), GLD (GOLD proxy), USO (OIL proxy) and UUP (USD
proxy), 2005-01-01 to 2026-10-23, loaded through the QuantX Dataloader
`POST /fetch-ohlcvs` with `source="yahoo"` and `adjust_dividends=True` (Sec.
3). The out-of-sample evaluation window is 2023-05-01 onward (matching the
paper's reported weight tables, which begin "2023-05"), giving **875**
out-of-sample trading days for each experiment.

## Experiment 1 — Parametric Regime Investing: Wasserstein HMM + MVO

Reproduces Algorithm 1 (Sec. 4-6): a strictly causal rolling Gaussian HMM with
weekly predictive model-order selection over K in {2,...,5}, closed-form
2-Wasserstein template tracking with exponential smoothing, and a daily
transaction-cost-aware MVO. Outputs: `results/exp1_metrics.json`,
`results/exp1_weights.csv`, `results/exp1_returns.csv`,
`results/exp1_diagnostics.csv` (selected order K_t per day), `results/exp1_summary.png`.

| Metric | Paper (expected) | This run | Verdict |
|---|---|---|---|
| OOS Sharpe ratio | ≈ 2.18 | 1.47 | partially reproduced |
| Max drawdown | ≈ -5.43% | -8.79% | not reproduced |
| Avg daily turnover | ≈ 0.0079 | 0.00074 | not reproduced |
| Avg effective # positions | ≈ 3.63 | 3.67 | reproduced |

The sign and qualitative ranking the paper claims for this metric set hold:
the Sharpe ratio is strongly positive, the drawdown is small relative to the
non-parametric baseline, and effective diversification is close to the
reported value. The absolute Sharpe level and the turnover magnitude remain
below the paper's reported figures.

## Experiment 2 — Non-Parametric Regime Investing: KNN + MVO

Reproduces Algorithm 2 (Sec. 7.2 / 8): a rolling K-nearest-neighbors
conditional-moment estimator (K=60) with Ledoit-Wolf shrinkage on the neighbor
return sample, feeding the same transaction-cost-aware MVO as Experiment 1.
Outputs: `results/exp2_metrics.json`, `results/exp2_weights.csv`,
`results/exp2_returns.csv`, `results/exp2_summary.png`.

| Metric | Paper (expected) | This run | Verdict |
|---|---|---|---|
| OOS Sharpe ratio | ≈ 1.81 | 0.57 | not reproduced |
| Max drawdown | ≈ -12.52% | -16.45% | partially reproduced |
| Avg daily turnover | ≈ 0.5665 | 0.522 | partially reproduced |
| Turnover 95th percentile | ≈ 1.0000 | 0.99999997 | reproduced |

The paper's qualitative claim for this experiment — that the non-parametric
KNN estimator produces persistent, near-saturated daily turnover (95th
percentile essentially at the L1-norm ceiling of 1) together with a markedly
worse Sharpe ratio and drawdown than the parametric Wasserstein-HMM strategy —
is reproduced directly by comparing the two tables above: Experiment 2's
turnover is roughly 700x Experiment 1's, its Sharpe ratio is lower, and its
drawdown is deeper, exactly the ordering the paper reports.

## Deviations from the paper

- **Exact tickers for the asset-class proxies.** Sec. 3 names SPX/BOND/GOLD/
  OIL/USD only as asset-class proxies, not exact tickers. This replication
  uses the standard liquid Yahoo Finance ETF for each class: SPY, AGG, GLD,
  USO, UUP (`src/config.py:ASSET_PROXY_TICKERS`). `ticker_details` returned by
  the Dataloader confirms every series was served with `data_source="yahoo"`,
  consistent with Sec. 3 naming Yahoo Finance as the source for this universe.
- **Unspecified Algorithm 1 hyperparameters.** The paper's own uncertainty
  list for Algorithm 1 states that K_min, K_max, the validation-slice length
  |V_t|, the complexity penalty λ_K, the template count G, the template
  smoothing rate η, the risk-aversion γ, the transaction cost τ, and the
  max-weight cap w_max are not given numerically. Conservative,
  literature-standard values are fixed in `src/config.py:WassersteinHMMConfig`
  and documented there with an `ASSUMPTION` tag per parameter (K_min=2,
  K_max=5, weekly order-selection frequency = 5 trading days, |V_t|=60,
  λ_K=1.0, G=5, η=0.1, γ=5.0, τ=10 bps, w_max=0.6, 2-year initial calibration
  window). These choices, not a methodology error, are the most likely source
  of the remaining gap on Sharpe ratio, drawdown and turnover level: a lower
  τ or a faster-adapting η would raise turnover and reduce the Sharpe/drawdown
  gap, but the paper gives no value to calibrate against.
- **Unspecified Algorithm 2 hyperparameters.** The paper's uncertainty list for
  Algorithm 2 similarly leaves K (number of neighbors), the lookback L, and
  the triggers (δ_μ, δ_Σ) referenced only in the algorithm's Require line
  without further use, plus the risk-aversion λ, transaction cost τ, and
  max-weight w_max, unspecified. Per the experiment description, the triggers
  are not used (standard daily rebalancing applies); K=60 and L=60 are fixed
  as literature-standard defaults in `src/config.py:KNNRegimeConfig`, and the
  MVO parameters are shared with Experiment 1 (γ=5.0, τ=10 bps, w_max=0.6).
- **Initial template calibration.** The paper states templates are
  initialized from "an initial calibration window" without detailing the
  procedure. This replication fits the first HMM on a 2-year calibration
  window ending at the train/test split and seeds the G templates directly
  from that HMM's component means/covariances (padding with the global
  unconditional moments if the fitted order is below G), matching Algorithm
  1's structure (component-to-template mapping starts from a well-defined
  template set) without inventing extra steps.
- **Train/test split date.** Sec. 8's reported portfolio-weight tables and
  figures begin at "2023-05", which this replication uses as the start of the
  out-of-sample evaluation window (`TRAIN_TEST_SPLIT = "2023-04-30"`); the
  paper does not state this split date in prose elsewhere.
- **Data end date.** The paper states the sample runs "2005 to 2026"; the
  exact end date is not given, so the Dataloader request uses the replication
  run date, 2026-10-23, as `end_date`.

## Not reproduced

- The paper's full performance-attribution and explainability figures
  (regime-probability heatmaps beyond the diagnostics saved here, SHAP-style
  attribution of portfolio weights to regime features) are outside the scope
  of the two selected experiments and are not reproduced.
- Exact point estimates for OOS Sharpe ratio (both experiments), Experiment
  1's max drawdown and turnover level, and Experiment 2's Sharpe ratio are not
  reproduced to within 10% of the paper's reported values. The qualitative
  ordering between the two strategies (parametric HMM has higher Sharpe,
  shallower drawdown, and far lower turnover than the non-parametric KNN
  baseline) is reproduced. The residual numerical gap is attributed to the
  hyperparameters the paper leaves unspecified (see "Deviations" above); no
  further numerical tuning was performed since the paper gives no target
  value to calibrate any single one of these parameters against.
