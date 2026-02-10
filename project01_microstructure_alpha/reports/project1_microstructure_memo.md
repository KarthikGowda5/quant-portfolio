# Project 1 — Crypto Microstructure Alpha (Research Prototype)

## Objective
Evaluate short-horizon predictability in BTCUSDT using high-frequency public data and an execution-aware backtest. The goal is to demonstrate a realistic research workflow (data engineering → features → time-series evaluation → cost/latency stress tests), not production HFT.

## Data
- Instrument: BTCUSDT (spot)
- Sample: 1 UTC day (2024-01-01)
- Source: Binance public historical data (binance.vision)

### Note on data access and limitations
The live Binance REST API endpoint (`api.binance.com`) returned HTTP 451 in the development environment, so the pipeline uses a free fallback: **1-second klines** from `data.binance.vision`. This implies:
- No full order book snapshots (no true microprice or book imbalance).
- No per-trade direction labels; trade imbalance is proxied.
- Spread is proxied using the normalized high–low range.

Despite these limitations, the pipeline remains useful for studying signal decay, time-series evaluation, and sensitivity to execution assumptions.

## Feature Set (1-second bars)
Features are derived from OHLCV and activity:
- Returns: `ret_1s`, `ret_5s`, `ret_10s` (log returns)
- Volatility proxy: `vol_60s` (rolling std of 1s returns)
- Signed volume proxy: `dvol_10s` (volume × sign(ret_1s), rolling)
- Activity proxy: `trade_count_10s`
- Spread proxy: `hl_spread = (high - low) / close`
- Pressure proxy: `co_move = (close - open) / close`
- Gap proxy: `gap_1s = (open - prev_close) / prev_close`
- Range proxy: `range_vol_60s` (rolling mean of hl_spread)

Outputs:
- `data/processed/project1/*_features.parquet`
- Diagnostics plots in `reports/figures/project1/`

## Targets
Forward-direction labels and forward returns were created for multiple horizons:
- `y_up_1s`, `y_up_5s`, `y_up_10s`
- `y_ret_1s`, `y_ret_5s`, `y_ret_10s`

Targets are constructed strictly forward-looking and aligned by timestamp.

## Modeling and Validation
Baseline model: Logistic Regression (fast and interpretable).

Validation protocol:
- Walk-forward cross-validation with an embargo gap to reduce leakage from overlapping horizons.
- Out-of-sample predictions are generated fold-by-fold and concatenated into an OOS signal series.

### Out-of-sample metrics (mean across folds)
Directional AUC (higher is better):
- 1s horizon: ~0.59
- 5s horizon: ~0.62
- 10s horizon: ~0.61

This indicates measurable short-horizon predictability in this single-day sample.

## Execution-Aware Backtest
A minimal backtest converts OOS probabilities into positions using thresholds:
- Long if p > 0.55
- Short if p < 0.45
- Flat otherwise

Execution assumptions:
- Latency delay applied to position changes (baseline 1s; swept 0–10s)
- Cost model includes:
  - taker fee (bps)
  - spread cost using a proxy (hl_spread)
  - optional slippage stress parameter

Artifacts:
- `data/processed/project1/*_backtest.parquet`
- `reports/figures/project1/equity_curve.png`
- `reports/figures/project1/sharpe_vs_latency.png`
- `reports/project1_latency_sweep.csv`

### Result (single-day, proxy-cost model)
- Gross signal shows predictive skill, but net PnL is negative after costs in this configuration.
- Total costs dominate net performance, consistent with microstructure research where small edges are easily consumed by execution frictions.

## Robustness and Failure Analysis
Key failure modes observed/expected:
1. **Costs overwhelm edge**: with 1-second data and threshold-based trading, turnover is high and spread/fees dominate.
2. **Latency sensitivity**: performance degrades as latency increases, consistent with short-horizon signal decay.
3. **Proxy limitations**: spread and order-book dynamics are approximated; true L1/L2 order book features may materially change results.
4. **Sample size**: results are from a single day; stability across days/regimes is not yet demonstrated.

## Next Steps
- Expand to multiple days and test stability across volatility regimes.
- Replace proxies with true order book + trades when accessible via archival datasets.
- Improve execution modeling:
  - maker/taker logic
  - queue/fill modeling (for market making in Project 2)
  - more realistic slippage models tied to volatility/liquidity
- Add robustness checks:
  - sensitivity to thresholds and holding rules
  - alternative target horizons
  - subsample and regime-conditioned evaluation
