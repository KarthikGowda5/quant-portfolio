# Project 1 — Short-Horizon Crypto Alpha: From Signal Discovery to Execution Failure

## Overview

This project presents a complete, end-to-end quantitative research pipeline investigating short-horizon return predictability in cryptocurrency markets. Using BTCUSDT data at 1-second resolution, I build microstructure-inspired features, train interpretable models under strict walk-forward validation, and evaluate economic viability under realistic execution assumptions.

The project intentionally emphasizes **research discipline over PnL optimization**. While statistically significant predictive signals are identified, most fail economically once execution costs are applied. The final result is a well-supported negative conclusion—an outcome that closely mirrors real-world quantitative research.

---

## Key Research Question

**Can short-horizon return predictability in BTC be monetized after accounting for realistic transaction costs and execution constraints?**

---

## Pipeline Summary

### 1. Data Ingestion & Quality Control
- Binance trade and bar data
- 1-second resolution, UTC-aligned
- Explicit checks for:
  - Missing timestamps
  - Duplicate bars
  - OHLC consistency
  - Volume and trade-count sanity

### 2. Feature Engineering
- Short-horizon returns and volatility
- Activity and liquidity proxies
- Price-range and spread-based measures
- Fully causal, no lookahead bias

### 3. Target Construction
- Forward log-returns and directional labels
- Multiple horizons (1s → 180s)
- Strict timestamp alignment

### 4. Modeling
- Logistic regression (deliberately simple and interpretable)
- Daily walk-forward retraining
- Fully out-of-sample evaluation

### 5. Execution-Aware Backtesting
- Latency modeling
- Taker fees and spread costs
- Position-based transaction costs
- Turnover-driven economics

---

## Core Findings

### ✅ Predictability Exists
- Out-of-sample AUC consistently ~0.6
- Stable across days and horizons
- Signal quality is real

### ❌ Profitability Does Not
- High-frequency execution leads to extreme turnover
- Transaction costs overwhelm gross returns
- Latency severely degrades performance

### 🔧 Execution Mitigations Help (But Don’t Fully Save It)
Tested systematically:
- Minimum holding periods
- Probability threshold hysteresis
- Volatility regime gating
- Coarser trading frequencies (10s, 30s)

The most effective mitigation was aligning trading frequency with signal horizon.

---

## Final Result (Pre-Committed Policy)

- Bar size: **30 seconds**
- Trade horizon: **90 seconds**
- Minimum holding: **90 seconds**
- Thresholds: **0.65 / 0.35**
- Evaluation: **14-day out-of-sample**

**Outcome:**
- Small, bounded losses
- Very low turnover
- Slightly negative expected PnL

This confirms that the signal, while real, is too weak to monetize under conservative assumptions.

---

## Why This Project Matters

- Demonstrates **full-stack quant research**, not just modeling
- Shows **why** signals fail, not just that they do
- Emphasizes execution realism
- Avoids overfitting and cherry-picking
- Mirrors real professional quant workflows

A negative result, rigorously established, is a *successful research outcome*.

---

## Repository Structure

- scripts/ **End-to-end pipeline scripts**
- src/ **Modular research code**
- data/ **Raw and processed datasets**
- reports/ **CSV outputs and figures**

---

## Next Steps

This project motivates future work on:
- Longer-horizon signals
- Cross-sectional crypto strategies
- Order-book–level features
- Regime-aware model design

---

## Disclaimer

This project is for research and educational purposes only. It is not investment advice.
