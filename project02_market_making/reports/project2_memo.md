# Project 02 — Simulated Market Making, Inventory Risk & Adverse Selection

## Research Question
In a controlled simulation, how do spread width, price volatility, and inventory-aware quoting impact:

- Spread capture (realized spread)
- Adverse selection
- Inventory risk and mark-to-market losses

The objective is diagnostic understanding, not PnL maximization.

---

## Simulator Overview (What Is Modeled)

### Price Process
- Midprice follows a random walk:
  mid_{t+1} = mid_t + mid_vol · N(0,1)
- Extended with persistent directional regimes via a latent state.

### Quoting Model
- Baseline symmetric quotes:
  bid = mid − spread/2 − skew  
  ask = mid + spread/2 − skew
- Inventory skew:
  skew = inv_skew · inventory  
  Long inventory shifts quotes downward to encourage selling.

### Order Flow
- Market order arrivals are probabilistic.
- Fill probability decays exponentially with quote distance:
  p_hit = base_rate · exp(−hit_decay · distance)

### Inventory & PnL
- Inventory capped by inventory_limit
- Mark-to-market PnL:
  PnL = cash + inventory · mid

---

## Metrics

### Per-Trade (One-Step Horizon)
- Realized spread: effective spread captured after the immediate post-trade move
- Adverse selection: post-trade mid-price movement against the market maker
- Net capture:
  net_capture = realized_spread − adverse_selection

### Inventory Risk
- inv_std: standard deviation of inventory
- inv_max_abs: maximum absolute inventory

---

## Diagnostics & Outputs

Artifacts written to reports/:

- sim_mm_trade_metrics.csv: per-trade realized spread and adverse selection
- sim_mm_sweep.csv: parameter sweep over spread × mid_vol
- net_capture_vs_vol.png: net capture vs volatility
- sim_mm_inventory_skew.csv: impact of inventory skew on risk and PnL

---

## Baseline Findings (Uninformed Flow)

1. Wider spreads increase realized spread capture.
2. Higher volatility increases adverse selection pressure.
3. Inventory skew reduces inventory variance by altering fill probabilities.
4. Inventory control trades profitability for risk reduction.

These behaviors align with standard market microstructure intuition.

---

## Limitations of the Baseline Model
- No queue priority or partial fills
- No explicit order book state (depth, imbalance)
- IID uninformed order flow
- One-step adverse-selection horizon

This simulator is a diagnostic lab, not a deployable market-making system.

---

## Informed Flow & Regime Stress Tests

To stress the market maker beyond idealized assumptions, the simulator was extended with:

### Informed Traders
- Order flow statistically correlated with future mid-price changes
- Introduces genuine adverse selection

### Persistent Price Regimes
- Latent regime induces directional drift persistence
- Inventory becomes directional exposure rather than mean-reverting noise

### Key Failure Mode
Under informed flow and persistent regimes:
- Realized spread remains positive
- Inventory accumulates in adverse regimes
- Mark-to-market losses dominate
- Naïve symmetric market making becomes unprofitable

---

## Defensive Controls Introduced

### Regime-Aware Risk-Off
- Quoting aggressiveness is reduced during directional regimes
- Trade frequency and adverse selection decline at the cost of lower spread capture

### Inventory-Conditioned Risk-Off
- Risk-off behavior strengthens with inventory magnitude
- Prevents compounding directional exposure once inventory is already large

### Inventory Unwind Bias
- Quotes are biased to favor trades that reduce inventory
- Stabilizes inventory at the cost of lower per-trade edge

---

## Stress Test Conclusions
- Risk-off mechanisms reduce adverse selection and tail losses
- Inventory-conditioned controls materially lower inventory volatility
- Unwind bias stabilizes inventory but can reduce profitability depending on regime

---

## Core Insight
Market making under informed flow is not a static optimization problem.  
It is a dynamic control problem balancing spread capture, inventory risk, and flow toxicity under regime persistence.

---

## Optional Extensions
- Regime-dependent informed flow intensity
- Multi-step adverse-selection horizons
- Latency, cancellation, and queue position modeling
