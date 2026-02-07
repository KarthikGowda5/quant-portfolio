from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostParams:
    taker_fee_bps: float = 4.0          # typical spot fee scale (configure later)
    spread_mult: float = 1.0            # multiply spread proxy to stress assumptions
    slippage_bps: float = 0.0           # extra per-trade slippage in bps


def bps_to_frac(x_bps: float) -> float:
    return x_bps * 1e-4


def compute_cost_frac(
    trade_abs: np.ndarray,
    hl_spread: np.ndarray,
    params: CostParams,
) -> np.ndarray:
    """
    Compute per-step cost in fraction-of-notional terms.

    trade_abs: abs(position_change) per step (e.g., from 0->1 is 1, 1->-1 is 2)
    hl_spread: proxy spread (high-low)/close for that bar
    """
    # Taker fee applies per unit traded
    fee = bps_to_frac(params.taker_fee_bps) * trade_abs

    # Spread: crossing half-spread each trade (proxy)
    spread_cost = 0.5 * (hl_spread * params.spread_mult) * trade_abs

    # Extra slippage in bps
    slip = bps_to_frac(params.slippage_bps) * trade_abs

    return fee + spread_cost + slip


def cost_series(
    df: pd.DataFrame,
    pos: pd.Series,
    params: CostParams,
    spread_col: str = "hl_spread",
) -> pd.Series:
    """
    df must contain spread_col aligned to pos index.
    pos is the position time series (-1, 0, +1). Costs are charged on position changes.
    """
    trade = pos.diff().fillna(0.0)
    trade_abs = trade.abs().to_numpy(dtype=float)

    hl = df[spread_col].to_numpy(dtype=float)
    c = compute_cost_frac(trade_abs=trade_abs, hl_spread=hl, params=params)
    return pd.Series(c, index=df.index, name="cost_frac")
