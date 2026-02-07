from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.costs import CostParams, cost_series


@dataclass(frozen=True)
class BacktestParams:
    p_enter_long: float = 0.55
    p_enter_short: float = 0.45
    latency_sec: int = 1
    costs: CostParams = CostParams()


def prob_to_position(p: pd.Series, enter_long: float, enter_short: float) -> pd.Series:
    pos = pd.Series(0.0, index=p.index)
    pos[p > enter_long] = 1.0
    pos[p < enter_short] = -1.0
    return pos


def run_backtest_1s(
    df: pd.DataFrame,
    signal: pd.DataFrame,
    params: BacktestParams,
    prob_col: str = "p_up_5s",
) -> pd.DataFrame:
    """
    df: features_targets dataframe (must include ts, ret_1s, hl_spread)
    signal: OOS signal dataframe (ts, p_up_5s)
    """
    # Align signal onto df timestamps (inner join)
    d = df.merge(signal[["ts", prob_col]], on="ts", how="inner").copy()
    d = d.sort_values("ts").reset_index(drop=True)

    p = d[prob_col].astype("float64")
    pos_raw = prob_to_position(p, params.p_enter_long, params.p_enter_short)

    # Latency: execute position with delay (shift forward)
    delay = int(params.latency_sec)
    pos = pos_raw.shift(delay).fillna(0.0)

    # Gross return (use log-return approximation already in ret_1s)
    r = d["ret_1s"].astype("float64").fillna(0.0)
    pnl_gross = pos * r

    # Costs
    c = cost_series(d, pos, params.costs, spread_col="hl_spread")

    pnl_net = pnl_gross - c

    out = pd.DataFrame(
        {
            "ts": d["ts"],
            "p": p,
            "pos": pos,
            "ret_1s": r,
            "pnl_gross": pnl_gross,
            "cost": c,
            "pnl_net": pnl_net,
        }
    )

    out["equity"] = out["pnl_net"].cumsum()
    return out

