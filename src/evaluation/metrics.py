from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestSummary:
    n: int
    total_pnl_net: float
    total_cost: float
    sharpe_ann: float
    max_drawdown: float
    turnover: float


def annualized_sharpe(r: pd.Series, seconds_per_step: int = 1) -> float:
    """
    r: per-step return series (log-return-like).
    For 1s bars: annualization factor uses seconds per year.
    """
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    if sig == 0:
        return float("nan")

    seconds_per_year = 365.0 * 24 * 3600
    steps_per_year = seconds_per_year / float(seconds_per_step)
    return (mu / sig) * float(np.sqrt(steps_per_year))


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min())  # negative number (drawdown)


def turnover(pos: pd.Series) -> float:
    # Sum of absolute position changes
    return float(pos.diff().abs().fillna(0.0).sum())


def summarize_backtest(bt: pd.DataFrame, seconds_per_step: int = 1) -> BacktestSummary:
    pnl_net = bt["pnl_net"].astype(float)
    cost = bt["cost"].astype(float)
    eq = bt["equity"].astype(float)
    pos = bt["pos"].astype(float)

    return BacktestSummary(
        n=int(len(bt)),
        total_pnl_net=float(pnl_net.sum()),
        total_cost=float(cost.sum()),
        sharpe_ann=annualized_sharpe(pnl_net, seconds_per_step=seconds_per_step),
        max_drawdown=max_drawdown(eq),
        turnover=turnover(pos),
    )
