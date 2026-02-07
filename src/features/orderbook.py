from __future__ import annotations

import pandas as pd


def build_orderbook_proxies_from_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback-mode proxies when order book snapshots are unavailable.
    Input df must contain: ts, open, high, low, close
    Returns a features df with ts and proxy columns.
    """
    out = pd.DataFrame({"ts": pd.to_datetime(df["ts"], utc=True)}).copy()

    o = df["open"].astype("float64")
    h = df["high"].astype("float64")
    l = df["low"].astype("float64")
    c = df["close"].astype("float64")

    # Spread proxy: normalized high-low range
    out["hl_spread"] = (h - l) / c

    # Price pressure proxy: close-open move
    out["co_move"] = (c - o) / c

    # Gap / jump proxy: open vs previous close
    prev_c = c.shift(1)
    out["gap_1s"] = (o - prev_c) / prev_c

    # Smoothed range proxy (volatility-like)
    out["range_vol_60s"] = out["hl_spread"].rolling(60, min_periods=60).mean()

    return out

