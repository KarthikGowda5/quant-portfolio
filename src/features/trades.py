from __future__ import annotations

import numpy as np
import pandas as pd


def build_bar_features_1s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback-mode features from 1-second OHLCV bars.
    Input df must contain: ts, close, volume, trade_count
    Returns a features df with ts and feature columns.
    """
    out = pd.DataFrame({"ts": pd.to_datetime(df["ts"], utc=True)}).copy()

    close = df["close"].astype("float64")
    vol = df["volume"].astype("float64")
    tc = df["trade_count"].astype("float64")

    # 1s log return
    out["ret_1s"] = np.log(close).diff()

    # multi-horizon returns (still based on close)
    out["ret_5s"] = np.log(close).diff(5)
    out["ret_10s"] = np.log(close).diff(10)

    # rolling volatility of 1s returns
    out["vol_60s"] = out["ret_1s"].rolling(60, min_periods=60).std()

    # signed volume proxy using return sign
    out["dvol_10s"] = (vol * np.sign(out["ret_1s"].fillna(0.0))).rolling(10, min_periods=10).sum()

    # activity proxy
    out["trade_count_10s"] = tc.rolling(10, min_periods=10).sum()

    return out
