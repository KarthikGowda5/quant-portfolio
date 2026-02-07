from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PATHS, SYMBOL, DAY_UTC


@dataclass(frozen=True)
class TargetsReport:
    n_rows_in: int
    n_rows_out: int
    horizons_sec: tuple[int, ...]
    out_path: str


def load_features(symbol: str, day_utc: str) -> pd.DataFrame:
    path = PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_prices(symbol: str, day_utc: str) -> pd.DataFrame:
    path = PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_bars_1s.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Bars file not found: {path}")
    df = pd.read_parquet(path)[["ts", "close"]]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["close"] = df["close"].astype("float64")
    return df


def make_forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    # forward log return over horizon seconds
    return np.log(close.shift(-horizon)) - np.log(close)


def build_targets(symbol: str = SYMBOL, day_utc: str = DAY_UTC, horizons_sec: tuple[int, ...] = (1, 5, 10)) -> TargetsReport:
    feats = load_features(symbol, day_utc)
    prices = load_prices(symbol, day_utc)

    # Align close prices onto feature timestamps
    df = feats.merge(prices, on="ts", how="left")
    n_in = int(len(df))

    for h in horizons_sec:
        df[f"y_ret_{h}s"] = make_forward_returns(df["close"], h)
        df[f"y_up_{h}s"] = (df[f"y_ret_{h}s"] > 0).astype("int8")

    # Drop rows that can't have forward returns
    df = df.drop(columns=["close"]).dropna().reset_index(drop=True)
    n_out = int(len(df))

    out_dir = PATHS.data_processed / "project1"
    out_path = out_dir / f"{symbol.lower()}_{day_utc}_features_targets.parquet"
    df.to_parquet(out_path, index=False)

    return TargetsReport(
        n_rows_in=n_in,
        n_rows_out=n_out,
        horizons_sec=horizons_sec,
        out_path=str(out_path),
    )
