from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PATHS, SYMBOL, DAY_UTC, ensure_dirs
from src.features.orderbook import build_orderbook_proxies_from_bars
from src.features.trades import build_bar_features_1s


@dataclass(frozen=True)
class FeaturesReport:
    n_rows_in: int
    n_rows_out: int
    n_features: int
    out_path: str


def load_processed_bars(symbol: str, day_utc: str) -> pd.DataFrame:
    path = PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_bars_1s.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed bars not found: {path}")
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def write_features(df_feat: pd.DataFrame, symbol: str, day_utc: str) -> Path:
    out_dir = PATHS.data_processed / "project1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol.lower()}_{day_utc}_features.parquet"
    df_feat.to_parquet(out_path, index=False)
    return out_path


def build_features(symbol: str = SYMBOL, day_utc: str = DAY_UTC) -> FeaturesReport:
    ensure_dirs()
    bars = load_processed_bars(symbol, day_utc)
    n_in = int(len(bars))

    f_trades = build_bar_features_1s(bars)
    f_book = build_orderbook_proxies_from_bars(bars)

    # Merge on ts (strict inner join to keep aligned rows)
    feat = f_trades.merge(f_book, on="ts", how="inner")

    # Drop rows with NaNs from rolling windows / diffs
    feat = feat.dropna().reset_index(drop=True)

    out_path = write_features(feat, symbol, day_utc)
    n_out = int(len(feat))
    n_features = int(feat.shape[1] - 1)  # excluding ts

    return FeaturesReport(
        n_rows_in=n_in,
        n_rows_out=n_out,
        n_features=n_features,
        out_path=str(out_path),
    )
