from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PATHS, SYMBOL, DAY_UTC, ensure_dirs


@dataclass(frozen=True)
class IngestReport:
    n_rows: int
    n_dupe_ts: int
    n_missing_bins: int
    start_ts: str
    end_ts: str


def load_raw_bars_1s(symbol: str, day_utc: str) -> pd.DataFrame:
    raw_path = PATHS.data_raw / "project1" / f"{symbol.lower()}_{day_utc}_bars_1s.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    df = pd.read_parquet(raw_path)

    required = {"ts", "ts_ms", "open", "high", "low", "close", "volume", "trade_count", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw bars: {missing}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["ts_ms"] = df["ts_ms"].astype("int64")

    # Sort + drop exact duplicate rows
    df = df.sort_values("ts_ms").drop_duplicates().reset_index(drop=True)
    return df


def diagnose_1s_grid(df: pd.DataFrame) -> IngestReport:
    # Duplicate timestamps (after sorting)
    n_dupe_ts = int(df["ts_ms"].duplicated().sum())

    # Missing bins on 1-second grid
    t0 = df["ts"].min()
    t1 = df["ts"].max()
    full = pd.date_range(start=t0, end=t1, freq="1s", tz="UTC")
    observed = df["ts"]
    n_missing_bins = int(len(full.difference(observed)))

    return IngestReport(
        n_rows=int(len(df)),
        n_dupe_ts=n_dupe_ts,
        n_missing_bins=n_missing_bins,
        start_ts=str(t0),
        end_ts=str(t1),
    )


def sanity_checks(df: pd.DataFrame) -> None:
    # OHLC consistency: low <= min(open, close) and high >= max(open, close)
    bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    if bad_low or bad_high:
        raise ValueError(f"OHLC sanity failed: bad_low={int(bad_low)} bad_high={int(bad_high)}")

    if (df["volume"] < 0).any():
        raise ValueError("Volume has negative values.")

    if (df["trade_count"] < 0).any():
        raise ValueError("trade_count has negative values.")

    if not df["ts_ms"].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic increasing after sort.")


def write_processed(df: pd.DataFrame, symbol: str, day_utc: str) -> Path:
    out_dir = PATHS.data_processed / "project1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol.lower()}_{day_utc}_bars_1s.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def ingest_project1_day(symbol: str = SYMBOL, day_utc: str = DAY_UTC) -> tuple[Path, IngestReport]:
    ensure_dirs()
    df = load_raw_bars_1s(symbol, day_utc)
    sanity_checks(df)
    report = diagnose_1s_grid(df)
    out_path = write_processed(df, symbol, day_utc)
    return out_path, report
