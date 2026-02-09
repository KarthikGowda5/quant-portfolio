from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import PATHS, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build features+targets for resampled bars (10s/30s) by day.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (optional, inclusive)")
    p.add_argument("--bar", choices=["10s", "30s"], default="10s", help="Which resampled bar file to use.")
    p.add_argument("--horizons_sec", default=None, help="Comma-separated horizons in seconds (optional).")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def to_date(s: Optional[str]) -> Optional[date]:
    if s is None:
        return None
    return date.fromisoformat(s)


def in_range(d: date, start: Optional[date], end: Optional[date]) -> bool:
    if start is not None and d < start:
        return False
    if end is not None and d > end:
        return False
    return True


def iter_days(processed_dir: Path, symbol: str, bar: str) -> Iterable[date]:
    prefix = f"{symbol.lower()}_"
    suffix = f"_bars_{bar}.parquet"
    for fp in sorted(processed_dir.glob(f"{prefix}*{suffix}")):
        ymd = fp.name[len(prefix) : len(prefix) + 10]
        try:
            yield date.fromisoformat(ymd)
        except ValueError:
            continue


def infer_step_seconds(bar: str) -> int:
    # bar like "10s" or "30s"
    if not bar.endswith("s"):
        raise ValueError(f"Unsupported bar tag: {bar}")
    return int(bar[:-1])


def make_forward_log_return(close: pd.Series, steps_ahead: int) -> pd.Series:
    return np.log(close.shift(-steps_ahead)) - np.log(close)


def build_features(df: pd.DataFrame, step_sec: int) -> pd.DataFrame:
    d = df.sort_values("ts").copy()
    close = d["close"].astype("float64")

    # one-bar log return
    d["ret_1"] = np.log(close) - np.log(close.shift(1))
    d["ret_3"] = np.log(close) - np.log(close.shift(3))
    d["ret_6"] = np.log(close) - np.log(close.shift(6))

    # volatility over ~60 seconds worth of bars
    win = max(1, int(60 / step_sec))
    d["vol_60s"] = d["ret_1"].rolling(win).std()

    # range proxy
    d["hl_range"] = np.log(d["high"].astype("float64") / d["low"].astype("float64"))

    # volume and trade intensity proxies
    d["dvol"] = d["volume"].astype("float64").diff()
    d["vol_sum_60s"] = d["volume"].astype("float64").rolling(win).sum()
    d["tc_sum_60s"] = d["trade_count"].astype("float64").rolling(win).sum()

    feat_cols = ["ts", "ret_1", "ret_3", "ret_6", "vol_60s", "hl_range", "dvol", "vol_sum_60s", "tc_sum_60s"]
    out = d[feat_cols].dropna().reset_index(drop=True)
    return out


def build_targets(feat: pd.DataFrame, prices: pd.DataFrame, horizons_sec: list[int], step_sec: int) -> pd.DataFrame:
    # Align close onto feature timestamps
    p = prices[["ts", "close"]].copy()
    p["ts"] = pd.to_datetime(p["ts"], utc=True)
    p["close"] = p["close"].astype("float64")

    df = feat.merge(p, on="ts", how="left")
    for h in horizons_sec:
        steps = int(h / step_sec)
        if steps <= 0 or (steps * step_sec) != h:
            raise ValueError(f"horizon {h}s not divisible by bar step {step_sec}s")
        df[f"y_ret_{h}s"] = make_forward_log_return(df["close"], steps)
        df[f"y_up_{h}s"] = (df[f"y_ret_{h}s"] > 0).astype("int8")

    df = df.drop(columns=["close"]).dropna().reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    start = to_date(args.start)
    end = to_date(args.end)

    ensure_dirs()

    step_sec = infer_step_seconds(args.bar)
    processed_dir = PATHS.data_processed / "project1"

    # default horizons per bar size (kept modest and interpretable)
    if args.horizons_sec is None:
        horizons = [step_sec, 3 * step_sec, 6 * step_sec]  # 1,3,6 bars ahead
    else:
        horizons = [int(x.strip()) for x in args.horizons_sec.split(",") if x.strip()]

    days = sorted([d for d in set(iter_days(processed_dir, args.symbol, args.bar)) if in_range(d, start, end)])
    print(f"[feat_resampled] bar={args.bar} step_sec={step_sec} horizons={horizons} days={len(days)}")

    for d in days:
        day = d.isoformat()
        bars_path = processed_dir / f"{args.symbol.lower()}_{day}_bars_{args.bar}.parquet"
        out_feat = processed_dir / f"{args.symbol.lower()}_{day}_features_{args.bar}.parquet"
        out_ft = processed_dir / f"{args.symbol.lower()}_{day}_features_targets_{args.bar}.parquet"

        if out_feat.exists() and out_ft.exists() and out_feat.stat().st_size > 10_000 and out_ft.stat().st_size > 10_000:
            print(f"[skip] {day}")
            continue

        try:
            bars = pd.read_parquet(bars_path)
            bars["ts"] = pd.to_datetime(bars["ts"], utc=True)

            feat = build_features(bars, step_sec=step_sec)
            feat.to_parquet(out_feat, index=False)

            ft = build_targets(feat, bars, horizons_sec=horizons, step_sec=step_sec)
            ft.to_parquet(out_ft, index=False)

            print(f"[ok] {day} feat_rows={len(feat):,} ft_rows={len(ft):,} -> {out_ft.name}")
        except Exception as e:
            print(f"[failed] {day}: {e}")
            if args.strict:
                raise


if __name__ == "__main__":
    main()

