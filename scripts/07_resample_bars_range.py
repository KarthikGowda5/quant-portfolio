from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import PATHS, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resample processed 1s bars to coarser bars by day.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (optional, inclusive)")
    p.add_argument("--freqs", default="10S,30S", help="Comma-separated pandas freqs (e.g., 10S,30S)")
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


def iter_days(processed_dir: Path, symbol: str) -> Iterable[date]:
    prefix = f"{symbol.lower()}_"
    suffix = "_bars_1s.parquet"
    for fp in sorted(processed_dir.glob(f"{prefix}*{suffix}")):
        ymd = fp.name[len(prefix) : len(prefix) + 10]
        try:
            yield date.fromisoformat(ymd)
        except ValueError:
            continue


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # Assumes df has ts (UTC), open/high/low/close/volume/trade_count
    d = df.sort_values("ts").set_index("ts")

    o = d["open"].resample(freq, label="left", closed="left").first()
    h = d["high"].resample(freq, label="left", closed="left").max()
    l = d["low"].resample(freq, label="left", closed="left").min()
    c = d["close"].resample(freq, label="left", closed="left").last()
    v = d["volume"].resample(freq, label="left", closed="left").sum()
    n = d["trade_count"].resample(freq, label="left", closed="left").sum()

    out = pd.concat({"open": o, "high": h, "low": l, "close": c, "volume": v, "trade_count": n}, axis=1)

    # Drop empty bins (where open is NaN)
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def main() -> None:
    args = parse_args()
    start = to_date(args.start)
    end = to_date(args.end)
    freqs = [f.strip().lower() for f in args.freqs.split(",") if f.strip()]


    ensure_dirs()
    processed_dir = PATHS.data_processed / "project1"

    days = sorted([d for d in set(iter_days(processed_dir, args.symbol)) if in_range(d, start, end)])
    print(f"[resample] days={len(days)} freqs={freqs}")

    for d in days:
        day = d.isoformat()
        in_path = processed_dir / f"{args.symbol.lower()}_{day}_bars_1s.parquet"
        df = pd.read_parquet(in_path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        for freq in freqs:
            tag = freq.lower()
            out_path = processed_dir / f"{args.symbol.lower()}_{day}_bars_{tag}.parquet"
            if out_path.exists() and out_path.stat().st_size > 10_000:
                print(f"[skip] {day} {out_path.name}")
                continue

            try:
                r = resample_ohlcv(df, freq)
                r["ts"] = pd.to_datetime(r["ts"], utc=True)
                r["ts_ms"] = (r["ts"].astype("int64") // 1_000_000).astype("int64")
                r["symbol"] = args.symbol
                r = r[["ts", "ts_ms", "open", "high", "low", "close", "volume", "trade_count", "symbol"]]

                r.to_parquet(out_path, index=False)
                print(f"[ok] {day} freq={freq} rows={len(r):,} -> {out_path.name}")
            except Exception as e:
                print(f"[failed] {day} freq={freq}: {e}")
                if args.strict:
                    raise


if __name__ == "__main__":
    main()
