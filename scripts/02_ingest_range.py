# scripts/02_ingest_range.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import _bootstrap  # keep consistent with your other scripts

from src.config import PATHS
from src.data.ingest import ingest_project1_day


@dataclass
class DayResult:
    day: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Project 1 raw daily bars over a date range.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (optional, inclusive)")
    p.add_argument("--strict", action="store_true", help="Stop on first failure.")
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


def iter_raw_days(raw_dir: Path, symbol: str) -> Iterable[date]:
    """
    Matches your downloader naming:
      {symbol_lower}_{YYYY-MM-DD}_bars_1s.parquet
    """
    prefix = f"{symbol.lower()}_"
    suffix = "_bars_1s.parquet"

    for fp in sorted(raw_dir.glob(f"{prefix}*{suffix}")):
        name = fp.name
        # btcusdt_2024-01-02_bars_1s.parquet -> 2024-01-02
        ymd = name[len(prefix) : len(prefix) + 10]
        try:
            yield date.fromisoformat(ymd)
        except ValueError:
            continue


def processed_exists_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10_000


def main() -> None:
    args = parse_args()
    start = to_date(args.start)
    end = to_date(args.end)

    raw_dir = PATHS.data_raw / "project1"
    processed_dir = PATHS.data_processed / "project1"
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_days = sorted(set(iter_raw_days(raw_dir, args.symbol)))
    days = [d for d in all_days if in_range(d, start, end)]

    print(f"[ingest_range] raw_dir={raw_dir} processed_dir={processed_dir}")
    print(f"[ingest_range] days_to_process={len(days)} first={days[0] if days else None} last={days[-1] if days else None}")

    results: list[DayResult] = []

    for d in days:
        day = d.isoformat()
        out_path = PATHS.data_processed / "project1" / f"{args.symbol.lower()}_{day}_bars_1s.parquet"

        out_path = PATHS.data_processed / "project1" / f"{args.symbol.lower()}_{day}_bars_1s.parquet"
        if out_path.exists() and out_path.stat().st_size > 10_000:
            results.append(DayResult(day=day, status="skipped", reason="already processed"))
            print(f"[skip] {day} {out_path.name}")
            continue

        try:
            out_path2, rep = ingest_project1_day(symbol=args.symbol, day_utc=day)
            results.append(DayResult(day=day, status="ok"))
            print(
                f"[ok] {day} {out_path2.name} "
                f"rows={rep.n_rows:,} dupe_ts={rep.n_dupe_ts} missing_bins={rep.n_missing_bins}"
            )
        except Exception as e:
            results.append(DayResult(day=day, status="failed", reason=repr(e)))
            print(f"[failed] {day}: {e}")
            if args.strict:
                raise


    ok = sum(r.status == "ok" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    failed = sum(r.status == "failed" for r in results)

    print(f"[ingest_range] done days={len(results)} ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
