# scripts/03_build_features_range.py
from __future__ import annotations
import _bootstrap
from src.features.build_features import build_features, FeaturesReport
from src.modeling.targets import build_targets, TargetsReport

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

from src.config import PATHS, ensure_dirs


@dataclass
class DayResult:
    day: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Project 1 features over a date range (day-isolated).")
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


def iter_processed_days(processed_dir: Path, symbol: str) -> Iterable[date]:
    """
    Matches processed naming:
      {symbol_lower}_{YYYY-MM-DD}_bars_1s.parquet
    """
    prefix = f"{symbol.lower()}_"
    suffix = "_bars_1s.parquet"
    for fp in sorted(processed_dir.glob(f"{prefix}*{suffix}")):
        name = fp.name
        ymd = name[len(prefix) : len(prefix) + 10]
        try:
            yield date.fromisoformat(ymd)
        except ValueError:
            continue


def features_path(symbol: str, day_utc: str) -> Path:
    return PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_features.parquet"


def features_targets_path(symbol: str, day_utc: str) -> Path:
    return PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_features_targets.parquet"


def exists_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10_000


def main() -> None:
    args = parse_args()
    start = to_date(args.start)
    end = to_date(args.end)

    ensure_dirs()

    processed_dir = PATHS.data_processed / "project1"
    days_all = sorted(set(iter_processed_days(processed_dir, args.symbol)))
    days = [d for d in days_all if in_range(d, start, end)]

    print(f"[features_range] processed_dir={processed_dir}")
    print(f"[features_range] days_to_process={len(days)} first={days[0] if days else None} last={days[-1] if days else None}")

    results: list[DayResult] = []

    for d in days:
        day = d.isoformat()
        f_path = features_path(args.symbol, day)
        ft_path = features_targets_path(args.symbol, day)

        if exists_ok(f_path) and exists_ok(ft_path):
            results.append(DayResult(day, "skipped", "already built"))
            print(f"[skip] {day}")
            continue

        # Placeholder: we will wire in your real single-day feature builder here.
        try:
            fr: FeaturesReport = build_features(symbol=args.symbol, day_utc=day)
            tr: TargetsReport = build_targets(symbol=args.symbol, day_utc=day)

            results.append(DayResult(day, "ok"))
            print(
                f"[ok] {day} "
                f"features(rows_in={fr.n_rows_in:,} rows_out={fr.n_rows_out:,} n_feat={fr.n_features}) "
                f"targets(rows_in={tr.n_rows_in:,} rows_out={tr.n_rows_out:,} horizons={tr.horizons_sec})"
            )
        except Exception as e:
            results.append(DayResult(day, "failed", repr(e)))
            print(f"[failed] {day}: {e}")
            if args.strict:
                raise


    ok = sum(r.status == "ok" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    failed = sum(r.status == "failed" for r in results)
    print(f"[features_range] done days={len(results)} ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
