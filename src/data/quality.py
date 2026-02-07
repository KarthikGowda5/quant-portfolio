from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PATHS, SYMBOL, DAY_UTC, ensure_dirs


@dataclass(frozen=True)
class QualityReport:
    n_rows: int
    n_missing_bins: int
    start_ts: str
    end_ts: str
    hl_spread_p50: float
    hl_spread_p95: float


def load_processed_bars_1s(symbol: str, day_utc: str) -> pd.DataFrame:
    path = PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_bars_1s.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def missing_bins_1s(df: pd.DataFrame) -> int:
    t0 = df["ts"].min()
    t1 = df["ts"].max()
    full = pd.date_range(start=t0, end=t1, freq="1s", tz="UTC")
    return int(len(full.difference(df["ts"])))


def plot_series(df: pd.DataFrame, col: str, out_path: Path, title: str) -> None:
    fig = plt.figure()
    plt.plot(df["ts"], df[col])
    plt.title(title)
    plt.xlabel("Time (UTC)")
    plt.ylabel(col)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hl_spread_proxy(df: pd.DataFrame, out_path: Path) -> tuple[float, float]:
    # Proxy spread (fallback bars): normalized high-low range
    hl_spread = (df["high"] - df["low"]) / df["close"]
    p50 = float(hl_spread.quantile(0.50))
    p95 = float(hl_spread.quantile(0.95))

    fig = plt.figure()
    plt.hist(hl_spread.clip(upper=hl_spread.quantile(0.999)), bins=100)
    plt.title("Spread Proxy (High-Low / Close) — clipped at 99.9%")
    plt.xlabel("(high - low) / close")
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return p50, p95


def run_quality(symbol: str = SYMBOL, day_utc: str = DAY_UTC) -> QualityReport:
    ensure_dirs()
    df = load_processed_bars_1s(symbol, day_utc)

    out_dir = PATHS.reports_figures / "project1"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_series(
        df,
        col="volume",
        out_path=out_dir / "activity_volume.png",
        title=f"{symbol} {day_utc} — 1s Volume",
    )
    plot_series(
        df,
        col="trade_count",
        out_path=out_dir / "activity_trades.png",
        title=f"{symbol} {day_utc} — 1s Trade Count",
    )

    p50, p95 = plot_hl_spread_proxy(df, out_dir / "hl_spread_proxy.png")
    n_missing = missing_bins_1s(df)

    return QualityReport(
        n_rows=int(len(df)),
        n_missing_bins=n_missing,
        start_ts=str(df["ts"].min()),
        end_ts=str(df["ts"].max()),
        hl_spread_p50=p50,
        hl_spread_p95=p95,
    )
