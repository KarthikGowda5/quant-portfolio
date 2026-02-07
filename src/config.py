from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_processed: Path
    data_cache: Path
    reports_figures: Path


def repo_root() -> Path:
    # src/config.py -> src -> repo root
    return Path(__file__).resolve().parents[1]


PATHS = Paths(
    root=repo_root(),
    data_raw=repo_root() / "data" / "raw",
    data_processed=repo_root() / "data" / "processed",
    data_cache=repo_root() / "data" / "cache",
    reports_figures=repo_root() / "reports" / "figures",
)

# Project 1 defaults (Phase A)
SYMBOL = "BTCUSDT"
# Use UTC date string. We'll start with a single day and expand later.
DAY_UTC = "2024-01-01"

# Data cadence for fallback mode (if order book snapshots are blocked)
BAR_SECONDS = 1

# Binance (Spot) endpoints used in Phase A
BINANCE_BASE_URL = "https://api.binance.com"
ENDPOINT_TRADES = "/api/v3/aggTrades"  # aggregated trades
ENDPOINT_DEPTH = "/api/v3/depth"       # order book snapshot (top N levels)

# Order book snapshot settings (research, not production)
DEPTH_LIMIT = 100          # valid values include 100; keep it small to start
SNAPSHOT_INTERVAL_SEC = 1  # target cadence (best-effort; rate limits apply)


def ensure_dirs() -> None:
    PATHS.data_raw.mkdir(parents=True, exist_ok=True)
    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    PATHS.data_cache.mkdir(parents=True, exist_ok=True)
    PATHS.reports_figures.mkdir(parents=True, exist_ok=True)
