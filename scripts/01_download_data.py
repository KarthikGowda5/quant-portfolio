
from __future__ import annotations
import _bootstrap 

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import io
import zipfile



def download_fallback_1s_bars(symbol: str, day_utc: str) -> pd.DataFrame:
    """
    Free fallback if api.binance.com is blocked.
    Downloads 1-second klines from data.binance.vision (daily zip).

    Output schema:
      ts, ts_ms, open, high, low, close, volume, trade_count, symbol
    """
    # Example:
    # https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2024-01-01.zip
    url = (
        "https://data.binance.vision/data/spot/daily/klines/"
        f"{symbol}/1s/{symbol}-1s-{day_utc}.zip"
    )

    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        raise RuntimeError(f"BINANCE_VISION_404: {url}")
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    # Usually exactly one CSV inside
    csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    with zf.open(csv_name) as f:
        raw = f.read().decode("utf-8", errors="replace")

    # Binance kline columns (standard)
    cols = [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols)

    df["open"] = df["open"].astype("float64")
    df["high"] = df["high"].astype("float64")
    df["low"] = df["low"].astype("float64")
    df["close"] = df["close"].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    df["trade_count"] = df["trade_count"].astype("int64")

    df["ts_ms"] = df["open_time_ms"].astype("int64")
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["symbol"] = symbol

    # Keep only useful columns for Phase A fallback
    df = df[["ts", "ts_ms", "open", "high", "low", "close", "volume", "trade_count", "symbol"]]
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df



from src.config import (
    BINANCE_BASE_URL,
    DAY_UTC,
    DEPTH_LIMIT,
    ENDPOINT_DEPTH,
    ENDPOINT_TRADES,
    PATHS,
    SNAPSHOT_INTERVAL_SEC,
    SYMBOL,
    ensure_dirs,
)


def utc_day_to_ms(day_utc: str) -> tuple[int, int]:
    dt0 = datetime.fromisoformat(day_utc).replace(tzinfo=timezone.utc)
    dt1 = dt0.replace(hour=23, minute=59, second=59, microsecond=999000)
    start_ms = int(dt0.timestamp() * 1000)
    end_ms = int(dt1.timestamp() * 1000)
    return start_ms, end_ms


def http_get(path: str, params: dict[str, Any], timeout: int = 30) -> Any:
    url = BINANCE_BASE_URL + path
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code == 451:
        raise RuntimeError("BINANCE_BLOCKED_451")
    r.raise_for_status()
    return r.json()



def download_agg_trades(symbol: str, day_utc: str) -> pd.DataFrame:
    """
    Downloads aggTrades for one UTC day using startTime/endTime paging.
    Binance limits: up to 1000 rows per request.
    """
    start_ms, end_ms = utc_day_to_ms(day_utc)

    rows: list[dict[str, Any]] = []
    next_start = start_ms
    n_requests = 0

    while True:
        data = http_get(
            ENDPOINT_TRADES,
            params={
                "symbol": symbol,
                "startTime": next_start,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        n_requests += 1
        if not data:
            break

        rows.extend(data)

        # Advance by last trade time + 1ms to avoid duplicates
        last_ms = int(data[-1]["T"])
        next_start = last_ms + 1

        # Stop if we've reached the end of day
        if last_ms >= end_ms:
            break

        # Light throttling to reduce ban risk
        if n_requests % 20 == 0:
            time.sleep(0.2)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize schema / types
    df = df.rename(
        columns={
            "a": "agg_trade_id",
            "p": "price",
            "q": "qty",
            "f": "first_trade_id",
            "l": "last_trade_id",
            "T": "ts_ms",
            "m": "is_buyer_maker",
            "M": "ignore",
        }
    )
    df["price"] = df["price"].astype("float64")
    df["qty"] = df["qty"].astype("float64")
    df["ts_ms"] = df["ts_ms"].astype("int64")
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.sort_values("ts_ms").reset_index(drop=True)
    df["symbol"] = symbol
    return df


def try_download_book_snapshots(symbol: str, day_utc: str) -> pd.DataFrame:
    """
    Best-effort order book snapshots at SNAPSHOT_INTERVAL_SEC.
    This can hit rate limits; if it fails, we return empty DataFrame.
    """
    start_ms, end_ms = utc_day_to_ms(day_utc)

    # 1 snapshot per interval over the day (can be large). For day-1, keep it limited.
    # We will cap to first N snapshots to avoid bans; expand later.
    max_snaps = 600  # ~10 minutes if interval=1s. Safe starter.
    snaps: list[dict[str, Any]] = []

    t_ms = start_ms
    n = 0
    while t_ms <= end_ms and n < max_snaps:
        try:
            data = http_get(
                ENDPOINT_DEPTH,
                params={"symbol": symbol, "limit": DEPTH_LIMIT},
                timeout=10,
            )
            snaps.append(
                {
                    "ts_ms": t_ms,
                    "ts": pd.to_datetime(t_ms, unit="ms", utc=True),
                    "lastUpdateId": data.get("lastUpdateId"),
                    "bids": data.get("bids"),
                    "asks": data.get("asks"),
                    "symbol": symbol,
                }
            )
        except Exception:
            # If rate limited or blocked, stop and return what we have.
            break

        n += 1
        t_ms += SNAPSHOT_INTERVAL_SEC * 1000
        time.sleep(0.2)  # throttle

    return pd.DataFrame(snaps)


def main() -> None:
    ensure_dirs()

    out_dir = PATHS.data_raw / "project1"
    out_dir.mkdir(parents=True, exist_ok=True)

    sym = SYMBOL
    day = DAY_UTC

    print(f"Downloading aggTrades: symbol={sym} day_utc={day}")
    try:
        trades = download_agg_trades(sym, day)
    except RuntimeError as e:
        if str(e) == "BINANCE_BLOCKED_451":
            print("Binance API blocked (HTTP 451). Using free fallback: 1-second bars.")
            trades = download_fallback_1s_bars(sym, day)
        else:
            raise

    print(f"aggTrades rows: {len(trades):,}")

    kind = "trades" if "price" in trades.columns and "qty" in trades.columns else "bars_1s"
    trades_path = out_dir / f"{sym.lower()}_{day}_{kind}.parquet"

    trades.to_parquet(trades_path, index=False)
    print(f"Wrote: {trades_path}")

    book = pd.DataFrame()
    print("Order book snapshots skipped in fallback mode.")


    meta = {
        "symbol": sym,
        "day_utc": day,
        "trades_rows": int(len(trades)),
        "book_rows": int(len(book)),
        "depth_limit": int(DEPTH_LIMIT),
        "snapshot_interval_sec": int(SNAPSHOT_INTERVAL_SEC),
        "paths": {k: str(v) for k, v in asdict(PATHS).items()},
    }
    meta_path = out_dir / f"{sym.lower()}_{day}_meta.json"
    meta_path.write_text(pd.Series(meta).to_json())
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
