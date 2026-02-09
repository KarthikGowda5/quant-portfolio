# scripts/06_backtest_range.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.backtest.engine import BacktestParams, run_backtest_1s
from src.config import PATHS, ensure_dirs
from src.evaluation.metrics import summarize_backtest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 1 multi-day: train->signal->backtest by day (walk-forward).")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (optional, inclusive)")
    p.add_argument("--trade_horizon", type=int, default=5, choices=[1, 5, 10], help="Which y_up_{h}s to trade.")
    p.add_argument("--train_window_days", type=int, default=0, help="If >0, train only on last N prior days.")
    p.add_argument("--latency_sec", type=int, default=1)
    p.add_argument("--fees_bps", type=float, default=None, help="Override fee bps if BacktestParams supports it.")
    p.add_argument("--spread_bps", type=float, default=None, help="Override spread bps if BacktestParams supports it.")
    p.add_argument("--out_csv", default="reports/project1_multiday_pnl.csv")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--min_hold_sec", type=int, default=0, help="Minimum holding time in seconds (0 = off).")
    p.add_argument("--p_enter_long", type=float, default=0.55)
    p.add_argument("--p_enter_short", type=float, default=0.45)
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


def iter_day_files(processed_dir: Path, symbol: str) -> Iterable[tuple[date, Path]]:
    prefix = f"{symbol.lower()}_"
    suffix = "_features_targets.parquet"
    for fp in sorted(processed_dir.glob(f"{prefix}*{suffix}")):
        ymd = fp.name[len(prefix) : len(prefix) + 10]
        try:
            yield date.fromisoformat(ymd), fp
        except ValueError:
            continue


def load_day(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def split_Xy(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in df.columns if c != "ts" and not c.startswith("y_")]
    ycol = f"y_up_{horizon}s"
    if ycol not in df.columns:
        raise KeyError(f"Missing target column: {ycol}")
    X = df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    y = df[ycol].to_numpy(dtype=np.int8, copy=False)
    return X, y, feature_cols


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )


def write_signal(symbol: str, day_utc: str, ts: pd.Series, p_up: np.ndarray, horizon: int) -> Path:
    out_dir = PATHS.data_processed / "project1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol.lower()}_{day_utc}_signal_oos.parquet"

    col = f"p_up_{horizon}s"

    sig = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, utc=True),
            col: p_up.astype("float64"),
        }
    )

    # Ensure stable sort + no duplicates
    sig = sig.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)

    sig.to_parquet(out_path, index=False)
    return out_path



def sharpe_per_day(pnl_net: pd.Series, seconds_per_step: int = 1) -> float:
    r = pnl_net.astype(float)
    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    if sig == 0:
        return float("nan")
    steps_per_day = (24 * 3600) / float(seconds_per_step)
    return (mu / sig) * (steps_per_day ** 0.5)


def main() -> None:
    args = parse_args()
    start = to_date(args.start)
    end = to_date(args.end)

    ensure_dirs()

    processed_dir = PATHS.data_processed / "project1"
    day_files = [(d, fp) for (d, fp) in iter_day_files(processed_dir, args.symbol) if in_range(d, start, end)]
    day_files.sort(key=lambda x: x[0])

    if not day_files:
        raise SystemExit(f"No *_features_targets.parquet found in {processed_dir} for symbol={args.symbol}")

    print(f"[range_bt] days={len(day_files)} first={day_files[0][0]} last={day_files[-1][0]}")
    print(f"[range_bt] trade_horizon={args.trade_horizon}s train_window_days={args.train_window_days} latency_sec={args.latency_sec}")

    # preload small ranges; if you scale up, change to lazy loading
    data_by_day: dict[date, pd.DataFrame] = {d: load_day(fp) for d, fp in day_files}

    # Backtest params
    params = BacktestParams(
        p_enter_long=float(args.p_enter_long),
        p_enter_short=float(args.p_enter_short),
        latency_sec=int(args.latency_sec),
        min_hold_sec=int(args.min_hold_sec),
    )


    # Optional overrides if your BacktestParams has these fields
    if args.fees_bps is not None and hasattr(params, "fees_bps"):
        setattr(params, "fees_bps", float(args.fees_bps))
    if args.spread_bps is not None and hasattr(params, "spread_bps"):
        setattr(params, "spread_bps", float(args.spread_bps))

    rows: list[dict[str, object]] = []

    for i, (d_test, _fp) in enumerate(day_files):
        day = d_test.isoformat()
        df_test = data_by_day[d_test]

        prior_days = [d for (d, _fp2) in day_files[:i]]
        if args.train_window_days and args.train_window_days > 0:
            prior_days = prior_days[-args.train_window_days :]

        if not prior_days:
            print(f"[range_bt] {day}: train_days=0 -> skip (need prior days)")
            continue

        df_train = pd.concat([data_by_day[d] for d in prior_days], axis=0, ignore_index=True)

        # train model & predict probs for tradable horizon
        X_tr, y_tr, _ = split_Xy(df_train, args.trade_horizon)
        X_te, y_te, _ = split_Xy(df_test, args.trade_horizon)

        model = make_model()
        model.fit(X_tr, y_tr)
        p_up = model.predict_proba(X_te)[:, 1]

        sig_path = write_signal(args.symbol, day, df_test["ts"], p_up, horizon=args.trade_horizon)


        # run backtest
        sig_df = pd.read_parquet(sig_path)
        sig_df["ts"] = pd.to_datetime(sig_df["ts"], utc=True)
        bt = run_backtest_1s(df_test, sig_df, params)


        out_bt = processed_dir / f"{args.symbol.lower()}_{day}_backtest.parquet"
        bt.to_parquet(out_bt, index=False)

        summ = summarize_backtest(bt, seconds_per_step=1)

        total_pnl = float(bt["pnl_net"].sum())
        total_cost = float(bt["cost"].sum())
        turnover = float(bt["pos"].diff().abs().fillna(0.0).sum())
        sharpe_d = float(sharpe_per_day(bt["pnl_net"], seconds_per_step=1))

        row = {
            "day": day,
            "n_train_days": len(prior_days),
            "trade_horizon_sec": args.trade_horizon,
            "latency_sec": args.latency_sec,
            "total_pnl_net": total_pnl,
            "total_cost": total_cost,
            "sharpe_day": sharpe_d,
            "turnover": turnover,
            "signal_path": str(sig_path),
            "backtest_path": str(out_bt),
            "summary": str(summ),
            "params": str(asdict(params)) if hasattr(params, "__dataclass_fields__") else str(params),
        }
        rows.append(row)

        print(f"[range_bt] {day} pnl={total_pnl:.6f} cost={total_cost:.6f} sharpe_day={sharpe_d:.3f} turnover={turnover:.1f}")

    out_df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[range_bt] wrote: {out_path}")

    if not out_df.empty:
        frac_pos = float((out_df["total_pnl_net"] > 0).mean())
        print(f"[range_bt] frac_positive_days={frac_pos:.3f} (n={len(out_df)})")


if __name__ == "__main__":
    main()


