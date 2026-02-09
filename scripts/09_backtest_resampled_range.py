from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PATHS, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest resampled bars (10s/30s) with optional regime gating.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--bar", choices=["10s", "30s"], default="10s")

    # which target to trade (in seconds, must exist in file)
    p.add_argument("--trade_horizon_sec", type=int, default=None)

    # execution knobs
    p.add_argument("--p_enter_long", type=float, default=0.60)
    p.add_argument("--p_enter_short", type=float, default=0.40)
    p.add_argument("--min_hold_sec", type=int, default=0)

    # cost knobs (simple, but consistent)
    p.add_argument("--taker_fee_bps", type=float, default=4.0)
    p.add_argument("--spread_mult", type=float, default=1.0)

    # regime gating
    p.add_argument("--gate", choices=["none", "high"], default="none")
    p.add_argument("--gate_q", type=float, default=0.75)

    p.add_argument("--train_window_days", type=int, default=0)
    p.add_argument("--out_csv", default="reports/project1_resampled_pnl.csv")
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


def iter_day_files(processed_dir: Path, symbol: str, bar: str) -> Iterable[tuple[date, Path]]:
    prefix = f"{symbol.lower()}_"
    suffix = f"_features_targets_{bar}.parquet"
    for fp in sorted(processed_dir.glob(f"{prefix}*{suffix}")):
        ymd = fp.name[len(prefix) : len(prefix) + 10]
        try:
            yield date.fromisoformat(ymd), fp
        except ValueError:
            continue


def infer_step_seconds(bar: str) -> int:
    return int(bar[:-1])  # "10s" -> 10


def load_day(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )


def split_Xy(df: pd.DataFrame, trade_horizon_sec: int) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    feature_cols = [c for c in df.columns if c != "ts" and not c.startswith("y_")]
    ycol = f"y_up_{trade_horizon_sec}s"
    if ycol not in df.columns:
        raise KeyError(f"Missing target column: {ycol}")
    X = df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    y = df[ycol].to_numpy(dtype=np.int8, copy=False)
    return X, y, feature_cols, ycol


def prob_to_position(p: np.ndarray, enter_long: float, enter_short: float) -> np.ndarray:
    pos = np.zeros_like(p, dtype=np.float64)
    pos[p > enter_long] = 1.0
    pos[p < enter_short] = -1.0
    return pos


def apply_min_hold(pos: np.ndarray, min_hold_steps: int) -> np.ndarray:
    if min_hold_steps <= 0:
        return pos

    out = np.zeros_like(pos, dtype=np.float64)
    hold = 0
    cur = 0.0
    for i in range(len(pos)):
        desired = float(pos[i])
        if hold > 0:
            out[i] = cur
            hold -= 1
            continue
        if desired != cur:
            cur = desired
            hold = min_hold_steps - 1
        out[i] = cur
    return out


def trade_costs(pos: np.ndarray, close: np.ndarray, hl_range: np.ndarray, taker_fee_bps: float, spread_mult: float) -> np.ndarray:
    """
    Simple cost model per step:
      cost = turnover * (fee + spread_proxy)
    turnover is abs(delta position).
    fee applied in log-return units: bps * 1e-4
    spread proxy uses hl_range as a rough per-bar half-spread proxy (scaled).
    """
    dp = np.abs(np.diff(pos, prepend=0.0))
    fee = (taker_fee_bps * 1e-4) * dp

    # hl_range = log(high/low). Half-spread proxy is some fraction of this.
    spread = (spread_mult * 0.5 * hl_range) * dp
    return fee + spread


def main() -> None:
    args = parse_args()
    start = to_date(args.start)
    end = to_date(args.end)

    ensure_dirs()

    step_sec = infer_step_seconds(args.bar)
    processed_dir = PATHS.data_processed / "project1"

    day_files = [(d, fp) for (d, fp) in iter_day_files(processed_dir, args.symbol, args.bar) if in_range(d, start, end)]
    day_files.sort(key=lambda x: x[0])
    if not day_files:
        raise SystemExit("No resampled features_targets files found.")

    # default trade horizon: 3 bars ahead (consistent with earlier horizons list)
    if args.trade_horizon_sec is None:
        trade_h = 3 * step_sec
    else:
        trade_h = int(args.trade_horizon_sec)

    min_hold_steps = int(args.min_hold_sec // step_sec)

    print(f"[res_bt] bar={args.bar} step_sec={step_sec} trade_h={trade_h}s min_hold={args.min_hold_sec}s ({min_hold_steps} steps)")
    print(f"[res_bt] gate={args.gate} gate_q={args.gate_q} thresholds=({args.p_enter_long},{args.p_enter_short})")

    data_by_day = {d: load_day(fp) for d, fp in day_files}

    rows: list[dict[str, object]] = []

    for i, (d_test, _fp) in enumerate(day_files):
        day = d_test.isoformat()
        prior_days = [d for (d, _fp2) in day_files[:i]]
        if args.train_window_days and args.train_window_days > 0:
            prior_days = prior_days[-args.train_window_days :]

        if not prior_days:
            print(f"[res_bt] {day}: train_days=0 -> skip")
            continue

        df_train = pd.concat([data_by_day[d] for d in prior_days], axis=0, ignore_index=True)
        df_test = data_by_day[d_test].copy()

        # regime gating threshold computed on TRAIN only
        gate_thresh = None
        if args.gate == "high":
            gate_thresh = float(df_train["vol_60s"].quantile(args.gate_q))
            allow = (df_test["vol_60s"].astype(float) >= gate_thresh).to_numpy()
        else:
            allow = np.ones(len(df_test), dtype=bool)

        X_tr, y_tr, feat_cols, ycol = split_Xy(df_train, trade_h)
        X_te, y_te, _feat_cols2, _ycol2 = split_Xy(df_test, trade_h)

        model = make_model()
        model.fit(X_tr, y_tr)
        p_up = model.predict_proba(X_te)[:, 1]

        # position rule + gating
        pos_raw = prob_to_position(p_up, args.p_enter_long, args.p_enter_short)
        pos_raw = np.where(allow, pos_raw, 0.0)

        # min-hold
        pos = apply_min_hold(pos_raw, min_hold_steps)

        # returns and costs
        r = df_test["ret_1"].astype("float64").to_numpy()
        pnl_gross = pos * r
        c = trade_costs(
            pos=pos,
            close=df_test.get("close", pd.Series(np.nan, index=df_test.index)).astype("float64").to_numpy()
            if "close" in df_test.columns
            else np.full(len(df_test), np.nan),
            hl_range=df_test["hl_range"].astype("float64").to_numpy(),
            taker_fee_bps=float(args.taker_fee_bps),
            spread_mult=float(args.spread_mult),
        )
        pnl_net = pnl_gross - c

        turnover = float(np.abs(np.diff(pos, prepend=0.0)).sum())
        total_pnl = float(np.sum(pnl_net))
        total_cost = float(np.sum(c))

        rows.append(
            {
                "day": day,
                "bar": args.bar,
                "step_sec": step_sec,
                "trade_horizon_sec": trade_h,
                "n_train_days": len(prior_days),
                "gate": args.gate,
                "gate_q": args.gate_q,
                "gate_thresh": gate_thresh,
                "p_enter_long": args.p_enter_long,
                "p_enter_short": args.p_enter_short,
                "min_hold_sec": args.min_hold_sec,
                "turnover": turnover,
                "total_cost": total_cost,
                "total_pnl_net": total_pnl,
            }
        )

        print(f"[res_bt] {day} pnl={total_pnl:.6f} cost={total_cost:.6f} turnover={turnover:.1f} gate={args.gate}")

    out_df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[res_bt] wrote: {out_path}")

    if not out_df.empty:
        frac_pos = float((out_df["total_pnl_net"] > 0).mean())
        print(f"[res_bt] frac_positive_days={frac_pos:.3f} (n={len(out_df)})")


if __name__ == "__main__":
    main()
