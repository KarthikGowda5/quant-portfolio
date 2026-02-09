from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import _bootstrap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PATHS, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 1 multi-day stability evaluation (day walk-forward AUC).")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (optional, inclusive)")
    p.add_argument("--horizons", default="1,5,10", help="Comma-separated horizons in seconds (default: 1,5,10)")
    p.add_argument("--train_window_days", type=int, default=0, help="If >0, use only last N prior days for training.")
    p.add_argument("--out", default="reports/project1_multiday_stability.csv")
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
    """
    Looks for:
      {symbol_lower}_{YYYY-MM-DD}_features_targets.parquet
    """
    prefix = f"{symbol.lower()}_"
    suffix = "_features_targets.parquet"
    for fp in sorted(processed_dir.glob(f"{prefix}*{suffix}")):
        name = fp.name
        ymd = name[len(prefix) : len(prefix) + 10]
        try:
            yield date.fromisoformat(ymd), fp
        except ValueError:
            continue


def load_day(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def split_Xy(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Features: everything except ts and y_*
    feature_cols = [c for c in df.columns if c != "ts" and not c.startswith("y_")]
    target_col = f"y_up_{horizon}s"
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    X = df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    y = df[target_col].to_numpy(dtype=np.int8, copy=False)
    return X, y, feature_cols


def realized_vol_from_ret_1s(df: pd.DataFrame) -> float:
    # Simple daily realized vol proxy from 1-second log returns.
    # Use sqrt(mean(ret^2)) as stable scale proxy (works even with slight drift).
    r = df["ret_1s"].to_numpy(dtype=np.float64, copy=False)
    return float(np.sqrt(np.mean(r * r)))


def make_model() -> Pipeline:
    # Standard baseline; keep it deterministic and stable.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )


def main() -> None:
    args = parse_args()
    start = to_date(args.start)
    end = to_date(args.end)
    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())

    ensure_dirs()

    processed_dir = PATHS.data_processed / "project1"
    day_files = [(d, fp) for (d, fp) in iter_day_files(processed_dir, args.symbol) if in_range(d, start, end)]
    day_files.sort(key=lambda x: x[0])

    if not day_files:
        raise SystemExit(f"No *_features_targets.parquet files found in {processed_dir} for symbol={args.symbol}")

    print(f"[stability] processed_dir={processed_dir}")
    print(f"[stability] days={len(day_files)} first={day_files[0][0]} last={day_files[-1][0]}")
    print(f"[stability] horizons={horizons} train_window_days={args.train_window_days}")

    # Preload all days (small range is fine; if you scale to months, switch to lazy loading)
    data_by_day: dict[date, pd.DataFrame] = {d: load_day(fp) for d, fp in day_files}

    rows: list[dict[str, object]] = []

    # Day-walk-forward: for each day t, train on prior days (< t), test on day t
    for i, (d_test, _fp) in enumerate(day_files):
        df_test = data_by_day[d_test]
        rv = realized_vol_from_ret_1s(df_test)
        n_test = int(len(df_test))

        # choose training days
        prior_days = [d for (d, _fp2) in day_files[:i]]
        if args.train_window_days and args.train_window_days > 0:
            prior_days = prior_days[-args.train_window_days :]

        base = {
            "day": d_test.isoformat(),
            "n_test": n_test,
            "realized_vol_proxy": rv,
            "n_train_days": len(prior_days),
        }

        if not prior_days:
            # Not enough history to train; record NaNs.
            for h in horizons:
                base[f"auc_{h}s"] = np.nan
                base[f"pos_rate_{h}s"] = float(df_test[f"y_up_{h}s"].mean())
            rows.append(base)
            print(f"[stability] {d_test} train_days=0 -> skip AUC (need prior days)")
            continue

        df_train = pd.concat([data_by_day[d] for d in prior_days], axis=0, ignore_index=True)

        for h in horizons:
            X_tr, y_tr, feat_cols = split_Xy(df_train, h)
            X_te, y_te, _ = split_Xy(df_test, h)

            model = make_model()
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)[:, 1]

            # AUC requires both classes present in test set; handle edge case.
            if len(np.unique(y_te)) < 2:
                auc = np.nan
            else:
                auc = float(roc_auc_score(y_te, p))

            base[f"auc_{h}s"] = auc
            base[f"pos_rate_{h}s"] = float(np.mean(y_te))

        rows.append(base)
        aucs_str = " ".join([f"AUC{h}={base[f'auc_{h}s']:.4f}" if pd.notna(base[f"auc_{h}s"]) else f"AUC{h}=nan" for h in horizons])
        print(f"[stability] {d_test} train_days={len(prior_days)} {aucs_str}")

    out_df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)

    # Regime labeling by realized vol across evaluated days (quantiles)
    # (Quantiles are computed on available values; stable even if first day has NaN AUC.)
    q25 = float(out_df["realized_vol_proxy"].quantile(0.25))
    q75 = float(out_df["realized_vol_proxy"].quantile(0.75))

    def label_regime(x: float) -> str:
        if x <= q25:
            return "low_vol"
        if x >= q75:
            return "high_vol"
        return "mid_vol"

    out_df["vol_regime"] = out_df["realized_vol_proxy"].apply(label_regime)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[stability] wrote: {out_path}")

    # Small, useful rollups (no plots yet)
    # Fraction of days with AUC > 0.5 for each horizon (ignoring NaNs)
    for h in horizons:
        s = out_df[f"auc_{h}s"].dropna()
        if len(s) == 0:
            continue
        frac = float((s > 0.5).mean())
        print(f"[stability] frac_days_auc_gt_0.5 horizon={h}s: {frac:.3f} (n={len(s)})")


if __name__ == "__main__":
    main()
