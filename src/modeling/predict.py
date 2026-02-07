from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import PATHS, SYMBOL, DAY_UTC, ensure_dirs
from src.modeling.cv import assert_time_ordered, walk_forward_folds


@dataclass(frozen=True)
class SignalReport:
    target: str
    n_rows_signal: int
    out_path: str


def load_dataset(symbol: str, day_utc: str) -> pd.DataFrame:
    path = PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_features_targets.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if (c != "ts") and (not c.startswith("y_"))]


def zscore_fit_transform(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sig = X_train.std(axis=0)
    sig[sig == 0] = 1.0
    return (X_train - mu) / sig, (X_test - mu) / sig


def build_oos_signal(
    symbol: str = SYMBOL,
    day_utc: str = DAY_UTC,
    horizon_s: int = 5,
    n_folds: int = 5,
    embargo: int = 10,
) -> SignalReport:
    ensure_dirs()
    df = load_dataset(symbol, day_utc)
    assert_time_ordered(df)

    ycol = f"y_up_{horizon_s}s"
    feats = feature_columns(df)
    X = df[feats].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=int)

    rows: list[pd.DataFrame] = []
    folds = list(walk_forward_folds(df, n_folds=n_folds, embargo=embargo))

    for fold in folds:
        X_train, X_test = X[fold.train_idx], X[fold.test_idx]
        y_train = y[fold.train_idx]

        X_train_z, X_test_z = zscore_fit_transform(X_train, X_test)

        model = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
        model.fit(X_train_z, y_train)
        p = model.predict_proba(X_test_z)[:, 1]

        part = pd.DataFrame(
            {
                "ts": df.loc[fold.test_idx, "ts"].to_numpy(),
                f"p_up_{horizon_s}s": p.astype("float64"),
                "fold": int(fold.fold),
            }
        )
        rows.append(part)

    sig = pd.concat(rows, axis=0).sort_values("ts").reset_index(drop=True)

    out_dir = PATHS.data_processed / "project1"
    out_path = out_dir / f"{symbol.lower()}_{day_utc}_signal_oos.parquet"
    sig.to_parquet(out_path, index=False)

    return SignalReport(target=ycol, n_rows_signal=int(len(sig)), out_path=str(out_path))


if __name__ == "__main__":
    rep = build_oos_signal()
    print(rep)
