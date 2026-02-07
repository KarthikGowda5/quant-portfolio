from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.config import PATHS, SYMBOL, DAY_UTC, ensure_dirs
from src.modeling.cv import assert_time_ordered, walk_forward_folds


@dataclass(frozen=True)
class TrainConfig:
    n_folds: int = 5
    embargo: int = 10
    target_horizon: int = 5  # seconds
    max_iter: int = 200
    C: float = 1.0


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


def fit_predict_proba(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    model = LogisticRegression(
        max_iter=cfg.max_iter,
        C=cfg.C,
        solver="lbfgs",
        n_jobs=None,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def eval_fold(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    # Guard against degenerate folds
    out: dict[str, float] = {}
    if len(np.unique(y_true)) < 2:
        out["auc"] = np.nan
    else:
        out["auc"] = float(roc_auc_score(y_true, p))

    out["logloss"] = float(log_loss(y_true, p, labels=[0, 1]))
    out["acc"] = float(accuracy_score(y_true, (p >= 0.5).astype(int)))
    return out


def run_oos(symbol: str = SYMBOL, day_utc: str = DAY_UTC, cfg: TrainConfig = TrainConfig()) -> pd.DataFrame:
    ensure_dirs()
    df = load_dataset(symbol, day_utc)
    assert_time_ordered(df)

    feats = feature_columns(df)
    X = df[feats].to_numpy(dtype=float)

    results: list[dict[str, float | int | str]] = []

    horizons = (1, 5, 10)  # decay study
    folds = list(walk_forward_folds(df, n_folds=cfg.n_folds, embargo=cfg.embargo))

    for h in horizons:
        ycol = f"y_up_{h}s"
        y = df[ycol].to_numpy(dtype=int)

        for fold in folds:
            X_train, X_test = X[fold.train_idx], X[fold.test_idx]
            y_train, y_test = y[fold.train_idx], y[fold.test_idx]

            # Standardize using train stats only (time-series safe)
            mu = X_train.mean(axis=0)
            sig = X_train.std(axis=0)
            sig[sig == 0] = 1.0
            X_train_z = (X_train - mu) / sig
            X_test_z = (X_test - mu) / sig

            p = fit_predict_proba(X_train_z, y_train, X_test_z, cfg)
            m = eval_fold(y_test, p)

            results.append(
                {
                    "symbol": symbol,
                    "day_utc": day_utc,
                    "horizon_s": h,
                    "fold": int(fold.fold),
                    "train_n": int(len(fold.train_idx)),
                    "test_n": int(len(fold.test_idx)),
                    **m,
                }
            )

    res = pd.DataFrame(results)
    out_path = Path("reports") / "project1_model_oos.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    return res


if __name__ == "__main__":
    df = run_oos()
    print(df.groupby("horizon_s")[["auc", "logloss", "acc"]].mean(numeric_only=True))
    print("WROTE reports/project1_model_oos.csv")
