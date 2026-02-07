from __future__ import annotations

import _bootstrap  # noqa: F401

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import BacktestParams, run_backtest_1s
from src.config import DAY_UTC, PATHS, SYMBOL
from src.evaluation.metrics import summarize_backtest


def load_inputs(symbol: str, day_utc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_features_targets.parquet")
    sig = pd.read_parquet(PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_signal_oos.parquet")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True)
    return df, sig


def plot_equity(bt: pd.DataFrame, out_path: Path, title: str) -> None:
    fig = plt.figure()
    plt.plot(bt["ts"], bt["equity"])
    plt.title(title)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity (cumulative log-return units)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sharpe_per_day(pnl_net: pd.Series, seconds_per_step: int = 1) -> float:
    # day scaling is more interpretable for 1-second data
    r = pnl_net.astype(float)
    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    if sig == 0:
        return float("nan")
    steps_per_day = (24 * 3600) / float(seconds_per_step)
    return (mu / sig) * (steps_per_day ** 0.5)


def main() -> None:
    symbol, day_utc = SYMBOL, DAY_UTC
    df, sig = load_inputs(symbol, day_utc)

    fig_dir = PATHS.reports_figures / "project1"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Baseline run
    params = BacktestParams(latency_sec=1)
    bt = run_backtest_1s(df, sig, params)

    out_bt = PATHS.data_processed / "project1" / f"{symbol.lower()}_{day_utc}_backtest.parquet"
    bt.to_parquet(out_bt, index=False)

    summ = summarize_backtest(bt, seconds_per_step=1)
    print("SUMMARY", summ)

    plot_equity(bt, fig_dir / "equity_curve.png", f"{symbol} {day_utc} — Equity Curve (latency={params.latency_sec}s)")
    print(f"WROTE {out_bt}")
    print(f"WROTE {fig_dir / 'equity_curve.png'}")

    # Latency sweep
    latencies = [0, 1, 2, 5, 10]
    rows = []
    for L in latencies:
        p = BacktestParams(latency_sec=L)
        b = run_backtest_1s(df, sig, p)
        rows.append(
            {
                "latency_sec": L,
                "total_pnl_net": float(b["pnl_net"].sum()),
                "total_cost": float(b["cost"].sum()),
                "sharpe_day": float(sharpe_per_day(b["pnl_net"], seconds_per_step=1)),
                "turnover": float(b["pos"].diff().abs().fillna(0.0).sum()),
            }
        )
    sweep = pd.DataFrame(rows).sort_values("latency_sec")
    sweep_path = Path("reports") / "project1_latency_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    fig = plt.figure()
    plt.plot(sweep["latency_sec"], sweep["sharpe_day"], marker="o")
    plt.title(f"{symbol} {day_utc} — Sharpe (per-day scaling) vs Latency")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Sharpe (daily scaling)")
    plt.tight_layout()
    fig.savefig(fig_dir / "sharpe_vs_latency.png", dpi=150)
    plt.close(fig)

    print(f"WROTE {sweep_path}")
    print(f"WROTE {fig_dir / 'sharpe_vs_latency.png'}")
    print(sweep)


if __name__ == "__main__":
    main()
