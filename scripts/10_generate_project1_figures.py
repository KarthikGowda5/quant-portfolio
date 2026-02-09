from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
REPORTS = Path("reports")
FIG_DIR = REPORTS / "figures" / "project1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent plotting style
plt.style.use("seaborn-v0_8-whitegrid")


# ============================================================
# Figure 1 — Predictive Stability (AUC vs Day)
# ============================================================
def fig1_predictive_stability():
    df = pd.read_csv(REPORTS / "project1_multiday_stability.csv")

    fig, ax = plt.subplots(figsize=(8, 4))

    for h in [1, 5, 10]:
        ax.plot(
            df["day"],
            df[f"auc_{h}s"],
            marker="o",
            label=f"{h}s horizon",
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Out-of-Sample Predictive Stability (AUC)")
    ax.set_ylabel("AUC")
    ax.set_xlabel("Day")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out = FIG_DIR / "fig1_predictive_stability_auc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")


# ============================================================
# Figure 2 — Latency Sensitivity
# ============================================================
def fig2_latency_sensitivity():
    df = pd.read_csv(REPORTS / "project1_latency_sweep.csv")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        df["latency_sec"],
        df["sharpe_day"],
        marker="o",
    )

    ax.set_title("Latency Sensitivity of Strategy Performance")
    ax.set_xlabel("Execution Latency (seconds)")
    ax.set_ylabel("Sharpe (daily scaling)")
    fig.tight_layout()

    out = FIG_DIR / "fig2_latency_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")


# ============================================================
# Figure 3 — Turnover Explosion at 1s Frequency
# ============================================================
def fig3_turnover_1s():
    df = pd.read_csv(REPORTS / "project1_multiday_pnl.csv")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(df["day"], df["turnover"])
    ax.set_title("Daily Turnover — 1s Trading Frequency")
    ax.set_ylabel("Turnover (absolute position changes)")
    ax.set_xlabel("Day")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = FIG_DIR / "fig3_turnover_1s.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")


# ============================================================
# Figure 4 — Cost vs Trading Frequency (1s / 10s / 30s)
# ============================================================
def fig4_cost_vs_frequency():
    files = {
        "10s": REPORTS / "pnl_10s_nogate.csv",
        "30s": REPORTS / "pnl_30s_nogate.csv",
    }

    rows = []
    for k, path in files.items():
        df = pd.read_csv(path)
        rows.append(
            {
                "bar": k,
                "mean_cost": df["total_cost"].mean(),
                "mean_turnover": df["turnover"].mean(),
            }
        )

    out_df = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.bar(out_df["bar"], out_df["mean_cost"])
    ax1.set_title("Transaction Cost vs Trading Frequency")
    ax1.set_ylabel("Average Daily Cost")
    ax1.set_xlabel("Bar Size")

    fig.tight_layout()
    out = FIG_DIR / "fig4_cost_vs_frequency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")


# ============================================================
# Figure 5 — Final Policy Daily PnL (30s)
# ============================================================
def fig5_final_policy_pnl():
    df = pd.read_csv(
        REPORTS / "pnl_30s_finalpolicy_2024-01-01_2024-01-15.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(df["day"], df["total_pnl_net"])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Final Policy — Daily Out-of-Sample PnL (30s Bars)")
    ax.set_ylabel("Daily Net PnL")
    ax.set_xlabel("Day")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = FIG_DIR / "fig5_final_policy_pnl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")


# ============================================================
# Main
# ============================================================
def main():
    fig1_predictive_stability()
    fig2_latency_sensitivity()
    fig3_turnover_1s()
    fig4_cost_vs_frequency()
    fig5_final_policy_pnl()


if __name__ == "__main__":
    main()

