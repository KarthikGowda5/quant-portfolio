from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


# Make repo root importable so shared packages (`scripts`, `src`) work from inside project folders
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts._bootstrap  # noqa: F401

import numpy as np
import pandas as pd

def hit_prob(dist: float, base: float, k: float) -> float:
    """
    dist >= 0 (quote distance from mid). Probability decays with distance.
    base in (0,1). k controls decay speed.
    """
    p = base * np.exp(-k * dist)
    return float(np.clip(p, 0.0, 1.0))



def simulate_mm(
    n_steps: int = 50_000,
    seed: int = 42,
    mid0: float = 100.0,
    mid_vol: float = 0.02,          # per-step mid volatility (in price units)
    spread: float = 0.10,           # quoted spread (price units)
    lambda_buy: float = 0.45,       # prob a buy market order arrives this step
    lambda_sell: float = 0.45,      # prob a sell market order arrives this step
    inventory_limit: int = 10,
    inv_skew: float = 0.0,   # price units per inventory unit
    hit_decay: float = 5.0,   # larger => fills decay faster with distance
    regime_mu: float = 0.0,            # drift per step when regime is active (price units)
    regime_persist: float = 0.99,      # probability regime stays the same each step
    informed_strength: float = 0.0,    # how strongly flow tilts with regime (0 = no informed flow)
    risk_off: float = 0.0,
    risk_off_neutral: float = 0.0,
    risk_off_inv: float = 0.0,
    unwind_strength: float = 0.0,


) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    mid = float(mid0)
    inv = 0
    cash = 0.0

    rows = []
    regime = 0  # -1 sell-informed, +1 buy-informed, 0 neutral

    for t in range(n_steps):
        # Inventory-aware skew: positive inv lowers quotes, negative inv raises quotes
                # Regime update (Markov-like)
        if rng.random() > regime_persist:
            regime = int(rng.choice([-1, 0, 1], p=[0.45, 0.10, 0.45]))

        skew = inv_skew * inv
        bid = mid - spread / 2.0 - skew
        ask = mid + spread / 2.0 - skew
        mid_pre = mid

        ask_dist = max(0.0, ask - mid)
        bid_dist = max(0.0, mid - bid)

        # Informed flow tilt: in buy regime (+1), more buy MOs hit our ask (we sell before price rises)
        # in sell regime (-1), more sell MOs hit our bid (we buy before price falls)
        # Informed flow tilt
        tilt = informed_strength * float(regime)

        lambda_buy_eff = float(np.clip(lambda_buy * (1.0 + tilt), 0.0, 1.0))
        lambda_sell_eff = float(np.clip(lambda_sell * (1.0 - tilt), 0.0, 1.0))

        # Regime-aware risk-off: reduce fills on the toxic side
        # Toxic side = the side that gets hit more in that regime:
        #   regime=+1 (buy-informed): toxic side is our ASK (we sell before price drifts up)
        #   regime=-1 (sell-informed): toxic side is our BID (we buy before price drifts down)
        ro = float(np.clip(risk_off, 0.0, 1.0))
        ro0 = float(np.clip(risk_off_neutral, 0.0, 1.0))
        roi = float(np.clip(risk_off_inv, 0.0, 1.0))

        # inventory-conditioned risk-off (0..1)
        if inventory_limit > 0:
            inv_frac = min(1.0, abs(inv) / float(inventory_limit))
        else:
            inv_frac = 0.0

        ro_eff = float(np.clip(ro + roi * inv_frac, 0.0, 1.0))

        if regime == 1:
            lambda_buy_eff *= (1.0 - ro_eff)     # reduce toxic ask fills
        elif regime == -1:
            lambda_sell_eff *= (1.0 - ro_eff)    # reduce toxic bid fills
        else:
            lambda_buy_eff *= (1.0 - float(np.clip(ro0 + roi * inv_frac, 0.0, 1.0)))
            lambda_sell_eff *= (1.0 - float(np.clip(ro0 + roi * inv_frac, 0.0, 1.0)))

            # Inventory unwind bias: boost the side that reduces |inv|
        uw = float(np.clip(unwind_strength, 0.0, 1.0))
        if inventory_limit > 0:
            inv_frac = min(1.0, abs(inv) / float(inventory_limit))
        else:
            inv_frac = 0.0

        unwind_boost = 1.0 + uw * inv_frac  # in [1, 2] if uw<=1

        if inv > 0:
            # long -> want to SELL -> encourage ask fills (buy MOs hit our ask)
            lambda_buy_eff = float(np.clip(lambda_buy_eff * unwind_boost, 0.0, 1.0))
        elif inv < 0:
            # short -> want to BUY -> encourage bid fills (sell MOs hit our bid)
            lambda_sell_eff = float(np.clip(lambda_sell_eff * unwind_boost, 0.0, 1.0))


        p_buy = hit_prob(ask_dist, base=lambda_buy_eff, k=hit_decay)
        p_sell = hit_prob(bid_dist, base=lambda_sell_eff, k=hit_decay)


        buy_hit = rng.random() < p_buy
        sell_hit = rng.random() < p_sell

        # Inventory constraints: stop quoting the side that would worsen inventory
        if inv >= inventory_limit:
            sell_hit = False  # buying would increase inv further
        if inv <= -inventory_limit:
            buy_hit = False   # selling would decrease inv further (more negative)

        # Executes
        trade_px = np.nan
        trade_side = 0  # +1 means we bought (hit bid), -1 means we sold (hit ask)

        if sell_hit:
            inv += 1
            cash -= bid
            trade_px = bid
            trade_side = +1

        if buy_hit:
            inv -= 1
            cash += ask
            trade_px = ask
            trade_side = -1

        # Mid evolves after potential fills (adverse selection channel)
        mid += (regime_mu * float(regime)) + (mid_vol * float(rng.standard_normal()))

        mid_post = mid

        pnl_mtm = cash + inv * mid

        rows.append(
            {
                "t": t,
                "mid_pre": mid_pre,
                "mid_post": mid_post,
                "mid": mid_post,
                "bid": bid,
                "ask": ask,
                "inv": inv,
                "cash": cash,
                "pnl_mtm": pnl_mtm,
                "trade_px": trade_px,
                "trade_side": trade_side,
                "p_buy": p_buy,
                "p_sell": p_sell,
                "ask_dist": ask_dist,
                "bid_dist": bid_dist,
                "regime": regime,

            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 2: simulated market making lab")
    parser.add_argument("--n_steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mid0", type=float, default=100.0)
    parser.add_argument("--mid_vol", type=float, default=0.02)
    parser.add_argument("--spread", type=float, default=0.10)
    parser.add_argument("--lambda_buy", type=float, default=0.45)
    parser.add_argument("--lambda_sell", type=float, default=0.45)
    parser.add_argument("--inventory_limit", type=int, default=10)
    parser.add_argument("--inv_skew", type=float, default=0.0)
    parser.add_argument("--hit_decay", type=float, default=5.0)

    # Sweep controls (comma-separated lists)
    parser.add_argument("--sweep_spreads", type=str, default="0.05,0.10,0.20")
    parser.add_argument("--sweep_vols", type=str, default="0.01,0.02,0.04")

    # Inventory skew comparison
    parser.add_argument("--skew_values", type=str, default="0.0,0.01,0.02")
    parser.add_argument("--regime_mu", type=float, default=0.0)
    parser.add_argument("--regime_persist", type=float, default=0.99)
    parser.add_argument("--informed_strength", type=float, default=0.0)
    parser.add_argument("--risk_off", type=float, default=0.0, help="0..1: how defensive to be in informed regimes")
    parser.add_argument("--risk_off_neutral", type=float, default=0.0, help="optional: also be defensive in regime=0")
    parser.add_argument("--risk_off_inv", type=float, default=0.0, help="additional risk-off scaled by |inv|/limit (0..1)")
    parser.add_argument("--unwind_strength", type=float, default=0.0, help="0..1: boost unwind-side fills as |inv| approaches limit")

    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    reports_dir = project_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Base run ---
    df = simulate_mm(
        n_steps=args.n_steps,
        seed=args.seed,
        mid0=args.mid0,
        mid_vol=args.mid_vol,
        spread=args.spread,
        lambda_buy=args.lambda_buy,
        lambda_sell=args.lambda_sell,
        inventory_limit=args.inventory_limit,
        inv_skew=args.inv_skew,
        hit_decay=args.hit_decay,
        regime_mu=args.regime_mu,
        regime_persist=args.regime_persist,
        informed_strength=args.informed_strength,
        risk_off=args.risk_off,
        risk_off_neutral=args.risk_off_neutral,
        risk_off_inv=args.risk_off_inv,
        unwind_strength=args.unwind_strength,

    )

    out_csv = reports_dir / "sim_mm_path.csv"
    df.to_csv(out_csv, index=False)
    print(f"WROTE {out_csv}")

    # Trade-level diagnostics (only rows where a trade happened)
    trades = df[df["trade_side"] != 0].copy()
    trades["adverse_selection"] = -trades["trade_side"] * (trades["mid_post"] - trades["mid_pre"])
    trades["realized_spread"] = trades["trade_side"] * (trades["mid_post"] - trades["trade_px"])

    metrics = trades[
        ["t", "trade_side", "trade_px", "mid_pre", "mid_post", "realized_spread", "adverse_selection"]
    ]
    out_trades = reports_dir / "sim_mm_trade_metrics.csv"
    metrics.to_csv(out_trades, index=False)
    print(f"WROTE {out_trades}")

    summary = pd.DataFrame(
        {
            "n_trades": [len(metrics)],
            "mean_realized_spread": [metrics["realized_spread"].mean()],
            "mean_adverse_selection": [metrics["adverse_selection"].mean()],
            "total_realized_spread": [metrics["realized_spread"].sum()],
            "total_adverse_selection": [metrics["adverse_selection"].sum()],
        }
    )
    print(summary.to_string(index=False))

    print(df.tail(5))

    # --- Parameter sweep ---
    spreads = [float(x) for x in args.sweep_spreads.split(",") if x.strip()]
    vols = [float(x) for x in args.sweep_vols.split(",") if x.strip()]

    sweep_rows = []
    for sp in spreads:
        for vol in vols:
            d = simulate_mm(
                n_steps=args.n_steps,
                seed=args.seed,
                mid0=args.mid0,
                mid_vol=vol,
                spread=sp,
                lambda_buy=args.lambda_buy,
                lambda_sell=args.lambda_sell,
                inventory_limit=args.inventory_limit,
                inv_skew=0.0,               # sweep baseline: no skew
                hit_decay=args.hit_decay,
                regime_mu=args.regime_mu,
                regime_persist=args.regime_persist,
                informed_strength=args.informed_strength,
                risk_off=args.risk_off,
                risk_off_neutral=args.risk_off_neutral,
                risk_off_inv=args.risk_off_inv,
                unwind_strength=args.unwind_strength,


            )

            tr = d[d["trade_side"] != 0].copy()
            tr["adverse_selection"] = -tr["trade_side"] * (tr["mid_post"] - tr["mid_pre"])
            tr["realized_spread"] = tr["trade_side"] * (tr["mid_post"] - tr["trade_px"])

            sweep_rows.append(
                {
                    "spread": sp,
                    "mid_vol": vol,
                    "n_trades": int(len(tr)),
                    "mean_realized_spread": float(tr["realized_spread"].mean()) if len(tr) else float("nan"),
                    "mean_adverse_selection": float(tr["adverse_selection"].mean()) if len(tr) else float("nan"),
                    "mean_net_capture": float((tr["realized_spread"] - tr["adverse_selection"]).mean())
                    if len(tr)
                    else float("nan"),
                    "inv_std": float(d["inv"].std(ddof=1)),
                    "inv_max_abs": int(d["inv"].abs().max()),
                    "pnl_mtm_end": float(d["pnl_mtm"].iloc[-1]),
                }
            )

    sweep = pd.DataFrame(sweep_rows).sort_values(["spread", "mid_vol"])
    out_sweep = reports_dir / "sim_mm_sweep.csv"
    sweep.to_csv(out_sweep, index=False)
    print(f"WROTE {out_sweep}")
    print(sweep)

    # Plot: mean_net_capture vs mid_vol, one line per spread
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for sp in sorted(sweep["spread"].unique()):
        sub = sweep[sweep["spread"] == sp].sort_values("mid_vol")
        plt.plot(sub["mid_vol"], sub["mean_net_capture"], marker="o", label=f"spread={sp}")
    plt.title("Simulated MM: Mean Net Capture vs Volatility")
    plt.xlabel("mid_vol (per-step)")
    plt.ylabel("mean_net_capture (realized - adverse)")
    plt.legend()
    plt.tight_layout()

    out_png = reports_dir / "net_capture_vs_vol.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"WROTE {out_png}")

    # --- Inventory skew comparison ---
    skews = [float(x) for x in args.skew_values.split(",") if x.strip()]
    skew_rows = []
    for k in skews:
        d = simulate_mm(
            n_steps=args.n_steps,
            seed=args.seed,
            mid0=args.mid0,
            mid_vol=args.mid_vol,
            spread=args.spread,
            lambda_buy=args.lambda_buy,
            lambda_sell=args.lambda_sell,
            inventory_limit=args.inventory_limit,
            inv_skew=k,
            hit_decay=args.hit_decay,
            regime_mu=args.regime_mu,
            regime_persist=args.regime_persist,
            informed_strength=args.informed_strength,
            risk_off=args.risk_off,
            risk_off_neutral=args.risk_off_neutral,
            risk_off_inv=args.risk_off_inv,
            unwind_strength=args.unwind_strength,


        )

        tr = d[d["trade_side"] != 0].copy()
        tr["adverse_selection"] = -tr["trade_side"] * (tr["mid_post"] - tr["mid_pre"])
        tr["realized_spread"] = tr["trade_side"] * (tr["mid_post"] - tr["trade_px"])

        skew_rows.append(
            {
                "inv_skew": k,
                "n_trades": int(len(tr)),
                "mean_realized_spread": float(tr["realized_spread"].mean()),
                "mean_adverse_selection": float(tr["adverse_selection"].mean()),
                "mean_net_capture": float((tr["realized_spread"] - tr["adverse_selection"]).mean()),
                "inv_std": float(d["inv"].std(ddof=1)),
                "inv_max_abs": int(d["inv"].abs().max()),
                "pnl_mtm_end": float(d["pnl_mtm"].iloc[-1]),
            }
        )

    skew_df = pd.DataFrame(skew_rows)
    out_skew = reports_dir / "sim_mm_inventory_skew.csv"
    skew_df.to_csv(out_skew, index=False)
    print(f"WROTE {out_skew}")
    print(skew_df)




if __name__ == "__main__":
    main()
