# -*- coding: utf-8 -*-
"""Create a monthly DCA buy plan for the live alpha portfolio.

If alpha_lab/current_holdings.csv exists, it should contain:
ticker,shares

Without holdings, the script assumes a fresh monthly contribution and buys the
most underfunded target assets by minimum practical lots.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "alpha_lab"

MONTHLY_KRW = 200_000
MAX_BUYS = 3
MIN_BUY_KRW = 20_000
HOLDINGS_FILE = OUT / "current_holdings.csv"
PORTFOLIO_FILE = os.environ.get("DCA_PORTFOLIO", "live_alpha_portfolio.csv")
OUTPUT_FILE = os.environ.get("DCA_OUTPUT", "monthly_dca_plan.csv")


def load_holdings(portfolio: pd.DataFrame) -> pd.DataFrame:
    if HOLDINGS_FILE.exists():
        h = pd.read_csv(HOLDINGS_FILE, encoding="utf-8-sig")
        return h[["ticker", "shares"]].rename(columns={"shares": "current_shares"})
    return portfolio[["ticker"]].assign(current_shares=0.0)


def main() -> None:
    p = pd.read_csv(OUT / PORTFOLIO_FILE, encoding="utf-8-sig")
    h = load_holdings(p)
    df = p.merge(h, on="ticker", how="left")
    df["current_shares"] = df["current_shares"].fillna(0.0)
    df["holding_value"] = df["current_shares"] * df["price_krw"]
    current_value = df["holding_value"].sum()
    target_total = current_value + MONTHLY_KRW
    df["target_value_after_dca"] = target_total * df["weight"]
    df["gap"] = df["target_value_after_dca"] - df["holding_value"]
    df["gap_score"] = df["gap"] / df["price_krw"].clip(lower=1)

    # Fresh-account fallback: if there are no holdings, buy largest target KRW weights.
    if current_value <= 0:
        candidates = df.sort_values("amount_10m_krw", ascending=False).copy()
        candidates["suggested_amount"] = MONTHLY_KRW
    else:
        candidates = df[df["gap"] > MIN_BUY_KRW].sort_values("gap", ascending=False).copy()
        candidates["suggested_amount"] = candidates["gap"].clip(upper=MONTHLY_KRW)

    buys = []
    remaining = MONTHLY_KRW
    for row in candidates.itertuples(index=False):
        if remaining < MIN_BUY_KRW or len(buys) >= MAX_BUYS:
            break
        raw_amt = min(float(row.suggested_amount), remaining)
        if str(row.ticker).endswith(".KS"):
            shares = int(raw_amt // row.price_krw)
            if shares <= 0 and row.price_krw <= remaining:
                shares = 1
            amt = shares * row.price_krw
        else:
            amt = raw_amt
            shares = amt / row.price_krw
        if amt < MIN_BUY_KRW or amt > remaining:
            continue
        buys.append({
            "ticker": row.ticker,
            "name": row.name,
            "asset_class": row.asset_class,
            "target_weight": row.weight,
            "buy_amount_krw": round(amt),
            "price_krw": row.price_krw,
            "estimated_shares": round(shares, 4),
            "note": "whole shares" if str(row.ticker).endswith(".KS") else "fractional ok",
        })
        remaining -= amt

    if buys and remaining >= MIN_BUY_KRW and not str(buys[0]["ticker"]).endswith(".KS"):
        buys[0]["buy_amount_krw"] += round(remaining)
        buys[0]["estimated_shares"] = round(buys[0]["buy_amount_krw"] / buys[0]["price_krw"], 4)
        remaining = 0

    out = pd.DataFrame(buys)
    out.to_csv(OUT / OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Monthly contribution: {MONTHLY_KRW:,} KRW")
    print(f"Current value from holdings file: {current_value:,.0f} KRW")
    print("\n=== Buy plan ===")
    print(out.to_string(index=False, formatters={"target_weight": "{:.1%}".format, "buy_amount_krw": "{:,.0f}".format, "price_krw": "{:,.0f}".format}))
    print(f"\nUnallocated cash: {remaining:,.0f} KRW")


if __name__ == "__main__":
    main()
