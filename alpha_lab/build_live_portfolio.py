# -*- coding: utf-8 -*-
"""Build a practical live portfolio file from alpha_lab backtest outputs."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "alpha_lab"

PORTFOLIO_KRW = 10_000_000
MODEL = os.environ.get("LIVE_MODEL", "ETF_Only_Core")
SOURCE = os.environ.get("LIVE_SOURCE", "reject_model_latest_weights.csv")
OUTPUT = os.environ.get("LIVE_OUTPUT", "live_alpha_portfolio.csv")


def main() -> None:
    weights = pd.read_csv(OUT / SOURCE, encoding="utf-8-sig")
    raw = pd.read_csv(DATA / "prices_raw.csv", index_col=0, parse_dates=True)
    meta = pd.read_csv(DATA / "meta.csv", encoding="utf-8-sig").set_index("ticker")
    fx = raw["KRW=X"].ffill().iloc[-1]

    w = weights[weights["model"] == MODEL].copy()
    w = w.sort_values("weight", ascending=False)
    w["weight"] = w["weight"] / w["weight"].sum()

    rows = []
    for row in w.itertuples(index=False):
        tk = row.ticker
        px = raw[tk].ffill().iloc[-1]
        px_krw = px * fx if meta.loc[tk, "currency"] == "USD" else px
        amount = PORTFOLIO_KRW * row.weight
        rows.append({
            "ticker": tk,
            "name": row.name,
            "asset_class": row.asset_class,
            "weight": row.weight,
            "amount_10m_krw": round(amount),
            "price_krw": round(px_krw),
            "shares": round(amount / px_krw, 4),
            "note": "fractional needed" if amount < px_krw else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / OUTPUT, index=False, encoding="utf-8-sig")

    cls = out.groupby("asset_class")["weight"].sum().sort_values(ascending=False)
    print(f"Live alpha portfolio model: {MODEL}")
    print(f"FX: {fx:,.2f} KRW/USD | capital: {PORTFOLIO_KRW:,} KRW")
    print("\n=== Portfolio ===")
    print(out.to_string(index=False, formatters={"weight": "{:.1%}".format, "amount_10m_krw": "{:,.0f}".format, "price_krw": "{:,.0f}".format}))
    print("\n=== Asset class ===")
    print(cls.to_string(float_format=lambda v: f"{v:.1%}"))


if __name__ == "__main__":
    main()
