# -*- coding: utf-8 -*-
"""Validation checks for the current alpha_lab live candidate."""
from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "alpha_lab"

MODEL = os.environ.get("VALIDATE_MODEL", "ETF_Only_Core")
PORTFOLIO_FILE = os.environ.get("VALIDATE_PORTFOLIO", "live_alpha_portfolio.csv")
RETURNS_FILE = os.environ.get("VALIDATE_RETURNS", "reject_model_returns.csv")
LATEST_WEIGHTS_FILE = os.environ.get("VALIDATE_LATEST", "reject_model_latest_weights.csv")
MDD_LIMIT = -0.20

CAPS = {
    "kr_stock": 0.15,
    "us_stock": 0.15,
    "kr_equity_etf": 0.30,
    "us_equity_etf": 0.30,
    "us_equity_krx": 0.30,
    "gold": 0.20,
    "cash_usd": 0.20,
    "cash_krw": 0.25,
    "bond": 0.20,
}
DEFENSIVE = {"cash_krw", "cash_usd", "bond"}


def metrics(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    cum = (1 + r).cumprod()
    years = len(r) / 12
    cagr = cum.iloc[-1] ** (1 / years) - 1
    vol = r.std() * math.sqrt(12)
    sharpe = r.mean() / r.std() * math.sqrt(12) if r.std() else float("nan")
    mdd = (cum / cum.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MDD": mdd}


def main() -> None:
    weights = pd.read_csv(OUT / PORTFOLIO_FILE, encoding="utf-8-sig")
    returns = pd.read_csv(OUT / RETURNS_FILE, index_col=0, parse_dates=True, encoding="utf-8-sig")
    latest = pd.read_csv(OUT / LATEST_WEIGHTS_FILE, encoding="utf-8-sig")

    checks = []
    wsum = weights["weight"].sum()
    checks.append(("weight_sum", abs(wsum - 1.0) < 1e-8, wsum))

    by_class = weights.groupby("asset_class")["weight"].sum()
    defensive = by_class.reindex(list(DEFENSIVE), fill_value=0).sum()
    gold = by_class.get("gold", 0.0)
    checks.append(("defensive_lte_35pct", defensive <= 0.35000001, defensive))
    checks.append(("defensive_plus_gold_lte_50pct", defensive + gold <= 0.50000001, defensive + gold))

    cap_ok = True
    cap_detail = []
    for row in weights.itertuples(index=False):
        cap = CAPS[row.asset_class]
        ok = row.weight <= cap + 1e-8
        cap_ok = cap_ok and ok
        if not ok:
            cap_detail.append(f"{row.ticker}:{row.weight:.3f}>{cap:.3f}")
    checks.append(("single_asset_caps", cap_ok, ";".join(cap_detail) if cap_detail else "ok"))

    full_m = metrics(returns[MODEL])
    checks.append(("mdd_above_limit", full_m["MDD"] >= MDD_LIMIT, full_m["MDD"]))

    starts = ["2019-02-28", "2020-01-31", "2021-01-31", "2022-01-31", "2023-01-31"]
    robust_rows = []
    for start in starts:
        seg = returns.loc[start:, MODEL]
        if len(seg) >= 24:
            robust_rows.append({"start": start, **metrics(seg)})
    robust = pd.DataFrame(robust_rows)
    checks.append(("all_start_mdd_above_limit", bool((robust["MDD"] >= MDD_LIMIT).all()), robust["MDD"].min()))

    latest_model = latest[latest["model"] == MODEL]
    checks.append(("latest_weights_match_live", len(latest_model) == len(weights), len(latest_model)))

    check_df = pd.DataFrame(checks, columns=["check", "pass", "value"])
    prefix = "regime" if MODEL == "Regime_Overlay" else "alpha"
    check_df.to_csv(OUT / f"{prefix}_validation_checks.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(OUT / f"{prefix}_start_sensitivity.csv", index=False, encoding="utf-8-sig")

    print("=== Validation checks ===")
    print(check_df.to_string(index=False))
    print("\n=== Start-date sensitivity ===")
    print(robust.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    if not check_df["pass"].all():
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
