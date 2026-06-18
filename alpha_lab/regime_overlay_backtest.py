# -*- coding: utf-8 -*-
"""Macro regime overlay for the ETF-only core portfolio.

The overlay uses only information known at each monthly rebalance:
VIX, SPY trend, KOSPI trend, USD/KRW momentum, and short-rate pressure.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml_alpha_backtest import RF_TICKER, START_DATE, load_data, metrics
from reject_model_backtest import (
    ETF_ONLY_CLASSES,
    all_inverse_vol_weight,
    enforce_group_limits,
    transaction_cost,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "alpha_lab"

EQUITY_CLASSES = {"kr_equity_etf", "us_equity_etf", "us_equity_krx"}
DEF_CLASSES = {"cash_krw", "cash_usd", "bond"}
GOLD_CLASSES = {"gold"}


def load_raw_monthly() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "prices_raw.csv", index_col=0, parse_dates=True).ffill()
    monthly = raw.resample("ME").last()
    if raw.index.max() < monthly.index[-1]:
        monthly = monthly.iloc[:-1]
    return monthly


def regime_table(raw_m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw_m.index)
    vix = raw_m["^VIX"]
    vix_band = vix.rolling(24, min_periods=12).median() + 0.5 * vix.rolling(24, min_periods=12).std()
    out["vix_stress"] = (vix > 25) | (vix > vix_band)
    out["spy_downtrend"] = raw_m["SPY"] / raw_m["SPY"].shift(6) - 1 < 0
    out["kospi_downtrend"] = raw_m["^KS11"] / raw_m["^KS11"].shift(6) - 1 < 0
    out["fx_stress"] = raw_m["KRW=X"].pct_change(3) > 0.05
    out["rate_stress"] = raw_m["^IRX"] - raw_m["^IRX"].shift(3) > 0.50
    signal_cols = ["vix_stress", "spy_downtrend", "kospi_downtrend", "fx_stress", "rate_stress"]
    out["risk_score"] = out[signal_cols].sum(axis=1)
    out["regime"] = np.select(
        [out["risk_score"] >= 2, out["risk_score"] == 1],
        ["risk_off", "neutral"],
        default="risk_on",
    )
    return out


def apply_overlay(w: pd.Series, regime: str, meta: pd.DataFrame) -> pd.Series:
    if regime == "risk_off":
        mult = {"equity": 0.65, "gold": 1.80, "def": 1.10}
    elif regime == "risk_on":
        mult = {"equity": 1.12, "gold": 0.85, "def": 0.90}
    else:
        mult = {"equity": 1.0, "gold": 1.0, "def": 1.0}

    adj = w.copy()
    for tk in adj.index:
        ac = meta.loc[tk, "asset_class"]
        if ac in EQUITY_CLASSES:
            adj.loc[tk] *= mult["equity"]
        elif ac in GOLD_CLASSES:
            adj.loc[tk] *= mult["gold"]
        elif ac in DEF_CLASSES:
            adj.loc[tk] *= mult["def"]
    adj = adj / adj.sum()
    return enforce_group_limits(adj, meta)


def main() -> None:
    prices, monthly, meta = load_data()
    raw_m = load_raw_monthly()
    regimes = regime_table(raw_m)
    mret = monthly.pct_change(fill_method=None)
    rf = mret[RF_TICKER]

    returns = {"ETF_Only_Core": [], "Regime_Overlay": [], "BM_KODEX200": []}
    prev = {k: None for k in ["ETF_Only_Core", "Regime_Overlay"]}
    latest = {}
    regime_history = []

    dates = [d for d in monthly.index if d >= pd.Timestamp(START_DATE)]
    for dt, next_dt in zip(dates[:-1], dates[1:]):
        if dt not in regimes.index:
            continue
        vol_3m = prices.pct_change(fill_method=None).rolling(63).std().resample("ME").last().loc[dt] * math.sqrt(252)
        base = all_inverse_vol_weight(vol_3m, meta, ETF_ONLY_CLASSES)
        regime = regimes.loc[dt, "regime"]
        overlay = apply_overlay(base, regime, meta)

        weights = {"ETF_Only_Core": base, "Regime_Overlay": overlay}
        for name, w in weights.items():
            tc = transaction_cost(w, prev[name], mret.loc[dt], meta)
            gross = float((w * mret.loc[next_dt].reindex(w.index).fillna(0)).sum())
            returns[name].append((next_dt, gross - tc))
            prev[name] = w
            latest[name] = w

        returns["BM_KODEX200"].append((next_dt, float(mret.loc[next_dt, "069500.KS"])))
        regime_history.append({
            "date": dt,
            "regime": regime,
            "risk_score": regimes.loc[dt, "risk_score"],
            "vix_stress": regimes.loc[dt, "vix_stress"],
            "spy_downtrend": regimes.loc[dt, "spy_downtrend"],
            "kospi_downtrend": regimes.loc[dt, "kospi_downtrend"],
            "fx_stress": regimes.loc[dt, "fx_stress"],
            "rate_stress": regimes.loc[dt, "rate_stress"],
        })

    ret = pd.DataFrame({k: pd.Series(dict(v)) for k, v in returns.items()}).dropna(how="all")
    summary = pd.DataFrame({c: metrics(ret[c], rf) for c in ret.columns}).T
    history = pd.DataFrame(regime_history)

    latest_rows = []
    for model, w in latest.items():
        tmp = w.sort_values(ascending=False).to_frame("weight")
        tmp["model"] = model
        tmp["name"] = [meta.loc[t, "name"] for t in tmp.index]
        tmp["asset_class"] = [meta.loc[t, "asset_class"] for t in tmp.index]
        latest_rows.append(tmp.reset_index(names="ticker"))
    latest_df = pd.concat(latest_rows, ignore_index=True)

    ret.to_csv(OUT / "regime_overlay_returns.csv", encoding="utf-8-sig")
    summary.to_csv(OUT / "regime_overlay_summary.csv", encoding="utf-8-sig")
    history.to_csv(OUT / "regime_overlay_history.csv", index=False, encoding="utf-8-sig")
    latest_df.to_csv(OUT / "regime_overlay_latest_weights.csv", index=False, encoding="utf-8-sig")

    print(f"Regime overlay walk-forward: {ret.index.min().date()} ~ {ret.index.max().date()} ({len(ret)} months)")
    print("\n=== Summary ===")
    print(summary.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\n=== Regime counts ===")
    print(history["regime"].value_counts().to_string())
    print("\n=== Latest regime ===")
    print(history.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
