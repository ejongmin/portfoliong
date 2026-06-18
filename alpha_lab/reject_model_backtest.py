# -*- coding: utf-8 -*-
"""Reject-model portfolio backtest.

The model predicts cross-sectional next-month relative returns, then uses the
signal only to reject the weakest assets. The remaining assets are allocated
with simple robust rules. This is often more realistic than trusting a small
sample ML model to size every position directly.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml_alpha_backtest import (
    CAPS,
    COST_ETF,
    COST_STOCK,
    DEFENSIVE,
    RF_TICKER,
    START_DATE,
    build_feature_panel,
    fit_ridge_predict,
    load_data,
    metrics,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "alpha_lab"

REJECT_BOTTOM = 0.35
TARGET_COUNT = 12
IC_LOOKBACK = 12
IC_GATE = 0.0
ETF_ONLY_CLASSES = {"kr_equity_etf", "us_equity_etf", "us_equity_krx", "gold", "cash_usd", "cash_krw", "bond"}


def cap_and_normalize(raw: pd.Series, meta: pd.DataFrame) -> pd.Series:
    w = raw.clip(lower=0).copy()
    if w.sum() <= 0:
        return w
    w /= w.sum()
    caps = pd.Series({tk: CAPS[meta.loc[tk, "asset_class"]] for tk in w.index})
    for _ in range(100):
        over = w > caps
        if not over.any():
            break
        excess = (w[over] - caps[over]).sum()
        w[over] = caps[over]
        room = (~over) & (w < caps)
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += excess * w[room] / w[room].sum()
    return w / w.sum()


def enforce_group_limits(w: pd.Series, meta: pd.DataFrame) -> pd.Series:
    groups = [
        (set(meta[meta["asset_class"].isin(DEFENSIVE)].index), 0.35),
        (set(meta[meta["asset_class"].isin(DEFENSIVE | {"gold"})].index), 0.50),
    ]
    w = w.copy()
    for _ in range(50):
        changed = False
        for members, limit in groups:
            inside = w.index.isin(members)
            if w[inside].sum() > limit:
                changed = True
                excess = w[inside].sum() - limit
                w[inside] *= limit / w[inside].sum()
                outside = ~inside
                if w[outside].sum() > 0:
                    w[outside] += excess * w[outside] / w[outside].sum()
        w = cap_and_normalize(w, meta)
        if not changed:
            break
    return w / w.sum()


def transaction_cost(new_w: pd.Series, prev_w: pd.Series | None, returns_at_rebalance: pd.Series, meta: pd.DataFrame) -> float:
    if prev_w is None:
        turnover = new_w.abs()
    else:
        drift = prev_w * (1 + returns_at_rebalance.reindex(prev_w.index).fillna(0))
        drift = drift / drift.sum()
        union = new_w.index.union(drift.index)
        turnover = new_w.reindex(union).fillna(0).sub(drift.reindex(union).fillna(0)).abs()

    costs = pd.Series({
        tk: COST_STOCK if meta.loc[tk, "asset_class"] in {"kr_stock", "us_stock"} else COST_ETF
        for tk in turnover.index
    })
    return float((turnover * costs).sum())


def select_survivors(pred: pd.Series) -> pd.Index:
    scores = pred.droplevel("date").sort_values(ascending=False)
    keep_n = max(TARGET_COUNT, int(math.ceil(len(scores) * (1 - REJECT_BOTTOM))))
    return scores.head(keep_n).index


def equal_weight(survivors: pd.Index, meta: pd.DataFrame) -> pd.Series:
    raw = pd.Series(1.0, index=survivors)
    return enforce_group_limits(cap_and_normalize(raw, meta), meta)


def inverse_vol_weight(survivors: pd.Index, vol_3m: pd.Series, meta: pd.DataFrame) -> pd.Series:
    vols = vol_3m.reindex(survivors).replace(0, np.nan)
    vols = vols.fillna(vols.median()).clip(lower=0.05)
    raw = 1.0 / vols
    return enforce_group_limits(cap_and_normalize(raw, meta), meta)


def all_inverse_vol_weight(vol_3m: pd.Series, meta: pd.DataFrame, asset_classes: set[str] | None = None) -> pd.Series:
    investable = []
    for tk in vol_3m.index:
        if tk not in meta.index or meta.loc[tk, "asset_class"] not in CAPS:
            continue
        if asset_classes is not None and meta.loc[tk, "asset_class"] not in asset_classes:
            continue
        investable.append(tk)
    return inverse_vol_weight(pd.Index(investable), vol_3m, meta)


def top_score_inverse_vol(pred: pd.Series, vol_3m: pd.Series, meta: pd.DataFrame) -> pd.Series:
    scores = pred.droplevel("date").sort_values(ascending=False).head(TARGET_COUNT)
    shifted = scores - scores.min() + 1e-6
    vols = vol_3m.reindex(scores.index).replace(0, np.nan)
    vols = vols.fillna(vols.median()).clip(lower=0.05)
    raw = shifted / vols
    return enforce_group_limits(cap_and_normalize(raw, meta), meta)


def main() -> None:
    prices, monthly, meta = load_data()
    x, y = build_feature_panel(prices, monthly, meta)
    mret = monthly.pct_change(fill_method=None)
    rf = mret[RF_TICKER]

    returns = {
        "Core_InvVol_All": [],
        "ETF_Only_Core": [],
        "Reject_EW": [],
        "Reject_InvVol": [],
        "IC_Gated_RejectInvVol": [],
        "TopScore_InvVol": [],
        "EqualWeight_All": [],
        "BM_KODEX200": [],
    }
    prev = {k: None for k in ["Core_InvVol_All", "ETF_Only_Core", "Reject_EW", "Reject_InvVol", "IC_Gated_RejectInvVol", "TopScore_InvVol"]}
    latest_weights = {}
    reject_ic = {}
    reject_hit = {}

    dates = [d for d in monthly.index if d >= pd.Timestamp(START_DATE)]
    for dt, next_dt in zip(dates[:-1], dates[1:]):
        train_end = monthly.index[monthly.index.get_loc(dt) - 1]
        train_idx = x.index.get_level_values("date") <= train_end
        pred_idx = x.index.get_level_values("date") == dt
        if train_idx.sum() < 360 or pred_idx.sum() == 0:
            continue

        pred = fit_ridge_predict(x.loc[train_idx], y.loc[train_idx], x.loc[pred_idx])
        realized = y.loc[pred_idx]
        recent_ic = pd.Series(reject_ic).tail(IC_LOOKBACK).mean()
        ic_gate_on = bool(np.isfinite(recent_ic) and recent_ic > IC_GATE)
        if pred.std() > 0 and realized.std() > 0:
            reject_ic[dt] = pred.rank().corr(realized.rank())

        scores = pred.droplevel("date")
        bad_cut = scores.quantile(REJECT_BOTTOM)
        rejected = scores[scores <= bad_cut].index
        accepted = scores[scores > bad_cut].index
        if len(rejected) and len(accepted):
            rejected_ret = mret.loc[next_dt].reindex(rejected).mean()
            accepted_ret = mret.loc[next_dt].reindex(accepted).mean()
            reject_hit[dt] = float(rejected_ret < accepted_ret)

        vol_3m = prices.pct_change(fill_method=None).rolling(63).std().resample("ME").last().loc[dt] * math.sqrt(252)
        survivors = select_survivors(pred)
        weights = {
            "Core_InvVol_All": all_inverse_vol_weight(vol_3m, meta),
            "ETF_Only_Core": all_inverse_vol_weight(vol_3m, meta, ETF_ONLY_CLASSES),
            "Reject_EW": equal_weight(survivors, meta),
            "Reject_InvVol": inverse_vol_weight(survivors, vol_3m, meta),
            "TopScore_InvVol": top_score_inverse_vol(pred, vol_3m, meta),
        }
        weights["IC_Gated_RejectInvVol"] = weights["Reject_InvVol"] if ic_gate_on else weights["Core_InvVol_All"]

        for name, w in weights.items():
            tc = transaction_cost(w, prev[name], mret.loc[dt], meta)
            gross = float((w * mret.loc[next_dt].reindex(w.index).fillna(0)).sum())
            returns[name].append((next_dt, gross - tc))
            prev[name] = w
            latest_weights[name] = w

        ew_assets = mret.loc[next_dt].dropna().index
        returns["EqualWeight_All"].append((next_dt, float(mret.loc[next_dt, ew_assets].mean())))
        returns["BM_KODEX200"].append((next_dt, float(mret.loc[next_dt, "069500.KS"])))

    ret = pd.DataFrame({k: pd.Series(dict(v)) for k, v in returns.items()}).dropna(how="all")
    summary = pd.DataFrame({c: metrics(ret[c], rf) for c in ret.columns}).T
    for name in ["Reject_EW", "Reject_InvVol", "IC_Gated_RejectInvVol", "TopScore_InvVol"]:
        summary.loc[name, "AvgIC"] = pd.Series(reject_ic).mean()
        summary.loc[name, "RejectHit"] = pd.Series(reject_hit).mean()

    latest = []
    for model, w in latest_weights.items():
        tmp = w.sort_values(ascending=False).to_frame("weight")
        tmp["model"] = model
        tmp["name"] = [meta.loc[t, "name"] for t in tmp.index]
        tmp["asset_class"] = [meta.loc[t, "asset_class"] for t in tmp.index]
        latest.append(tmp.reset_index(names="ticker"))
    latest_df = pd.concat(latest, ignore_index=True)

    ret.to_csv(OUT / "reject_model_returns.csv", encoding="utf-8-sig")
    summary.to_csv(OUT / "reject_model_summary.csv", encoding="utf-8-sig")
    latest_df.to_csv(OUT / "reject_model_latest_weights.csv", index=False, encoding="utf-8-sig")

    print(f"Reject model walk-forward: {ret.index.min().date()} ~ {ret.index.max().date()} ({len(ret)} months)")
    print("\n=== Summary ===")
    print(summary.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\n=== Latest weights ===")
    print(latest_df[latest_df["model"] == "Reject_InvVol"].to_string(index=False, formatters={"weight": "{:.1%}".format}))


if __name__ == "__main__":
    main()
