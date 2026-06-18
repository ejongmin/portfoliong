# -*- coding: utf-8 -*-
"""Monthly ML alpha backtest with no external ML dependencies.

This module is intentionally separate from the existing Claude-generated
pipeline. It reads the shared data files, but writes only under alpha_lab/.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "alpha_lab"
DATA = ROOT / "data"

EST_MONTHS = 36
START_DATE = "2019-01-31"
TOP_N = 8
ALPHAS = (0.1, 1.0, 10.0, 50.0)

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
COST_STOCK = 0.0025
COST_ETF = 0.0010
RF_TICKER = "153130.KS"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(DATA / "prices_krw.csv", index_col=0, parse_dates=True)
    meta = pd.read_csv(DATA / "meta.csv", encoding="utf-8-sig").set_index("ticker")
    investable = [c for c in prices.columns if c in meta.index and meta.loc[c, "asset_class"] in CAPS]
    prices = prices[investable].sort_index().ffill()
    monthly = prices.resample("ME").last()
    if prices.index.max() < monthly.index[-1]:
        monthly = monthly.iloc[:-1]
    return prices, monthly, meta


def zscore_row(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def build_feature_panel(prices: pd.DataFrame, monthly: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    mret = monthly.pct_change(fill_method=None)
    daily_ret = prices.pct_change(fill_method=None)

    feats = {}
    feats["mom_1m"] = monthly.pct_change(1)
    feats["mom_3m"] = monthly.pct_change(3)
    feats["mom_6m"] = monthly.pct_change(6)
    feats["mom_12_1m"] = monthly.shift(1) / monthly.shift(12) - 1
    feats["rev_1m"] = -feats["mom_1m"]
    feats["vol_1m"] = daily_ret.rolling(21).std().resample("ME").last() * math.sqrt(252)
    feats["vol_3m"] = daily_ret.rolling(63).std().resample("ME").last() * math.sqrt(252)
    feats["dd_12m"] = (prices / prices.rolling(252).max() - 1).resample("ME").last()

    rows = []
    labels = []
    next_ret = mret.shift(-1)
    rel_next = next_ret.sub(next_ret.mean(axis=1), axis=0)

    asset_classes = sorted(set(meta.loc[monthly.columns, "asset_class"]))
    for dt in monthly.index:
        if dt < pd.Timestamp("2016-01-31"):
            continue
        for tk in monthly.columns:
            vals = {name: frame.loc[dt, tk] if dt in frame.index else np.nan for name, frame in feats.items()}
            if not np.isfinite(list(vals.values())).all():
                continue
            if dt not in rel_next.index or not np.isfinite(rel_next.loc[dt, tk]):
                continue
            ac = meta.loc[tk, "asset_class"]
            for cls in asset_classes:
                vals[f"ac_{cls}"] = 1.0 if ac == cls else 0.0
            vals["ticker"] = tk
            vals["date"] = dt
            rows.append(vals)
            labels.append(rel_next.loc[dt, tk])

    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    y = pd.Series(labels, index=panel.index, name="target_rel_next")

    numeric = panel.columns
    base = panel[numeric].copy()
    by_date_cols = [c for c in base.columns if not c.startswith("ac_")]
    base.loc[:, by_date_cols] = base.groupby(level="date")[by_date_cols].transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() != 0 else np.nan)
    )
    base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return base, y


def fit_ridge_predict(x_train: pd.DataFrame, y_train: pd.Series, x_pred: pd.DataFrame) -> pd.Series:
    x_mu = x_train.mean()
    x_sd = x_train.std().replace(0, 1.0)
    xs = (x_train - x_mu) / x_sd
    xp = (x_pred - x_mu) / x_sd

    y_mu = y_train.mean()
    yc = y_train - y_mu
    X = np.c_[np.ones(len(xs)), xs.values]
    XP = np.c_[np.ones(len(xp)), xp.values]

    preds = []
    for alpha in ALPHAS:
        penalty = np.eye(X.shape[1]) * alpha
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(X.T @ X + penalty) @ X.T @ yc.values
        preds.append(XP @ beta + y_mu)
    return pd.Series(np.mean(preds, axis=0), index=x_pred.index)


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
        if not room.any():
            break
        w[room] += excess * w[room] / w[room].sum()
    return w / w.sum()


def enforce_group_limits(w: pd.Series, meta: pd.DataFrame) -> pd.Series:
    w = w.copy()
    groups = [
        (set(meta[meta["asset_class"].isin(DEFENSIVE)].index), 0.35),
        (set(meta[meta["asset_class"].isin(DEFENSIVE | {"gold"})].index), 0.50),
    ]
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


def make_weights(pred: pd.Series, vol_3m: pd.Series, meta: pd.DataFrame) -> pd.Series:
    scores = pred.droplevel("date").sort_values(ascending=False)
    selected = scores.head(TOP_N)
    selected = selected[selected > 0]
    if len(selected) < 4:
        selected = scores.head(max(4, min(TOP_N, len(scores))))

    vols = vol_3m.reindex(selected.index).replace(0, np.nan).fillna(vol_3m.median())
    raw = selected.clip(lower=0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=selected.index)
    raw = raw / vols.clip(lower=0.05)
    return enforce_group_limits(cap_and_normalize(raw, meta), meta)


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


def metrics(r: pd.Series, rf: pd.Series) -> dict[str, float]:
    r = r.dropna()
    cum = (1 + r).cumprod()
    years = len(r) / 12
    cagr = cum.iloc[-1] ** (1 / years) - 1
    vol = r.std() * math.sqrt(12)
    ex = r - rf.reindex(r.index).fillna(0)
    sharpe = ex.mean() / r.std() * math.sqrt(12) if r.std() > 0 else np.nan
    mdd = (cum / cum.cummax() - 1).min()
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }


def main() -> None:
    prices, monthly, meta = load_data()
    x, y = build_feature_panel(prices, monthly, meta)
    mret = monthly.pct_change()
    rf = mret[RF_TICKER]

    model_returns = []
    ew_returns = []
    bm_returns = []
    weights_hist = {}
    pred_ic = {}
    prev_w = None

    dates = [d for d in monthly.index if d >= pd.Timestamp(START_DATE)]
    for dt, next_dt in zip(dates[:-1], dates[1:]):
        train_end = monthly.index[monthly.index.get_loc(dt) - 1]
        train_idx = x.index.get_level_values("date") <= train_end
        pred_idx = x.index.get_level_values("date") == dt
        if train_idx.sum() < EST_MONTHS * 10 or pred_idx.sum() == 0:
            continue

        pred = fit_ridge_predict(x.loc[train_idx], y.loc[train_idx], x.loc[pred_idx])
        realized = y.loc[pred_idx]
        if pred.std() > 0 and realized.std() > 0:
            pred_ic[dt] = pred.rank().corr(realized.rank())

        vol_3m = prices.pct_change().rolling(63).std().resample("ME").last().loc[dt] * math.sqrt(252)
        w = make_weights(pred, vol_3m, meta)
        tc = transaction_cost(w, prev_w, mret.loc[dt], meta)
        gross = float((w * mret.loc[next_dt].reindex(w.index).fillna(0)).sum())
        model_returns.append((next_dt, gross - tc))
        weights_hist[dt] = w
        prev_w = w

        ew_assets = mret.loc[next_dt].dropna().index
        ew_returns.append((next_dt, float(mret.loc[next_dt, ew_assets].mean())))
        bm_returns.append((next_dt, float(mret.loc[next_dt, "069500.KS"] if "069500.KS" in mret.columns else np.nan)))

    ret = pd.DataFrame({
        "ML_RidgeAlpha": pd.Series(dict(model_returns)),
        "EqualWeight_All": pd.Series(dict(ew_returns)),
        "BM_KODEX200": pd.Series(dict(bm_returns)),
    }).dropna(how="all")

    summary = pd.DataFrame({c: metrics(ret[c], rf) for c in ret.columns}).T
    summary["AvgIC"] = np.nan
    summary.loc["ML_RidgeAlpha", "AvgIC"] = pd.Series(pred_ic).mean()
    summary.loc["ML_RidgeAlpha", "HitIC"] = (pd.Series(pred_ic) > 0).mean()

    latest_dt = max(weights_hist)
    latest_w = weights_hist[latest_dt].sort_values(ascending=False).to_frame("weight")
    latest_w["name"] = [meta.loc[t, "name"] for t in latest_w.index]
    latest_w["asset_class"] = [meta.loc[t, "asset_class"] for t in latest_w.index]

    ret.to_csv(OUT / "ml_alpha_returns.csv", encoding="utf-8-sig")
    summary.to_csv(OUT / "ml_alpha_summary.csv", encoding="utf-8-sig")
    latest_w.to_csv(OUT / "ml_alpha_latest_weights.csv", encoding="utf-8-sig")

    print(f"ML Alpha walk-forward: {ret.index.min().date()} ~ {ret.index.max().date()} ({len(ret)} months)")
    print("\n=== Summary ===")
    print(summary.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\n=== Latest weights @ {latest_dt.date()} ===")
    print(latest_w.to_string(formatters={"weight": "{:.1%}".format}))


if __name__ == "__main__":
    main()
