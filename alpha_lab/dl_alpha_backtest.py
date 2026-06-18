# -*- coding: utf-8 -*-
"""Tiny numpy MLP alpha experiment.

This is deliberately small and dependency-free. It is not meant to be a final
deep-learning trading engine; it is a falsifiable DL baseline for the current
data size.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml_alpha_backtest import (
    RF_TICKER,
    START_DATE,
    build_feature_panel,
    load_data,
    make_weights,
    metrics,
    transaction_cost,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "alpha_lab"

HIDDEN = 12
EPOCHS = 220
LR = 0.015
L2 = 0.002
SEED = 42


def standardize(x_train: pd.DataFrame, x_pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    mu = x_train.mean()
    sd = x_train.std().replace(0, 1.0)
    xs = ((x_train - mu) / sd).clip(-5, 5).values.astype(float)
    xp = ((x_pred - mu) / sd).clip(-5, 5).values.astype(float)
    return xs, xp, mu, sd


def fit_mlp_predict(x_train: pd.DataFrame, y_train: pd.Series, x_pred: pd.DataFrame, seed_offset: int = 0) -> pd.Series:
    X, XP, _, _ = standardize(x_train, x_pred)
    y = y_train.values.astype(float)
    y_mu, y_sd = float(y.mean()), float(y.std() if y.std() > 0 else 1.0)
    y = ((y - y_mu) / y_sd).reshape(-1, 1)

    rng = np.random.default_rng(SEED + seed_offset)
    n, p = X.shape
    W1 = rng.normal(0, 1 / math.sqrt(p), size=(p, HIDDEN))
    b1 = np.zeros((1, HIDDEN))
    W2 = rng.normal(0, 1 / math.sqrt(HIDDEN), size=(HIDDEN, 1))
    b2 = np.zeros((1, 1))

    # Full-batch training is enough for the small monthly cross-section panel.
    for _ in range(EPOCHS):
        Z1 = X @ W1 + b1
        A1 = np.tanh(Z1)
        pred = A1 @ W2 + b2
        err = pred - y

        d_pred = 2.0 * err / n
        dW2 = A1.T @ d_pred + L2 * W2
        db2 = d_pred.sum(axis=0, keepdims=True)
        dA1 = d_pred @ W2.T
        dZ1 = dA1 * (1 - A1**2)
        dW1 = X.T @ dZ1 + L2 * W1
        db1 = dZ1.sum(axis=0, keepdims=True)

        W2 -= LR * dW2
        b2 -= LR * db2
        W1 -= LR * dW1
        b1 -= LR * db1

    out = np.tanh(XP @ W1 + b1) @ W2 + b2
    out = out.ravel() * y_sd + y_mu
    return pd.Series(out, index=x_pred.index)


def main() -> None:
    prices, monthly, meta = load_data()
    x, y = build_feature_panel(prices, monthly, meta)
    mret = monthly.pct_change(fill_method=None)
    rf = mret[RF_TICKER]

    model_returns = []
    ridge_returns = pd.read_csv(OUT / "ml_alpha_returns.csv", index_col=0, parse_dates=True, encoding="utf-8-sig")
    weights_hist = {}
    pred_ic = {}
    prev_w = None

    dates = [d for d in monthly.index if d >= pd.Timestamp(START_DATE)]
    for i, (dt, next_dt) in enumerate(zip(dates[:-1], dates[1:])):
        train_end = monthly.index[monthly.index.get_loc(dt) - 1]
        train_idx = x.index.get_level_values("date") <= train_end
        pred_idx = x.index.get_level_values("date") == dt
        if train_idx.sum() < 360 or pred_idx.sum() == 0:
            continue

        pred = fit_mlp_predict(x.loc[train_idx], y.loc[train_idx], x.loc[pred_idx], seed_offset=i)
        realized = y.loc[pred_idx]
        if pred.std() > 0 and realized.std() > 0:
            pred_ic[dt] = pred.rank().corr(realized.rank())

        vol_3m = prices.pct_change(fill_method=None).rolling(63).std().resample("ME").last().loc[dt] * math.sqrt(252)
        w = make_weights(pred, vol_3m, meta)
        tc = transaction_cost(w, prev_w, mret.loc[dt], meta)
        gross = float((w * mret.loc[next_dt].reindex(w.index).fillna(0)).sum())
        model_returns.append((next_dt, gross - tc))
        weights_hist[dt] = w
        prev_w = w

    ret = pd.DataFrame({"DL_TinyMLP": pd.Series(dict(model_returns))})
    for col in ["ML_RidgeAlpha", "EqualWeight_All", "BM_KODEX200"]:
        if col in ridge_returns.columns:
            ret[col] = ridge_returns[col]
    ret = ret.dropna(how="all")

    summary = pd.DataFrame({c: metrics(ret[c], rf) for c in ret.columns}).T
    summary["AvgIC"] = np.nan
    summary["HitIC"] = np.nan
    summary.loc["DL_TinyMLP", "AvgIC"] = pd.Series(pred_ic).mean()
    summary.loc["DL_TinyMLP", "HitIC"] = (pd.Series(pred_ic) > 0).mean()

    latest_dt = max(weights_hist)
    latest_w = weights_hist[latest_dt].sort_values(ascending=False).to_frame("weight")
    latest_w["name"] = [meta.loc[t, "name"] for t in latest_w.index]
    latest_w["asset_class"] = [meta.loc[t, "asset_class"] for t in latest_w.index]

    ret.to_csv(OUT / "dl_alpha_returns.csv", encoding="utf-8-sig")
    summary.to_csv(OUT / "dl_alpha_summary.csv", encoding="utf-8-sig")
    latest_w.to_csv(OUT / "dl_alpha_latest_weights.csv", encoding="utf-8-sig")

    print(f"DL TinyMLP walk-forward: {ret.index.min().date()} ~ {ret.index.max().date()} ({len(ret)} months)")
    print("\n=== Summary ===")
    print(summary.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\n=== Latest weights @ {latest_dt.date()} ===")
    print(latest_w.to_string(formatters={"weight": "{:.1%}".format}))


if __name__ == "__main__":
    main()
