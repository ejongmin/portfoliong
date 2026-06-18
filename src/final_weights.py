# -*- coding: utf-8 -*-
"""현재 시점(데이터 최신일) 기준 4개 모델의 타깃 가중치 산출 → data/final_weights.csv"""
import warnings
import numpy as np
import pandas as pd
import models

warnings.filterwarnings("ignore")

EST_MONTHS, TOP_N = 36, 5
ETF_SLEEVE = ["069500.KS", "229200.KS", "360750.KS", "133690.KS", "458730.KS",
              "411060.KS", "261240.KS", "153130.KS", "148070.KS", "TLT"]
PROXY = {"411060.KS": "GLD", "360750.KS": "SPY", "458730.KS": "SCHD", "261240.KS": "_FX"}
CAPS = {"kr_stock": 0.15, "us_stock": 0.15,
        "kr_equity_etf": 0.30, "us_equity_etf": 0.30, "us_equity_krx": 0.30,
        "gold": 0.20, "cash_usd": 0.20, "cash_krw": 0.25, "bond": 0.20}

prices = pd.read_csv("data/prices_krw.csv", index_col=0, parse_dates=True)
raw = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
meta = pd.read_csv("data/meta.csv", encoding="utf-8-sig").set_index("ticker")
prices["_FX"] = raw["KRW=X"]
daily = prices.copy()
for tk, proxy in PROXY.items():
    s, p = daily[tk], daily[proxy]
    first = s.first_valid_index()
    pre = p.loc[:first].pct_change().iloc[:-1]
    ext = pd.concat([pre, s.loc[first:].pct_change()]).dropna()
    daily[tk] = (1 + ext).cumprod() * 100
daily = daily.drop(columns=["_FX"])
monthly = daily.resample("ME").last()
if daily.index.max() < monthly.index[-1]:
    monthly = monthly.iloc[:-1]          # 공분산은 완전월만
mret = monthly.pct_change()
rf = mret["153130.KS"]

DEF_SET = set(meta[meta["asset_class"].isin(["cash_krw", "cash_usd", "bond"])].index)
GOLD_SET = set(meta[meta["asset_class"] == "gold"].index)
GROUPS = [(DEF_SET, 0.35), (DEF_SET | GOLD_SET, 0.50)]

# 모멘텀 종목 선택 (오늘까지의 일별 데이터)
stock_pool = [t for t in meta[meta["asset_class"].isin(["kr_stock", "us_stock"])].index
              if t in daily.columns]
rows = {}
for tk in stock_pool:
    s = daily[tk].dropna()
    if len(s) < 280 or mret[tk].notna().sum() < EST_MONTHS:
        continue
    p = s.iloc[-1]
    rows[tk] = dict(mom_12_1=p / s.iloc[-252] - 1 - (p / s.iloc[-21] - 1),
                    mom_6=s.iloc[-21] / s.iloc[-126] - 1,
                    vol_60=s.pct_change().iloc[-60:].std() * np.sqrt(252),
                    region=meta.loc[tk, "asset_class"])
f = pd.DataFrame(rows).T
picks = []
for region in ["kr_stock", "us_stock"]:
    g = f[f["region"] == region].astype({c: float for c in ["mom_12_1", "mom_6", "vol_60"]})
    z = lambda x: (x - x.mean()) / x.std()
    score = 0.5 * z(g["mom_12_1"]) + 0.3 * z(g["mom_6"]) - 0.2 * z(g["vol_60"])
    picks += score.nlargest(TOP_N).index.tolist()

universe = ETF_SLEEVE + picks
window = mret[universe].iloc[-EST_MONTHS:].dropna(axis=1)
universe = window.columns.tolist()
caps = pd.Series({t: CAPS[meta.loc[t, "asset_class"]] for t in universe})
rf_m = rf.iloc[-12:].mean()

out = pd.DataFrame({
    "MVO_MaxSharpe": models.max_sharpe(window, rf_monthly=rf_m, caps=caps, groups=GROUPS),
    "RiskParity": models.risk_parity(window, caps=caps, groups=GROUPS),
    "HRP": models.hrp(window, caps=caps, groups=GROUPS),
    "EqualWeight": models.equal_weight(window, caps=caps, groups=GROUPS),
})
out["name"] = [meta.loc[t, "name"] for t in out.index]
out["asset_class"] = [meta.loc[t, "asset_class"] for t in out.index]
out.to_csv("data/final_weights.csv", encoding="utf-8-sig")

print(f"기준일: {daily.index.max().date()} | 모멘텀 선택: {[meta.loc[t,'name'] for t in picks]}\n")
print(out[["name", "asset_class", "MVO_MaxSharpe", "RiskParity", "HRP", "EqualWeight"]]
      .sort_values("EqualWeight", ascending=False)
      .to_string(float_format=lambda v: f"{v:.3f}"))
