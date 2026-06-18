# -*- coding: utf-8 -*-
"""Rolling out-of-sample 백테스트 (월간 리밸런싱, 원화 기준).

- 종목 선택 look-ahead 제거: 매 리밸런싱 시점의 데이터만으로 모멘텀 스코어 산출,
  지역별 상위 5종목을 그 달의 유니버스에 편입
- 신생 ETF는 동일 지수 추종 자산으로 수익률 프록시 연장 (공분산 추정용)
- 거래비용: ETF 편도 0.10%, 개별주 편도 0.25%
"""
import os
import warnings
import numpy as np
import pandas as pd
import models

warnings.filterwarnings("ignore")

EST_MONTHS = int(os.environ.get("EST_MONTHS", 36))  # 공분산/기대수익 추정 윈도우
TOP_N = int(os.environ.get("TOP_N", 5))             # 지역별 모멘텀 상위 종목 수
TAG = os.environ.get("TAG", "")                     # 민감도 테스트용 태그
BT_START = os.environ.get("BT_START", "2018-12-31")  # 첫 리밸런싱 (수익률은 2019-01부터)

ETF_SLEEVE = ["069500.KS", "229200.KS", "360750.KS", "133690.KS", "458730.KS",
              "411060.KS", "261240.KS", "153130.KS", "148070.KS", "TLT"]
PROXY = {"411060.KS": "GLD", "360750.KS": "SPY", "458730.KS": "SCHD", "261240.KS": "_FX"}
CAPS = {"kr_stock": 0.15, "us_stock": 0.15,
        "kr_equity_etf": 0.30, "us_equity_etf": 0.30, "us_equity_krx": 0.30,
        "gold": 0.20, "cash_usd": 0.20, "cash_krw": 0.25, "bond": 0.20}
COST = {"stock": 0.0025, "etf": 0.0010}
RF_TICKER = "153130.KS"  # KODEX 단기채권 = 원화 현금 수익률

# ---------- 데이터 ----------
prices = pd.read_csv("data/prices_krw.csv", index_col=0, parse_dates=True)
raw = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
meta = pd.read_csv("data/meta.csv", encoding="utf-8-sig").set_index("ticker")
prices["_FX"] = raw["KRW=X"]

# 신생 ETF 수익률 프록시 연장 → 합성 가격 재구성
daily = prices.copy()
for tk, proxy in PROXY.items():
    s, p = daily[tk], daily[proxy]
    first = s.first_valid_index()
    pre = p.loc[:first].pct_change().iloc[:-1]          # 상장 전 구간은 프록시 수익률
    post = s.loc[first:].pct_change()
    ext = pd.concat([pre, post]).dropna()
    daily[tk] = (1 + ext).cumprod() * 100
daily = daily.drop(columns=["_FX"])

monthly = daily.resample("ME").last()
if daily.index.max() < monthly.index[-1]:   # 진행 중인 달은 제외
    monthly = monthly.iloc[:-1]
mret = monthly.pct_change()
rf = mret[RF_TICKER]

# 그룹 제약: 방어자산(현금+채권) 합계 35% 이하, 방어+금 50% 이하 → 주식 50% 이상
DEF_SET = set(meta[meta["asset_class"].isin(["cash_krw", "cash_usd", "bond"])].index)
GOLD_SET = set(meta[meta["asset_class"] == "gold"].index)
GROUPS = [(DEF_SET, 0.35), (DEF_SET | GOLD_SET, 0.50)]

stock_pool = meta[meta["asset_class"].isin(["kr_stock", "us_stock"])].index.tolist()
stock_pool = [t for t in stock_pool if t in daily.columns]

def caps_for(universe):
    return pd.Series({t: CAPS[meta.loc[t, "asset_class"]] for t in universe})

def cost_vec(universe):
    return pd.Series({t: COST["stock"] if meta.loc[t, "asset_class"] in ("kr_stock", "us_stock")
                      else COST["etf"] for t in universe})

def momentum_select(t):
    """시점 t까지의 데이터만으로 지역별 모멘텀 상위 TOP_N 선택."""
    px = daily.loc[:t]
    rows = {}
    for tk in stock_pool:
        s = px[tk].dropna()
        if len(s) < 280 or mret[tk].loc[:t].notna().sum() < EST_MONTHS:
            continue
        p = s.iloc[-1]
        rows[tk] = dict(
            mom_12_1=p / s.iloc[-252] - 1 - (p / s.iloc[-21] - 1),
            mom_6=s.iloc[-21] / s.iloc[-126] - 1,
            vol_60=s.pct_change().iloc[-60:].std() * np.sqrt(252),
            region=meta.loc[tk, "asset_class"])
    f = pd.DataFrame(rows).T
    if f.empty:
        return []
    picks = []
    for region in ["kr_stock", "us_stock"]:
        g = f[f["region"] == region].astype({"mom_12_1": float, "mom_6": float, "vol_60": float})
        z = lambda x: (x - x.mean()) / x.std()
        score = 0.5 * z(g["mom_12_1"]) + 0.3 * z(g["mom_6"]) - 0.2 * z(g["vol_60"])
        picks += score.nlargest(TOP_N).index.tolist()
    return picks

# ---------- 백테스트 루프 ----------
MODELS = {"MVO_MaxSharpe": models.max_sharpe, "RiskParity": models.risk_parity,
          "HRP": models.hrp, "EqualWeight": models.equal_weight}

rebal_dates = mret.loc[BT_START:].index[:-1]   # 마지막 월말은 보유만
hold_months = mret.loc[BT_START:].index[1:]

results = {m: [] for m in MODELS}
weights_hist = {m: {} for m in MODELS}
turnover_hist = {m: [] for m in MODELS}
prev_w = {m: None for m in MODELS}

for t, t_next in zip(rebal_dates, hold_months):
    universe = ETF_SLEEVE + momentum_select(t)
    window = mret[universe].loc[:t].iloc[-EST_MONTHS:].dropna(axis=1)
    universe = window.columns.tolist()
    caps, costs = caps_for(universe), cost_vec(universe)
    r_next = mret[universe].loc[t_next].fillna(0)

    for name, fn in MODELS.items():
        kw = {"rf_monthly": rf.loc[:t].iloc[-12:].mean()} if name == "MVO_MaxSharpe" else {}
        w = fn(window, caps=caps, groups=GROUPS, **kw)
        pw = prev_w[name]
        if pw is None:
            dw = w
        else:
            drift = (pw * (1 + mret[pw.index].loc[t].fillna(0)))
            drift /= drift.sum()
            dw = (w - drift.reindex(w.index).fillna(0)).abs()
            dropped = drift.index.difference(w.index)
            dw = pd.concat([dw, drift[dropped]])
        tc = (dw * cost_vec(dw.index)).sum()
        gross = (w * r_next).sum()
        results[name].append(gross - tc)
        turnover_hist[name].append(dw.sum() / 2)
        weights_hist[name][t] = w
        prev_w[name] = w

bt = pd.DataFrame(results, index=hold_months)

# ---------- 벤치마크 ----------
bt["BM_KODEX200"] = mret["069500.KS"].loc[hold_months]
bt["BM_SP500_KRW"] = mret["SPY"].loc[hold_months]
b6040 = (0.30 * mret["069500.KS"] + 0.30 * mret["SPY"]
         + 0.25 * mret["148070.KS"] + 0.15 * mret["153130.KS"])
bt["BM_6040_Global"] = b6040.loc[hold_months]

# ---------- 성과지표 ----------
def metrics(r, rf_r):
    cum = (1 + r).cumprod()
    yrs = len(r) / 12
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    ex = r - rf_r.loc[r.index]
    sharpe = ex.mean() / r.std() * np.sqrt(12)
    downside = r[r < 0].std() * np.sqrt(12)
    sortino = ex.mean() * 12 / downside if downside > 0 else np.nan
    mdd = (cum / cum.cummax() - 1).min()
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino,
                MDD=mdd, Calmar=cagr / abs(mdd))

summary = pd.DataFrame({c: metrics(bt[c].dropna(), rf) for c in bt.columns}).T
for m in MODELS:
    summary.loc[m, "AvgTurnover"] = np.mean(turnover_hist[m])

print(f"백테스트: {hold_months[0].date()} ~ {hold_months[-1].date()} ({len(hold_months)}개월), 원화 기준\n")
print("=== 전체 기간 성과 ===")
print(summary.to_string(float_format=lambda v: f"{v:+.3f}"))

# ---------- 스트레스 구간 ----------
windows = {"COVID(2020.02-03)": ("2020-02", "2020-03"),
           "2022약세장(2022.01-12)": ("2022-01", "2022-12"),
           "최근12개월": ("2025-06", "2026-05")}
print("\n=== 스트레스/구간별 누적수익률 ===")
stress = {}
for label, (a, b) in windows.items():
    seg = bt.loc[a:b]
    stress[label] = (1 + seg).prod() - 1
print(pd.DataFrame(stress).to_string(float_format=lambda v: f"{v:+.3f}"))

# ---------- 저장 ----------
suffix = f"_{TAG}" if TAG else ""
bt.to_csv(f"data/bt_returns{suffix}.csv", encoding="utf-8-sig")
summary.to_csv(f"data/bt_summary{suffix}.csv", encoding="utf-8-sig")
if not TAG:
    for m in MODELS:
        pd.DataFrame(weights_hist[m]).T.to_csv(f"data/bt_weights_{m}.csv", encoding="utf-8-sig")

# 최신 시점 가중치 (최종 포트폴리오 후보)
print("\n=== 최신 리밸런싱 가중치 (3% 이상) ===")
for m in ["MVO_MaxSharpe", "RiskParity", "HRP"]:
    w = pd.DataFrame(weights_hist[m]).T.iloc[-1].dropna()
    w = w[w >= 0.03].sort_values(ascending=False)
    names = [f"{meta.loc[t, 'name']}({t.split('.')[0]}) {v:.1%}" for t, v in w.items()]
    print(f"\n[{m}] " + ", ".join(names))
