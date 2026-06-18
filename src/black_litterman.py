# -*- coding: utf-8 -*-
"""Black-Litterman: 리서치 뷰(2026-06-10)를 사전분포(EW 앵커 균형수익률)에 결합 → Max Sharpe.

- 모든 뷰는 원화 기준 (미국 자산 = USD 뷰 + 환율 뷰 -5%p 합성)
- 신뢰도 → Ω 매핑: ω_i = τ · pΣp' · (1-c)/c  (Idzorek 방식)
- 유니버스 = ETF 슬리브 + 모멘텀 픽 + 리서치 확신 종목(한화에어로, 한화오션)
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import models

warnings.filterwarnings("ignore")
# 공분산 60개월: 멜트업만 담긴 36개월 윈도우는 금-주식 상관이 양(+)으로 잡혀
# 주식 뷰 하향이 금 사후수익률까지 끌어내리는 왜곡 발생 (2차 실행에서 확인)
EST_MONTHS, TAU = 60, 0.025

# ---- 데이터 (final_weights.py와 동일한 전처리) ----
prices = pd.read_csv("data/prices_krw.csv", index_col=0, parse_dates=True)
raw = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
meta = pd.read_csv("data/meta.csv", encoding="utf-8-sig").set_index("ticker")
prices["_FX"] = raw["KRW=X"]
daily = prices.copy()
for tk, proxy in {"411060.KS": "GLD", "360750.KS": "SPY", "458730.KS": "SCHD", "261240.KS": "_FX"}.items():
    s, p = daily[tk], daily[proxy]
    first = s.first_valid_index()
    pre = p.loc[:first].pct_change().iloc[:-1]
    daily[tk] = (1 + pd.concat([pre, s.loc[first:].pct_change()]).dropna()).cumprod() * 100
daily = daily.drop(columns=["_FX"])
monthly = daily.resample("ME").last()
if daily.index.max() < monthly.index[-1]:
    monthly = monthly.iloc[:-1]
mret = monthly.pct_change()
rf_m = mret["153130.KS"].iloc[-12:].mean()

fw = pd.read_csv("data/final_weights.csv", index_col=0, encoding="utf-8-sig")
universe = fw.index.tolist() + ["012450.KS", "042660.KS"]   # 방산·조선 추가
window = mret[universe].iloc[-EST_MONTHS:].dropna(axis=1)
universe = window.columns.tolist()

CAPS = {"kr_stock": 0.15, "us_stock": 0.15,
        "kr_equity_etf": 0.30, "us_equity_etf": 0.30, "us_equity_krx": 0.30,
        "gold": 0.20, "cash_usd": 0.20, "cash_krw": 0.25, "bond": 0.20}
caps = pd.Series({t: CAPS[meta.loc[t, "asset_class"]] for t in universe})
DEF_SET = set(meta[meta["asset_class"].isin(["cash_krw", "cash_usd", "bond"])].index)
GOLD_SET = set(meta[meta["asset_class"] == "gold"].index)
GROUPS = [(DEF_SET, 0.35), (DEF_SET | GOLD_SET, 0.50)]

cov = models.shrink_cov(window)
Sigma = cov.values

# ---- 사전분포: EW 앵커의 균형 기대수익률 ----
# delta=2.5 (시장 표준). 최근 36개월 멜트업 구간으로 추정하면 delta가 비정상적으로
# 커져 사전수익률이 부풀고 사후분포가 왜곡됨 (1차 실행에서 확인)
w0 = models.equal_weight(window, caps=caps, groups=GROUPS).values
delta = 2.5
pi = delta * Sigma @ w0                       # 월간 초과수익률

# ---- 뷰 (연간 총수익률 %, 원화 기준, 신뢰도) ----
VIEWS = [
    (["360750.KS"],              [1.0],        0.0,  0.30, "미국 S&P500 (USD +5% / 환율 -5%)"),
    (["133690.KS"],              [1.0],        1.0,  0.25, "나스닥100 (USD +6% / 환율 -5%)"),
    (["XOM", "458730.KS"],       [0.5, 0.5],   2.0,  0.30, "에너지·배당가치 (USD +7% / 환율 -5%)"),
    (["069500.KS"],              [1.0],        4.0,  0.30, "KOSPI"),
    (["000660.KS", "005930.KS"], [0.5, 0.5],   8.0,  0.30, "한국 반도체"),
    (["012450.KS", "042660.KS"], [0.5, 0.5],  10.0,  0.40, "한국 방산·조선"),
    (["411060.KS"],              [1.0],        7.0,  0.40, "금 (USD +12% / 환율 -5%)"),
    (["261240.KS"],              [1.0],       -1.5,  0.40, "달러선물 (환율 -5% + 캐리 +3.5%)"),
    (["TLT"],                    [1.0],       -5.0,  0.35, "미국 장기채 (USD 0% / 환율 -5%)"),
    (["153130.KS"],              [1.0],        3.0,  0.85, "원화 단기채"),
    (["148070.KS"],              [1.0],        2.0,  0.30, "한국 국고채10년"),
]
rows, q, om, labels = [], [], [], []
for assets, wts, ann_pct, conf, label in VIEWS:
    if not all(a in universe for a in assets):
        continue
    p = np.zeros(len(universe))
    for a, wt in zip(assets, wts):
        p[universe.index(a)] = wt
    rows.append(p)
    q.append(ann_pct / 100 / 12 - rf_m)       # 월간 초과수익률로 변환
    om.append(TAU * p @ Sigma @ p * (1 - conf) / conf)
    labels.append(label)
P, q, Omega = np.array(rows), np.array(q), np.diag(om)

# ---- 사후 기대수익률 ----
tS_inv = np.linalg.inv(TAU * Sigma)
M = np.linalg.inv(tS_inv + P.T @ np.linalg.inv(Omega) @ P)
mu_bl = M @ (tS_inv @ pi + P.T @ np.linalg.inv(Omega) @ q)

# ---- Max Sharpe (사후 뷰 기반) ----
n = len(universe)
def neg_sharpe(w):
    return -(w @ mu_bl) / max(np.sqrt(w @ Sigma @ w), 1e-12)
cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + models.group_constraints(window.columns, GROUPS)
res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
               bounds=[(0, c) for c in caps.values], constraints=cons,
               options={"maxiter": 500, "ftol": 1e-10})
w_bl = pd.Series(res.x, index=universe).clip(lower=0)
w_bl = w_bl / w_bl.sum()

out = pd.DataFrame({"pi_ann%": (pi + rf_m) * 12 * 100, "mu_bl_ann%": (mu_bl + rf_m) * 12 * 100,
                    "weight": w_bl}, index=universe)
out["name"] = [meta.loc[t, "name"] for t in universe]
out["asset_class"] = [meta.loc[t, "asset_class"] for t in universe]
out.to_csv("data/bl_weights.csv", encoding="utf-8-sig")

port_vol = np.sqrt(w_bl.values @ Sigma @ w_bl.values) * np.sqrt(12)
port_mu = (w_bl.values @ mu_bl + rf_m) * 12
print(f"적용된 뷰 {len(labels)}개: {labels}\n")
print(out[["name", "asset_class", "pi_ann%", "mu_bl_ann%", "weight"]]
      .sort_values("weight", ascending=False).to_string(float_format=lambda v: f"{v:+.3f}"))
print(f"\nBL 포트폴리오 기대수익률(연) {port_mu:+.1%}, 변동성(연) {port_vol:.1%}")
